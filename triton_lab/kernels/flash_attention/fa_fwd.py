# pyright: reportUnreachable=false, reportArgumentType=false
# 1) @triton.jit 的 __call__ 永远 raise，pyright 会把后面的代码判成 unreachable；
# 2) tl.constexpr 是给 Triton 编译器看的标记，运行时就是 int/bool，
#    pyright 会把它当成独立类型，误报“实参与 constexpr 不兼容”。

import os

import torch
import torch_npu
import triton
import triton.backends.ascend.runtime
import triton.language as tl
from torch_npu.npu import get_device_properties

# 调试用：STACK 批处理宽度 —— 每轮一次性处理 STACK_NUM 个 BLOCK_K 块。
#   FA_STACK=1   -> 每轮 1 个块（每轮固定开销最高）
#   FA_STACK=2/4 -> 每轮 2/4 个块（摊薄每轮固定的同步/循环/rescale 开销）
#
# 注意：
#   1. @triton.jit 只能读取【标注为 tl.constexpr】的模块全局变量，否则报
#      "Cannot access global variable ... from within @jit'ed function"；
#   2. host 侧（torch.empty 的 shape）需要普通 Python int，不能传 constexpr 对象。
#   ⇒ 所以这里同时给出两个名字，值来自同一个环境变量。
_STACK = int(os.environ.get("FA_STACK", "4"))
if _STACK <= 0:
    raise ValueError(f"FA_STACK 必须是正整数，当前 {_STACK!r}")
# ⚠️ 必须用 tl.constexpr(...) 构造：Triton 的 _is_constexpr() 只看【值的类型】，
#    注解 `x: tl.constexpr = 4` 存进去的仍是普通 int，会被拒绝（实测确认）。
_STACK_NUM = tl.constexpr(_STACK)

# 注意：BKV=256 在 N 不能被 256 整除时会触发
# "load2d 写地址越界" 的 aicore 异常（507015），尾块修复前用 _prune_bk256 按形状屏蔽。
# 32x256 已实测崩溃；64x256 在 S=1000 编译 UB 超限。
_TILES = [
    # (128, 128) 已移除：两段拆分后每段各有一份 k/v 内联副本，UB 从 192KB
    # 涨到需要 323.6KB（1.68x 超限），在 N=4096 上 bishengir 直接 ub overflow。
    # 且这个 config 一旦编译失败，Ascend autotuner 会把【整批】判成
    # "No valid triton configs. NoneType: None"，掩盖真实错误。
    # 想单独验证请用 FAT_TILES=128x128。
    (64, 256),  # ★ N_k 能被 256 整除时启用（实测比 96x128 快 18%）
    (96, 128),
    (64, 128),
    (32, 128),
    (32, 64),
]

# 调试用：FAT_TILES 覆盖候选列表，避免为了隔离问题反复改代码。
#   FAT_TILES=96x128              -> 单 config（拿完整 traceback）
#   FAT_TILES=96x128,64x128       -> 指定若干 config
#   FAT_TILES=64x256,128x128      -> 重新试被注释掉的 tile
_env_tiles = os.environ.get("FAT_TILES")
if _env_tiles:
    _TILES = [tuple(int(v) for v in t.split("x")) for t in _env_tiles.split(",") if t]
    for _bq, _bk in _TILES:
        if _bq <= 0 or _bk <= 0:
            raise ValueError(
                f"FAT_TILES 格式错误: {_env_tiles!r}（应为 96x128,64x128）"
            )

# 调试用：循环展开因子。AIC 有 55.5% 的时间在跑标量（循环控制 + 地址计算），
# 展开循环能让每份工作量只付一次循环控制开销。
#   FA_UNROLL=1（默认，不展开） / 2 / 4
# 注意：@triton.jit 函数只能读取【标注为 tl.constexpr】的模块全局变量，
#       否则报 "Cannot access global variable ... from within @jit'ed function"。
_UNROLL = tl.constexpr(int(os.environ.get("FA_UNROLL", "1")))


def _pick_tile(n_q, n_k):
    """host 侧手工选 tile（不用 autotune）。

    为什么不用 @triton.autotune：Ascend 的 autotuner 在批量 benchmark 多个
    config 时会静默丢掉需要更多 UB 的 config。实测 (64,256) 直调 4.146ms /
    33.2 TFLOPS，但放进 autotune 候选后必然丢失（先报 "No valid triton
    configs. NoneType: None"，单 config 时报 IndexError），最终只剩 (96,128)
    4.908ms / 28.0 TFLOPS —— 白丢 18%。
    MindSpeed 的 FA 也不用 autotune，同样用手写常数/手写选择器。

    FAT_TILES 环境变量仍可强制覆盖整个表，取第一个。
    """
    if _env_tiles:
        return _TILES[0]

    # (64,256)：一次迭代覆盖 256 列 KV，实测在整除形状上最快。
    # BKV=256 的尾块 bug 未修（N 非整除时 aicore 507015），所以只在整除时使用。
    if n_k % 256 == 0:
        return 64, 256
    return 96, 128


@triton.jit
def fla_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    s_ptr,
    p_ptr,
    stride_qb,
    stride_qq,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vh,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_oh,
    stride_od,
    stride_sc,
    stride_sq,
    stride_sk,
    stride_pc,
    stride_pq,
    stride_pk,
    batch,
    head,
    scale,
    N_q: tl.constexpr,
    N_k: tl.constexpr,
    delta_qk: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    STACK_NUM: tl.constexpr = _STACK_NUM
    EVEN_Q = N_q % BLOCK_Q == 0
    blocks_q = tl.cdiv(N_q, BLOCK_Q)
    taskNum = batch * head * blocks_q
    coreIdx = tl.program_id(0)
    coreNum = tl.num_programs(0)
    for task in range(coreIdx, taskNum, coreNum):
        b = task // (head * blocks_q)
        i = blocks_q - 1 - ((task % (head * blocks_q)) // head)
        h = (task % (head * blocks_q)) % head
        q_base = q_ptr + b * stride_qb + h * stride_qh
        o_base = o_ptr + b * stride_ob + h * stride_oh
        k_base = k_ptr + b * stride_kb + h * stride_kh
        v_base = v_ptr + b * stride_vb + h * stride_vh
        q_block_ptr = tl.make_block_ptr(
            q_base,
            (N_q, HEAD_DIM),
            strides=(stride_qq, stride_qd),
            offsets=(i * BLOCK_Q, 0),
            block_shape=(BLOCK_Q, HEAD_DIM),
            order=(1, 0),
        )
        o_block_ptr = tl.make_block_ptr(
            o_base,
            (N_q, HEAD_DIM),
            strides=(stride_oq, stride_od),
            offsets=(i * BLOCK_Q, 0),
            block_shape=(BLOCK_Q, HEAD_DIM),
            order=(1, 0),
        )
        mi = tl.full([BLOCK_Q], -1e38, dtype=tl.float32)
        li = tl.zeros([BLOCK_Q], dtype=tl.float32)
        acc = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)

        if EVEN_Q:
            q = tl.load(q_block_ptr)
        else:
            q = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")

        q = q * scale.to(tl.float16)

        q_start_r = i * BLOCK_Q

        SW: tl.constexpr = BLOCK_K * STACK_NUM
        end = N_k
        if IS_CAUSAL:
            end = tl.minimum(q_start_r + BLOCK_Q + delta_qk, N_k)
        T_k = tl.cdiv(end, BLOCK_K)
        loopCnt = tl.cdiv(T_k, STACK_NUM)
        # 子块宽度固定为 BLOCK_K 后，最后一个 stack 可能整体越过 N_k
        # （即使 N_k % BLOCK_K == 0，例如 N_k=384/BN=64 时第 2 个 stack 覆盖 256..511），
        # 所以边界保护必须按 STACK 粒度判断。
        EVEN_ST: tl.constexpr = N_k % SW == 0

        q_idx = q_start_r + tl.arange(0, BLOCK_Q)

        for stack_idx in tl.range(0, loopCnt, loop_unroll_factor=_UNROLL):
            base = stack_idx * SW
            IS_DIAG = base + SW - 1 >= q_start_r + delta_qk

            offset_base = base + tl.arange(0, BLOCK_K)

            k_block_ptr = tl.make_block_ptr(
                k_base,
                (N_k, HEAD_DIM),
                strides=(stride_kk, stride_kd),
                offsets=(base, 0),
                block_shape=(BLOCK_K, HEAD_DIM),
                order=(1, 0),
            )
            v_block_ptr = tl.make_block_ptr(
                v_base,
                (N_k, HEAD_DIM),
                strides=(stride_vk, stride_vd),
                offsets=(base, 0),
                block_shape=(BLOCK_K, HEAD_DIM),
                order=(1, 0),
            )
            s_block_ptr = tl.make_block_ptr(
                s_ptr + coreIdx * stride_sc,
                (BLOCK_Q, BLOCK_K * STACK_NUM),
                strides=(stride_sq, stride_sk),
                offsets=(0, 0),
                block_shape=(BLOCK_Q, BLOCK_K),
                order=(1, 0),
            )
            p_block_ptr = tl.make_block_ptr(
                p_ptr + coreIdx * stride_pc,
                (BLOCK_Q, BLOCK_K * STACK_NUM),
                strides=(stride_pq, stride_pk),
                offsets=(0, 0),
                block_shape=(BLOCK_Q, BLOCK_K),
                order=(1, 0),
            )
            ori_s_block_ptr = s_block_ptr
            ori_p_block_ptr = p_block_ptr
            # ---- 阶段 1：连续发射 4 个 QK^T（AIC 侧聚类，中间不夹 Vector 算子）----
            if EVEN_ST:
                for si in range(STACK_NUM):
                    k = tl.load(k_block_ptr)
                    s_cube = tl.dot(q, tl.trans(k))
                    tl.store(s_block_ptr, s_cube)
                    k_block_ptr = tl.advance(k_block_ptr, (BLOCK_K, 0))
                    s_block_ptr = tl.advance(s_block_ptr, (0, BLOCK_K))
            else:
                for si in range(STACK_NUM):
                    k = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
                    s_cube = tl.dot(q, tl.trans(k))
                    tl.store(s_block_ptr, s_cube)
                    k_block_ptr = tl.advance(k_block_ptr, (BLOCK_K, 0))
                    s_block_ptr = tl.advance(s_block_ptr, (0, BLOCK_K))

            s_block_ptr = ori_s_block_ptr
            m_prev = mi
            # ---- 阶段 2：4 个块共享同一个联合行最大值（acc 每 4 块只缩放 1 次）----
            for si in range(STACK_NUM):
                s_vec = tl.load(s_block_ptr)
                if not EVEN_ST:
                    s_vec = tl.where(
                        (offset_base + si * BLOCK_K)[None, :] < N_k, s_vec, -1.0e6
                    )
                if IS_CAUSAL and IS_DIAG:
                    s_vec = tl.where(
                        (offset_base + si * BLOCK_K)[None, :]
                        <= q_idx[:, None] + delta_qk,
                        s_vec,
                        -1.0e6,
                    )
                mi = tl.maximum(mi, tl.max(s_vec, -1))
                s_block_ptr = tl.advance(s_block_ptr, (0, BLOCK_K))

            s_block_ptr = ori_s_block_ptr

            factor = tl.exp(m_prev - mi)
            acc = factor[:, None] * acc
            li = factor * li

            for si in range(STACK_NUM):
                s_vec = tl.load(s_block_ptr)
                if not EVEN_ST:
                    s_vec = tl.where(
                        (offset_base + si * BLOCK_K)[None, :] < N_k, s_vec, -1.0e6
                    )
                if IS_CAUSAL and IS_DIAG:
                    s_vec = tl.where(
                        (offset_base + si * BLOCK_K)[None, :]
                        <= q_idx[:, None] + delta_qk,
                        s_vec,
                        -1.0e6,
                    )
                p_vec = tl.exp(s_vec - mi[:, None])
                li = li + tl.sum(p_vec, -1)
                p_vec = p_vec.to(tl.float16)
                tl.store(p_block_ptr, p_vec)
                s_block_ptr = tl.advance(s_block_ptr, (0, BLOCK_K))
                p_block_ptr = tl.advance(p_block_ptr, (0, BLOCK_K))

            s_block_ptr = ori_s_block_ptr
            p_block_ptr = ori_p_block_ptr

            # ---- 阶段 3：连续发射 4 个 PV，acc 全程驻留 L0C 累加 ----
            if EVEN_ST:
                for si in range(STACK_NUM):
                    p_cube = tl.load(p_block_ptr)
                    v = tl.load(v_block_ptr)
                    acc = tl.dot(p_cube, v, acc)
                    p_block_ptr = tl.advance(p_block_ptr, (0, BLOCK_K))
                    v_block_ptr = tl.advance(v_block_ptr, (BLOCK_K, 0))
            else:
                for si in range(STACK_NUM):
                    p_cube = tl.load(p_block_ptr)
                    v = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")
                    acc = tl.dot(p_cube, v, acc)
                    p_block_ptr = tl.advance(p_block_ptr, (0, BLOCK_K))
                    v_block_ptr = tl.advance(v_block_ptr, (BLOCK_K, 0))

        acc = acc / li[:, None]
        if EVEN_Q:
            tl.store(o_block_ptr, acc.to(o_ptr.dtype.element_ty))
        else:
            tl.store(o_block_ptr, acc.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))


def fla_fwd(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, causal=False):
    assert Q.stride(3) == K.stride(3) == V.stride(3) == 1
    assert Q.stride(2) == K.stride(2) == V.stride(2)
    batch = Q.size(dim=0)
    N_q = Q.size(dim=1)
    N_k = K.size(dim=1)
    head = Q.size(dim=2)
    head_dim = Q.size(dim=3)

    O = torch.zeros_like(Q, device=Q.device)

    scale = head_dim**-0.5

    # 带 tl.dot 的 kernel 用 AI Core 数；纯 Vector 算子用 vector_core_num
    cores = get_device_properties(0).cube_core_num

    delta_qk = N_k - N_q

    BLOCK_Q, BLOCK_K = _pick_tile(N_q, N_k)

    grid = (min(batch * head * triton.cdiv(N_q, BLOCK_Q), cores),)
    n_progs = int(grid[0])
    S = torch.empty([n_progs, BLOCK_Q, BLOCK_K * _STACK], device=Q.device)
    P = torch.empty(
        [n_progs, BLOCK_Q, BLOCK_K * _STACK], device=Q.device, dtype=torch.float16
    )

    fla_fwd_kernel[grid](
        Q,
        K,
        V,
        O,
        S,
        P,
        Q.stride(0),
        Q.stride(1),
        Q.stride(2),
        Q.stride(3),
        K.stride(0),
        K.stride(1),
        K.stride(2),
        K.stride(3),
        V.stride(0),
        V.stride(1),
        V.stride(2),
        V.stride(3),
        O.stride(0),
        O.stride(1),
        O.stride(2),
        O.stride(3),
        S.stride(0),
        S.stride(1),
        S.stride(2),
        P.stride(0),
        P.stride(1),
        P.stride(2),
        batch,
        head,
        scale,
        N_q=N_q,
        N_k=N_k,
        delta_qk=delta_qk,
        BLOCK_Q=BLOCK_Q,
        BLOCK_K=BLOCK_K,
        HEAD_DIM=head_dim,
        IS_CAUSAL=causal,
    )
    return O
