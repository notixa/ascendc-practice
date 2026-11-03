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
_UNROLL: tl.constexpr = int(os.environ.get("FA_UNROLL", "1"))


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
    l_ptr,
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
    stride_lb,
    stride_lh,
    stride_lq,
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
    EVEN_Q = N_q % BLOCK_Q == 0
    EVEN_K = N_k % BLOCK_K == 0
    blocks_q = tl.cdiv(N_q, BLOCK_Q)
    taskNum = batch * head * blocks_q
    taskIdx = tl.program_id(0)
    coreNum = tl.num_programs(0)
    for task in range(taskIdx, taskNum, coreNum):
        b = task // (head * blocks_q)
        i = blocks_q - 1 - ((task % (head * blocks_q)) // head)
        h = (task % (head * blocks_q)) % head
        q_base = q_ptr + b * stride_qb + h * stride_qh
        o_base = o_ptr + b * stride_ob + h * stride_oh
        l_base = l_ptr + b * stride_lb + h * stride_lh
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
        l_block_ptr = tl.make_block_ptr(
            l_base,
            (N_q,),
            strides=(stride_lq,),
            offsets=(i * BLOCK_Q,),
            block_shape=(BLOCK_Q,),
            order=(0,),
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

        STACK_NUM: tl.constexpr = 4
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

            o0 = base + tl.arange(0, BLOCK_K)
            o1 = o0 + BLOCK_K
            o2 = o1 + BLOCK_K
            o3 = o2 + BLOCK_K

            kp0 = tl.make_block_ptr(
                k_base,
                (N_k, HEAD_DIM),
                strides=(stride_kk, stride_kd),
                offsets=(base + 0 * BLOCK_K, 0),
                block_shape=(BLOCK_K, HEAD_DIM),
                order=(1, 0),
            )
            kp1 = tl.make_block_ptr(
                k_base,
                (N_k, HEAD_DIM),
                strides=(stride_kk, stride_kd),
                offsets=(base + 1 * BLOCK_K, 0),
                block_shape=(BLOCK_K, HEAD_DIM),
                order=(1, 0),
            )
            kp2 = tl.make_block_ptr(
                k_base,
                (N_k, HEAD_DIM),
                strides=(stride_kk, stride_kd),
                offsets=(base + 2 * BLOCK_K, 0),
                block_shape=(BLOCK_K, HEAD_DIM),
                order=(1, 0),
            )
            kp3 = tl.make_block_ptr(
                k_base,
                (N_k, HEAD_DIM),
                strides=(stride_kk, stride_kd),
                offsets=(base + 3 * BLOCK_K, 0),
                block_shape=(BLOCK_K, HEAD_DIM),
                order=(1, 0),
            )

            vp0 = tl.make_block_ptr(
                v_base,
                (N_k, HEAD_DIM),
                strides=(stride_vk, stride_vd),
                offsets=(base + 0 * BLOCK_K, 0),
                block_shape=(BLOCK_K, HEAD_DIM),
                order=(1, 0),
            )
            vp1 = tl.make_block_ptr(
                v_base,
                (N_k, HEAD_DIM),
                strides=(stride_vk, stride_vd),
                offsets=(base + 1 * BLOCK_K, 0),
                block_shape=(BLOCK_K, HEAD_DIM),
                order=(1, 0),
            )
            vp2 = tl.make_block_ptr(
                v_base,
                (N_k, HEAD_DIM),
                strides=(stride_vk, stride_vd),
                offsets=(base + 2 * BLOCK_K, 0),
                block_shape=(BLOCK_K, HEAD_DIM),
                order=(1, 0),
            )
            vp3 = tl.make_block_ptr(
                v_base,
                (N_k, HEAD_DIM),
                strides=(stride_vk, stride_vd),
                offsets=(base + 3 * BLOCK_K, 0),
                block_shape=(BLOCK_K, HEAD_DIM),
                order=(1, 0),
            )

            # ---- 阶段 1：连续发射 4 个 QK^T（AIC 侧聚类，中间不夹 Vector 算子）----
            if EVEN_ST:
                k0 = tl.load(kp0)
                k1 = tl.load(kp1)
                k2 = tl.load(kp2)
                k3 = tl.load(kp3)
                s0 = tl.dot(q, tl.trans(k0))
                s1 = tl.dot(q, tl.trans(k1))
                s2 = tl.dot(q, tl.trans(k2))
                s3 = tl.dot(q, tl.trans(k3))
            else:
                k0 = tl.load(kp0, boundary_check=(0, 1), padding_option="zero")
                k1 = tl.load(kp1, boundary_check=(0, 1), padding_option="zero")
                k2 = tl.load(kp2, boundary_check=(0, 1), padding_option="zero")
                k3 = tl.load(kp3, boundary_check=(0, 1), padding_option="zero")
                s0 = tl.where(o0[None, :] < N_k, tl.dot(q, tl.trans(k0)), -1.0e6)
                s1 = tl.where(o1[None, :] < N_k, tl.dot(q, tl.trans(k1)), -1.0e6)
                s2 = tl.where(o2[None, :] < N_k, tl.dot(q, tl.trans(k2)), -1.0e6)
                s3 = tl.where(o3[None, :] < N_k, tl.dot(q, tl.trans(k3)), -1.0e6)

            if IS_CAUSAL and IS_DIAG:
                s0 = tl.where(o0[None, :] <= q_idx[:, None] + delta_qk, s0, -1.0e6)
                s1 = tl.where(o1[None, :] <= q_idx[:, None] + delta_qk, s1, -1.0e6)
                s2 = tl.where(o2[None, :] <= q_idx[:, None] + delta_qk, s2, -1.0e6)
                s3 = tl.where(o3[None, :] <= q_idx[:, None] + delta_qk, s3, -1.0e6)

            # ---- 阶段 2：4 个块共享同一个联合行最大值（acc 每 4 块只缩放 1 次）----
            mi0 = tl.maximum(mi, tl.max(s0, -1))
            p0 = tl.exp(s0 - mi0[:, None])
            li0 = tl.sum(p0, -1)

            mi1 = tl.maximum(mi0, tl.max(s1, -1))
            p1 = tl.exp(s1 - mi1[:, None])
            li1 = tl.sum(p1, -1)

            mi2 = tl.maximum(mi1, tl.max(s2, -1))
            p2 = tl.exp(s2 - mi2[:, None])
            li2 = tl.sum(p2, -1)

            mi_new = tl.maximum(mi2, tl.max(s3, -1))
            p3 = tl.exp(s3 - mi_new[:, None])
            li3 = tl.sum(p3, -1)
            p3 = p3.to(tl.float16)

            p0 = (p0 * tl.exp(mi0 - mi_new)[:, None]).to(tl.float16)
            p1 = (p1 * tl.exp(mi1 - mi_new)[:, None]).to(tl.float16)
            p2 = (p2 * tl.exp(mi2 - mi_new)[:, None]).to(tl.float16)

            factor = tl.exp(mi - mi_new)
            li = (
                factor * li
                + li0 * tl.exp(mi0 - mi_new)
                + li1 * tl.exp(mi1 - mi_new)
                + li2 * tl.exp(mi2 - mi_new)
                + li3
            )
            acc = factor[:, None] * acc
            mi = mi_new

            # ---- 阶段 3：连续发射 4 个 PV，acc 全程驻留 L0C 累加 ----
            if EVEN_ST:
                v0 = tl.load(vp0)
                v1 = tl.load(vp1)
                v2 = tl.load(vp2)
                v3 = tl.load(vp3)
            else:
                v0 = tl.load(vp0, boundary_check=(0, 1), padding_option="zero")
                v1 = tl.load(vp1, boundary_check=(0, 1), padding_option="zero")
                v2 = tl.load(vp2, boundary_check=(0, 1), padding_option="zero")
                v3 = tl.load(vp3, boundary_check=(0, 1), padding_option="zero")

            acc = tl.dot(p0, v0, acc)
            acc = tl.dot(p1, v1, acc)
            acc = tl.dot(p2, v2, acc)
            acc = tl.dot(p3, v3, acc)

        acc = acc / li[:, None]
        li = mi + tl.log(li)
        if EVEN_Q:
            tl.store(o_block_ptr, acc.to(o_ptr.dtype.element_ty))
            tl.store(l_block_ptr, li)
        else:
            tl.store(o_block_ptr, acc.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))
            tl.store(l_block_ptr, li, boundary_check=(0,))


def fla_fwd(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, causal=False):
    assert Q.stride(3) == K.stride(3) == V.stride(3) == 1
    assert Q.stride(2) == K.stride(2) == V.stride(2)
    batch = Q.size(dim=0)
    N_q = Q.size(dim=1)
    N_k = K.size(dim=1)
    head = Q.size(dim=2)
    head_dim = Q.size(dim=3)

    O = torch.zeros_like(Q, device=Q.device)
    L = torch.zeros([batch, head, N_q], dtype=torch.float32, device=Q.device)

    scale = head_dim**-0.5

    # 带 tl.dot 的 kernel 用 AI Core 数；纯 Vector 算子用 vector_core_num
    cores = get_device_properties(0).cube_core_num

    delta_qk = N_k - N_q

    BLOCK_Q, BLOCK_K = _pick_tile(N_q, N_k)
    grid = (min(batch * head * triton.cdiv(N_q, BLOCK_Q), cores),)
    fla_fwd_kernel[grid](
        Q,
        K,
        V,
        O,
        L,
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
        L.stride(0),
        L.stride(1),
        L.stride(2),
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
    return O, L
