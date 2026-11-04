# pyright: reportUnreachable=false, reportArgumentType=false
# 1) @triton.jit 的 __call__ 永远 raise，pyright 会把后面的代码判成 unreachable；
# 2) tl.constexpr 是给 Triton 编译器看的标记，运行时就是 int/bool，
#    pyright 会把它当成独立类型，误报“实参与 constexpr 不兼容”。

import os

import torch
import torch_npu
import triton
import triton.backends.ascend.runtime
import triton.extension.buffer.language as bl
import triton.language as tl
import triton.language.extra.cann.extension as al
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


# ---------------------------------------------------------------------------
# launch 级调参（直接改这个 dict，或走环境变量）
#   FA_PRELOAD=1 -> enable_preload=True            即 --enable-preload（skew/CV-pipelining）
#   FA_DCP=1     -> enable_dynamic_cv_pipeline     Triton 侧 dynamic CV pipeline pass
#   FA_IBA=1     -> inject_barrier_all             （同步问题诊断用）
# 注意：enable_dynamic_cv_pipeline 内部会设 set_workspace_multibuffer=0，
#       与显式 num_stages / multibuffer_mode 可能互斥。
# ---------------------------------------------------------------------------
_LAUNCH: dict = {}


def _refresh_launch_from_env() -> dict:
    """按环境变量填充 _LAUNCH（扫描脚本可直接改 dict，不走 env）。"""
    _LAUNCH.clear()
    env_map = {
        "enable_preload": ("FA_PRELOAD", lambda v: v != "0"),
        "enable_dynamic_cv_pipeline": ("FA_DCP", lambda v: v != "0"),
        "inject_barrier_all": ("FA_IBA", lambda v: v != "0"),
    }
    for key, (env, conv) in env_map.items():
        raw = os.environ.get(env)
        if raw is not None and raw != "":
            _LAUNCH[key] = conv(raw)
    return _LAUNCH


_refresh_launch_from_env()


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
    return 32, 128


@triton.jit
def fla_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    s_ptr,
    p_ptr,
    otmp_ptr,
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
    stride_otmp_c,
    stride_otmp_q,
    stride_otmp_d,
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
    NVEC: tl.constexpr,
):
    """手写 FA 前向：显式 GM workspace（S/P/OTmp）+ 可选的 AIV 子核切分。

    数据流与官方 triton-ascend-kernels 的 `dsa_prefill.py` 同构：
      cube:   s = q@k^T          -> 写 GM 的 S（全量）
      vector: S -> mask/softmax  -> 写 GM 的 P（本子核负责的行）
      cube:   otmp = P@v         -> 写 GM 的 OTmp（全量）
      vector: OTmp -> rescale 累加 -> 写 O（本子核负责的行）

    ── 三条硬性写法约束（都是被实测/官方代码验证过的）────────────────────
    1. GM workspace 的访问必须用【普通指针 + 在调用点内联的二维广播】：
           tl.load(p_base + rows_sub[:, None] * stride_pq
                          + cols_k[None, :] * stride_pk)
       原因：别名分析在 MLIR 层比较【仿射表达式】。一旦把二维偏移预计算成
       张量变量，它就变成"未知张量值" => 无法判定同址 => 不插跨核同步 =>
       读到旧值（实测 o_err ~ 0.9）；而且 [BLOCK_Q, BLOCK_K] 的 int 张量在
       UB 里就是 64 KB 量级，6 个足以吃爆 192 KB 并触发大规模 spill。
       一维索引张量（rows_sub / cols_k）可以预计算 —— 官方 arange_sub_G 也是。

    2. 不能用 tl.make_block_ptr 访问 S/P/OTmp：block_ptr 是不透明描述符，
       别名分析看不到内部地址 => 同样丢同步（实测 o_err ~ 0.9）。

    3. 不用 al.scope / al.sync_block_*：那套在 Triton-Ascend 3.6.0 上手写 flag
       会插在 bind-sub-block pass 之前，两段仍按程序顺序执行，双向等待必然
       死锁（已用 ttadapter IR 的 `func.func x 1` 证实）。同步交给 injectBlockSync。
    """
    EVEN_Q = N_q % BLOCK_Q == 0
    EVEN_K = N_k % BLOCK_K == 0
    blocks_q = tl.cdiv(N_q, BLOCK_Q)
    taskNum = batch * head * blocks_q
    taskIdx = tl.program_id(0)
    coreNum = tl.num_programs(0)

    # AIV 子核切分（910B3 是 1 AIC : 2 AIV）。BLOCK_SUB_Q 必须是 constexpr：
    # al.sub_vec_num() 是运行时值，不能做 block_shape / tl.arange 的参数。
    BLOCK_SUB_Q: tl.constexpr = BLOCK_Q // NVEC
    if NVEC == 1:
        row_lo = 0
    else:
        # sub_vec_id() 返回 int64，转 int32 让地址算术保持 32 位
        row_lo = (al.sub_vec_id() * BLOCK_SUB_Q).to(tl.int32)

    # ── 一维索引张量：可以预计算（官方 arange_sub_G 同样预计算）──
    rows_q = tl.arange(0, BLOCK_Q)
    rows_sub = row_lo + tl.arange(0, BLOCK_SUB_Q)
    cols_k = tl.arange(0, BLOCK_K)
    cols_d = tl.arange(0, HEAD_DIM)

    for task in range(taskIdx, taskNum, coreNum):
        b = task // (head * blocks_q)
        # 倒序分配 q 块：causal 下先算可见列多的块，压缩尾部离群
        i = blocks_q - 1 - ((task % (head * blocks_q)) // head)
        h = (task % (head * blocks_q)) % head

        q_base = q_ptr + b * stride_qb + h * stride_qh
        k_base = k_ptr + b * stride_kb + h * stride_kh
        v_base = v_ptr + b * stride_vb + h * stride_vh
        o_base = o_ptr + b * stride_ob + h * stride_oh

        # ── GM workspace 基址（标量，按核分区；标量预计算不影响别名分析）──
        s_base = s_ptr + taskIdx * stride_sc
        p_base = p_ptr + taskIdx * stride_pc
        otmp_base = otmp_ptr + taskIdx * stride_otmp_c

        q_start_r = i * BLOCK_Q

        # q / k / v：只读输入，用 block_ptr 拿 boundary_check（不涉及别名问题）
        q_block_ptr = tl.make_block_ptr(
            q_base, (N_q, HEAD_DIM), (stride_qq, stride_qd), (q_start_r, 0),
            (BLOCK_Q, HEAD_DIM), (1, 0))
        k_block_ptr = tl.make_block_ptr(
            k_base, (N_k, HEAD_DIM), (stride_kk, stride_kd), (0, 0),
            (BLOCK_K, HEAD_DIM), (1, 0))
        v_block_ptr = tl.make_block_ptr(
            v_base, (N_k, HEAD_DIM), (stride_vk, stride_vd), (0, 0),
            (BLOCK_K, HEAD_DIM), (1, 0))

        # 累积量按本子核负责的行数分配
        mi = tl.full([BLOCK_SUB_Q], -1e38, dtype=tl.float32)
        li = tl.zeros([BLOCK_SUB_Q], dtype=tl.float32)
        acc = tl.zeros([BLOCK_SUB_Q, HEAD_DIM], dtype=tl.float32)

        if EVEN_Q:
            q = tl.load(q_block_ptr)
        else:
            q = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        q = q * scale.to(tl.float16)

        end = N_k
        if IS_CAUSAL:
            end = tl.minimum(q_start_r + BLOCK_Q, N_k) + delta_qk

        # 本子核负责的绝对行号（causal mask / 输出写回用）
        o_idx = q_start_r + rows_sub

        for j in range(0, end, BLOCK_K):
            # ── cube: s = q @ k^T（全量）→ 写 GM 的 S ──
            if EVEN_K:
                k = tl.load(k_block_ptr)
            else:
                k = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
            tt = tl.dot(q, tl.trans(k)).to(tl.float16)
            tl.store(s_base
                     + rows_q[:, None] * stride_sq
                     + cols_k[None, :] * stride_sk,
                     tt)

            # ── vector: 读本子核的行 → mask / softmax → 写回 P ──
            s = tl.load(s_base
                        + rows_sub[:, None] * stride_sq
                        + cols_k[None, :] * stride_sk)
            k_idx = j + cols_k
            if IS_CAUSAL:
                s = tl.where(k_idx[None, :] <= o_idx[:, None] + delta_qk, s, -6.0e4)
            if not EVEN_K:
                s = tl.where(k_idx[None, :] < N_k, s, -6.0e4)

            m_prev = mi
            mi = tl.maximum(m_prev, tl.max(s, axis=-1))
            p = tl.exp((s - mi[:, None].to(tl.float16)).to(tl.float16))
            tl.store(p_base
                     + rows_sub[:, None] * stride_pq
                     + cols_k[None, :] * stride_pk,
                     p)

            factor = tl.exp(m_prev - mi)
            acc = factor[:, None] * acc
            li = factor * li + tl.sum(p.to(tl.float32), axis=-1)

            # ── cube: otmp = P @ v（P 读全量）→ 写 GM 的 OTmp ──
            pp = tl.load(p_base
                         + rows_q[:, None] * stride_pq
                         + cols_k[None, :] * stride_pk)
            if EVEN_K:
                v = tl.load(v_block_ptr)
            else:
                v = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")
            tl.store(otmp_base
                     + rows_q[:, None] * stride_otmp_q
                     + cols_d[None, :] * stride_otmp_d,
                     tl.dot(pp, v))

            # ── vector: 读本子核的 OTmp → 累加 ──
            acc += tl.load(otmp_base
                           + rows_sub[:, None] * stride_otmp_q
                           + cols_d[None, :] * stride_otmp_d)

            # 推进 KV 方向！漏了这两行会让每一轮都读同一个 j=0 的 k/v tile，
            # 结果表现为 o_err ~ 0.9（数值合理但数据全错），极难从现象反推。
            k_block_ptr = tl.advance(k_block_ptr, (BLOCK_K, 0))
            v_block_ptr = tl.advance(v_block_ptr, (BLOCK_K, 0))

        out = (acc / li[:, None]).to(o_ptr.dtype.element_ty)
        if EVEN_Q:
            tl.store(o_base + o_idx[:, None] * stride_oq + cols_d[None, :] * stride_od, out)
        else:
            tl.store(o_base + o_idx[:, None] * stride_oq + cols_d[None, :] * stride_od, out,
                     mask=o_idx[:, None] < N_q)


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
    used_core_num = int(grid[0])

    # GM Buffers
    S = torch.empty(
        [used_core_num, BLOCK_Q, BLOCK_K], device=Q.device, dtype=torch.float16
    )
    P = torch.empty(
        [used_core_num, BLOCK_Q, BLOCK_K], device=Q.device, dtype=torch.float16
    )
    OTmp = torch.empty(
        [used_core_num, BLOCK_Q, head_dim], device=Q.device, dtype=torch.float32
    )

    fla_fwd_kernel[grid](
        Q,
        K,
        V,
        O,
        S,
        P,
        OTmp,
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
        OTmp.stride(0),
        OTmp.stride(1),
        OTmp.stride(2),
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
        # 910B3 是 1 AIC : 2 AIV => vector 侧的块按行切 2 份。
        # 必须用编译期常量（BLOCK_SUB_Q 要当 block_shape / tl.arange 参数）。
        # FA_NVEC=1 可关掉切分（用于隔离"切分"和"显式 GM"两个变量）。
        NVEC=int(os.environ.get("FA_NVEC", "2")),
        **_LAUNCH,
    )
    return O
