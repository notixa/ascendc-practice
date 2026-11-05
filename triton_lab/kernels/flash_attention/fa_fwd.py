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
    # (64, 512)：实测 2.298 ms / 59.8 TFLOPS，比 (64,256) 的 3.257 ms 快 1.42x。
    #   KV 重复读随 BLOCK_K 翻倍而减半；这是 UB 允许的最大 BLOCK_K。
    #   BLOCK_Q=64 是下限（32x512 = 3.747 ms，慢 60%）。
    #   更大 tile 的 UB 需求（bishengir 实测）：
    #     96x512  -> 1572864 bits（正好等于 192KB 上限，仍失败）
    #     128x512 -> 2097152 bits（256KB，超 1.33x）
    #     64x1024 -> 4325376 bits（4.1MB）
    (64, 512),  # ★ N_k 能被 512 整除时启用
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
# 手动 CV 流水：把 cube 的 s_{j+1} = q·k_{j+1}^T 提到 vector 处理 s_j 之前。
# 这两件事在数据上互不依赖，摆成相邻语句后编译器/硬件才有可能重叠执行。
# 910B 无 L0C→UB，cube 的 s 本来就走 GM workspace（默认 6 份缓冲），
# 所以 s 双缓冲不额外吃 UB。
_ROTATE = tl.constexpr(int(os.environ.get("FA_ROTATE", "0")))

#   FA_UNROLL=1（默认，不展开） / 2 / 4
# 注意：@triton.jit 函数只能读取【标注为 tl.constexpr】的模块全局变量，
#       否则报 "Cannot access global variable ... from within @jit'ed function"。
_UNROLL: tl.constexpr = int(os.environ.get("FA_UNROLL", "1"))


# ---------------------------------------------------------------------------
# launch 级调参（官方 triton-ascend-kernels 仓库里【最高频】的调优手段）
#
#   官方使用频次：num_warps 25 次 / num_stages 18 次 / multibuffer 5 次
#   而官方对 al.scope / al.sync_block_* / bl.alloc 的使用次数是【0】。
#
# 关键：这里的 num_stages 是【launch 级编译选项】，不是 tl.range(num_stages=)
#       那个循环级参数 —— 后者实测无效（1/2/3/4 全在 3.87±0.01）。
#       官方文档（docs/sources/best-practices/chunk_scaled_dot_kkt.md）明说：
#         "multibuffer=True 是 NPU 侧的软件流水开关（类比 GPU 侧 num_stages
#          的作用），把下一个 K 块的 GM→UB load 与当前块的 tl.dot 重叠，
#          掩盖 load 时延。"
#
# _LAUNCH 是【可变 dict】：bench/tune_launch.py 直接改它做扫描，不用起子进程。
# 也支持环境变量，方便手跑单个配置：
#   FA_NW=8 FA_NS=3 FA_MB=1 python3 bench/triton_check.py --perf
# ---------------------------------------------------------------------------
_LAUNCH: dict = {}


def _refresh_launch_from_env() -> dict:
    """按环境变量填充 _LAUNCH（扫描脚本可直接改 dict，不走 env）。"""
    _LAUNCH.clear()
    env_map = {
        "num_warps": ("FA_NW", int),
        "num_stages": ("FA_NS", int),
        "multibuffer": ("FA_MB", lambda v: v != "0"),
        "set_workspace_multibuffer": ("FA_WSMB", int),
        "limit_auto_multi_buffer_only_for_local_buffer": (
            "FA_LAMB",
            lambda v: v != "0",
        ),
        "inject_barrier_all": ("FA_IBA", lambda v: v != "0"),
        # CV-pipelining 的两个官方总开关（都是正规 NPUOptions 字段）
        #   FA_PRELOAD=1 -> --enable-preload=true  即 skew/preload 模式（PreLaunch）
        #   FA_DCP=1     -> enable_dynamic_cv_pipeline（Triton 侧 pass，
        #                   内部自动配好 set_workspace_multibuffer=0 /
        #                   enable_mixed_cv=True / disable_auto_inject_block_sync=True）
        "enable_preload": ("FA_PRELOAD", lambda v: v != "0"),
        "enable_dynamic_cv_pipeline": ("FA_DCP", lambda v: v != "0"),
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
    # (64,512)：BKV=512 是 UB 允许的最大值，实测比 (64,256) 快 1.42x。
    # 守卫 n_k > 512：实测 (64,512) 在 N_k == BLOCK_K == 512 且 non-causal 时
    #   触发 aicore 507015（单次迭代 + 末尾 advance 越过 block_ptr 边界）。
    #   N_k >= 1024 的多迭代情形正常（1024/2048/4096 non-causal 均通过）。
    # BKV=512 同样有尾块 bug（N 非整除时 aicore 507015），只在整除时使用。
    if n_k % 512 == 0 and n_k > 512:
        return 64, 512
    if n_k % 256 == 0:
        return 64, 256
    return 32, 128


@triton.jit
def fla_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
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
        k_block_ptr = tl.make_block_ptr(
            k_base,
            (N_k, HEAD_DIM),
            strides=(stride_kk, stride_kd),
            offsets=(0, 0),
            block_shape=(BLOCK_K, HEAD_DIM),
            order=(1, 0),
        )
        v_block_ptr = tl.make_block_ptr(
            v_base,
            (N_k, HEAD_DIM),
            strides=(stride_vk, stride_vd),
            offsets=(0, 0),
            block_shape=(BLOCK_K, HEAD_DIM),
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

        end = N_k
        if IS_CAUSAL:
            end = tl.minimum(q_start_r + BLOCK_Q, N_k) + delta_qk
        # q_idx 只依赖 q_start_r / BLOCK_Q，是循环不变量，提到循环外只算一次。
        # 循环内原本每轮重算 1 条 [BLOCK_Q] 向量加法（N=4096 causal 时 17408 次）。
        q_idx = q_start_r + tl.arange(0, BLOCK_Q)
        if _ROTATE:
            # ═══ 旋转版：cube 抢跑一块 ═══════════════════════════════════════
            k0_bp = tl.make_block_ptr(
                k_base, (N_k, HEAD_DIM), (stride_kk, stride_kd),
                (0, 0), (BLOCK_K, HEAD_DIM), (1, 0))
            if EVEN_K:
                s_prev = tl.dot(q, tl.trans(tl.load(k0_bp))).to(tl.float16)
            else:
                k0 = tl.load(k0_bp, boundary_check=(0, 1), padding_option="zero")
                k0_off = tl.arange(0, BLOCK_K)
                s_prev = tl.where(
                    k0_off[None, :] < N_k,
                    tl.dot(q, tl.trans(k0)).to(tl.float16), -6.0e4)
            if IS_CAUSAL:
                k0_idx = tl.arange(0, BLOCK_K)
                s_prev = tl.where(k0_idx[None, :] <= q_idx[:, None] + delta_qk, s_prev, -6.0e4)

            for j in range(0, end, BLOCK_K):
                # ── cube：预取下一块的 s（与下面整段 vector 工作无依赖）──
                kn_bp = tl.make_block_ptr(
                    k_base, (N_k, HEAD_DIM), (stride_kk, stride_kd),
                    (j + BLOCK_K, 0), (BLOCK_K, HEAD_DIM), (1, 0))
                if EVEN_K:
                    s_next = tl.dot(q, tl.trans(tl.load(kn_bp))).to(tl.float16)
                else:
                    kn = tl.load(kn_bp, boundary_check=(0, 1), padding_option="zero")
                    kn_off = j + BLOCK_K + tl.arange(0, BLOCK_K)
                    s_next = tl.where(
                        kn_off[None, :] < N_k,
                        tl.dot(q, tl.trans(kn)).to(tl.float16), -6.0e4)
                if IS_CAUSAL:
                    kn_idx = j + BLOCK_K + tl.arange(0, BLOCK_K)
                    s_next = tl.where(kn_idx[None, :] <= q_idx[:, None] + delta_qk, s_next, -6.0e4)

                # ── 当前块的 v ──
                v_bp = tl.make_block_ptr(
                    v_base, (N_k, HEAD_DIM), (stride_vk, stride_vd),
                    (j, 0), (BLOCK_K, HEAD_DIM), (1, 0))
                if EVEN_K:
                    v = tl.load(v_bp)
                else:
                    v = tl.load(v_bp, boundary_check=(0, 1), padding_option="zero")

                # ── vector：处理上一块的 s ──
                m_prev = mi
                mi = tl.maximum(m_prev, tl.max(s_prev, axis=-1))
                p = tl.exp((s_prev - mi[:, None].to(tl.float16)).to(tl.float16))
                factor = tl.exp(m_prev - mi)
                acc = factor[:, None] * acc
                acc = tl.dot(p, v, acc)
                li = factor * li + tl.sum(p.to(tl.float32), axis=-1)
                s_prev = s_next
        else:
            for j in range(0, end, BLOCK_K):
                if EVEN_K:
                    k = tl.load(k_block_ptr)
                    v = tl.load(v_block_ptr)
                    s = tl.dot(q, tl.trans(k)).to(tl.float16)
                else:
                    k = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
                    v = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")
                    k_offsets = j + tl.arange(0, BLOCK_K)
                    s = tl.where(
                        k_offsets[None, :] < N_k,
                        tl.dot(q, tl.trans(k)).to(tl.float16),
                        -6.0e4,
                    )

                # 只用 constexpr 的 IS_CAUSAL 判断：没有运行时 IS_DIAG 就没有 scf.if，
                # 编译器对 causal/非 causal 各生成一份特化，UB 峰值只按一条路径算。
                # 代价是每个 tile 都做一次 mask（实测 mask 成本 ≈ 0）。
                if IS_CAUSAL:
                    k_idx = j + tl.arange(0, BLOCK_K)
                    s = tl.where(k_idx[None, :] <= q_idx[:, None] + delta_qk, s, -6.0e4)
                m_prev = mi
                mi = tl.maximum(m_prev, tl.max(s, axis=-1))
                s = tl.exp((s - mi[:, None].to(tl.float16)).to(tl.float16))
                factor = tl.exp(m_prev - mi)
                acc = factor[:, None] * acc
                acc = tl.dot(s, v, acc)
                li = factor * li + tl.sum(s.to(tl.float32), axis=-1)
                k_block_ptr = tl.advance(k_block_ptr, (BLOCK_K, 0))
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
    fla_fwd_kernel[grid](
        Q,
        K,
        V,
        O,
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
        **_LAUNCH,
    )
    return O
