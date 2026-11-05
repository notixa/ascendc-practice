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
    s_ws_ptr,
    p_ws_ptr,
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
    PIPE_STAGES: tl.constexpr,
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
        # ═══════════════════════════════════════════════════════════════════
        # CATLASS PreLaunch 范式：N 槽 ring + cube 抢跑 N 拍
        #
        # 环缓冲：s_ws[stage] 存 s_j，p_ws[stage] 存 softmax 后的 p_j
        #   stage = (j / BLOCK_K) % PIPE_STAGES
        #
        # 每迭代的三个阶段（跨迭代重叠，这是 PreLaunch 的核心）：
        #   ① cube  : s_{j+LOOKAHEAD} = q·k^T  → 写入 slot (i+LOOKAHEAD)%PIPE_STAGES（抢跑）
        #   ② vector: 从 slot stage 读 s_j → softmax → p_j 写入 slot stage
        #   ③ cube  : acc += p_j · v_j
        #
        # 同步（每 slot 一事件，event_id = stage，范围 [0,15]）：
        #   cube  set(cube→vector, stage, FIX, MTE2)   写满 slot，通知 vec
        #   vec   wait(cube→vector, stage, FIX, MTE2)  读 slot 前等
        #   vec   set(vector→cube, stage, MTE3, MTE1)  写好 p，通知 cube
        #   cube  wait(vector→cube, stage, MTE3, MTE1) PV 前等
        # ═══════════════════════════════════════════════════════════════════

        # prologue：cube 先把前 LOOKAHEAD 个 slot 填满（抢跑 = PIPE_STAGES-1 拍）
        # 槽数 = 抢跑拍数 + 1（CATLASS: preLaunch=2 配 3 槽），
        # 这样 cube 写的槽永远 != vector 正在读的槽。
        for pre in tl.static_range(0, PIPE_STAGES - 1):
            kp_bp = tl.make_block_ptr(
                k_base, (N_k, HEAD_DIM), (stride_kk, stride_kd),
                (pre * BLOCK_K, 0), (BLOCK_K, HEAD_DIM), (1, 0))
            if EVEN_K:
                s_pre = tl.dot(q, tl.trans(tl.load(kp_bp))).to(tl.float16)
            else:
                kp = tl.load(kp_bp, boundary_check=(0, 1), padding_option="zero")
                kp_off = pre * BLOCK_K + tl.arange(0, BLOCK_K)
                s_pre = tl.where(kp_off[None, :] < N_k,
                                 tl.dot(q, tl.trans(kp)).to(tl.float16), -6.0e4)
            if IS_CAUSAL:
                kp_idx = pre * BLOCK_K + tl.arange(0, BLOCK_K)
                s_pre = tl.where(kp_idx[None, :] <= q_idx[:, None] + delta_qk, s_pre, -6.0e4)
            s_slot = s_ws_ptr + pre * (BLOCK_Q * BLOCK_K)
            tl.store(s_slot + tl.arange(0, BLOCK_Q)[:, None] * BLOCK_K
                     + tl.arange(0, BLOCK_K)[None, :], s_pre)
            al.sync_block_set("cube", "vector", pre, al.PIPE.PIPE_FIX, al.PIPE.PIPE_MTE2)

        for j in range(0, end, BLOCK_K):
            stage = (j // BLOCK_K) % PIPE_STAGES
            # cube 抢跑 LOOKAHEAD 拍：写 (i+LOOKAHEAD) % PIPE_STAGES 槽
            nstage = ((j // BLOCK_K) + PIPE_STAGES - 1) % PIPE_STAGES
            jn = j + (PIPE_STAGES - 1) * BLOCK_K

            # ── ① cube：抢跑第 j+LOOKAHEAD 块，写入 slot nstage ──
            # 只在 jn < end 时才抢跑：否则该 slot 永远没人消费，
            # sync 信号量会泄漏（+1 不平），累积后让后续 task 的 wait 变成假通过。
            if jn < end:
              kn_bp = tl.make_block_ptr(
                  k_base, (N_k, HEAD_DIM), (stride_kk, stride_kd),
                  (jn, 0), (BLOCK_K, HEAD_DIM), (1, 0))
              if EVEN_K:
                  s_next = tl.dot(q, tl.trans(tl.load(kn_bp))).to(tl.float16)
              else:
                  kn = tl.load(kn_bp, boundary_check=(0, 1), padding_option="zero")
                  kn_off = jn + tl.arange(0, BLOCK_K)
                  s_next = tl.where(kn_off[None, :] < N_k,
                                    tl.dot(q, tl.trans(kn)).to(tl.float16), -6.0e4)
              if IS_CAUSAL:
                  kn_idx = jn + tl.arange(0, BLOCK_K)
                  s_next = tl.where(kn_idx[None, :] <= q_idx[:, None] + delta_qk, s_next, -6.0e4)
              s_nslot = s_ws_ptr + nstage * (BLOCK_Q * BLOCK_K)
              tl.store(s_nslot + tl.arange(0, BLOCK_Q)[:, None] * BLOCK_K
                       + tl.arange(0, BLOCK_K)[None, :], s_next)
              al.sync_block_set("cube", "vector", nstage, al.PIPE.PIPE_FIX, al.PIPE.PIPE_MTE2)

            # ── ② vector：读 slot stage 的 s_j，算 softmax ──
            al.sync_block_wait("cube", "vector", stage, al.PIPE.PIPE_FIX, al.PIPE.PIPE_MTE2)
            s_slot = s_ws_ptr + stage * (BLOCK_Q * BLOCK_K)
            s_cur = tl.load(s_slot + tl.arange(0, BLOCK_Q)[:, None] * BLOCK_K
                            + tl.arange(0, BLOCK_K)[None, :])
            m_prev = mi
            mi = tl.maximum(m_prev, tl.max(s_cur, axis=-1))
            p = tl.exp((s_cur - mi[:, None].to(tl.float16)).to(tl.float16))
            factor = tl.exp(m_prev - mi)
            acc = factor[:, None] * acc
            li = factor * li + tl.sum(p.to(tl.float32), axis=-1)
            p_slot = p_ws_ptr + stage * (BLOCK_Q * BLOCK_K)
            tl.store(p_slot + tl.arange(0, BLOCK_Q)[:, None] * BLOCK_K
                     + tl.arange(0, BLOCK_K)[None, :], p)
            al.sync_block_set("vector", "cube", stage, al.PIPE.PIPE_MTE3, al.PIPE.PIPE_MTE1)

            # ── ③ cube：acc += p_j · v_j ──
            al.sync_block_wait("vector", "cube", stage, al.PIPE.PIPE_MTE3, al.PIPE.PIPE_MTE1)
            p_read = tl.load(p_slot + tl.arange(0, BLOCK_Q)[:, None] * BLOCK_K
                             + tl.arange(0, BLOCK_K)[None, :])
            v_bp = tl.make_block_ptr(
                v_base, (N_k, HEAD_DIM), (stride_vk, stride_vd),
                (j, 0), (BLOCK_K, HEAD_DIM), (1, 0))
            if EVEN_K:
                v = tl.load(v_bp)
            else:
                v = tl.load(v_bp, boundary_check=(0, 1), padding_option="zero")
            acc = tl.dot(p_read, v, acc)

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

    # CATLASS PreLaunch：N 槽 ring workspace（s 和 p 各一条环）
    #   s_ws[stage] = s_{stage}（QK 结果，fp16）
    #   p_ws[stage] = softmax 后的 p（fp16）
    # 每核独立使用（一个核串行处理多个 task，环在 task 内复用）。
    PIPE_STAGES = int(os.environ.get("FA_STAGES", "2"))
    _BQ, _BK = _pick_tile(N_q, N_k)
    s_ws = torch.zeros((PIPE_STAGES, _BQ, _BK), dtype=torch.float16, device=Q.device)
    p_ws = torch.zeros((PIPE_STAGES, _BQ, _BK), dtype=torch.float16, device=Q.device)

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
        s_ws,
        p_ws,
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
        PIPE_STAGES=PIPE_STAGES,
        **_LAUNCH,
    )
    return O
