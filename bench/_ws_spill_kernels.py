"""ws_spill_probe 的 kernel 定义（必须是真实文件：@triton.jit 需要 inspect 能读到源码）。

两个 kernel 的 dot 数量、形状、累加方式完全一致，唯一差别是
**每个 dot 之后是否立刻 tl.store 到 workspace**：

    k_hold  : 4 个 dot 全部产出后才消费（s0..s3 同时存活）
    k_spill : 每个 dot 之后立刻 store 到 ws，再逐块 load 回来消费

判据见 ws_spill_probe.py。
"""

import triton
import triton.language as tl


@triton.jit
def k_hold(q_ptr, k_ptr, ws_ptr, o_ptr, N,
           BLOCK_Q: tl.constexpr, BK: tl.constexpr, DIM: tl.constexpr):
    q = tl.load(q_ptr + tl.arange(0, BLOCK_Q)[:, None] * DIM
                + tl.arange(0, DIM)[None, :])
    acc = tl.zeros([BLOCK_Q], tl.float32)
    for base in range(0, N, 4 * BK):
        k0 = tl.load(tl.make_block_ptr(k_ptr, (N, DIM), (DIM, 1), (base + 0 * BK, 0), (BK, DIM), (1, 0)))
        k1 = tl.load(tl.make_block_ptr(k_ptr, (N, DIM), (DIM, 1), (base + 1 * BK, 0), (BK, DIM), (1, 0)))
        k2 = tl.load(tl.make_block_ptr(k_ptr, (N, DIM), (DIM, 1), (base + 2 * BK, 0), (BK, DIM), (1, 0)))
        k3 = tl.load(tl.make_block_ptr(k_ptr, (N, DIM), (DIM, 1), (base + 3 * BK, 0), (BK, DIM), (1, 0)))
        s0 = tl.dot(q, tl.trans(k0))
        s1 = tl.dot(q, tl.trans(k1))
        s2 = tl.dot(q, tl.trans(k2))
        s3 = tl.dot(q, tl.trans(k3))
        # 四个 s 同时存活到这里
        acc += tl.max(tl.maximum(tl.maximum(s0, s1), tl.maximum(s2, s3)), -1)
    tl.store(o_ptr + tl.arange(0, BLOCK_Q), acc)


@triton.jit
def k_spill(q_ptr, k_ptr, ws_ptr, o_ptr, N,
            BLOCK_Q: tl.constexpr, BK: tl.constexpr, DIM: tl.constexpr):
    q = tl.load(q_ptr + tl.arange(0, BLOCK_Q)[:, None] * DIM
                + tl.arange(0, DIM)[None, :])
    acc = tl.zeros([BLOCK_Q], tl.float32)
    pid = tl.program_id(0)
    slot = pid * (4 * BLOCK_Q * BK)
    for base in range(0, N, 4 * BK):
        # ---- 阶段 1：每个 dot 之后立刻落盘，s_i 的生命周期在此终止 ----
        kw0 = tl.make_block_ptr(ws_ptr + slot + 0 * BLOCK_Q * BK, (BLOCK_Q, BK), (BK, 1), (0, 0), (BLOCK_Q, BK), (1, 0))
        kw1 = tl.make_block_ptr(ws_ptr + slot + 1 * BLOCK_Q * BK, (BLOCK_Q, BK), (BK, 1), (0, 0), (BLOCK_Q, BK), (1, 0))
        kw2 = tl.make_block_ptr(ws_ptr + slot + 2 * BLOCK_Q * BK, (BLOCK_Q, BK), (BK, 1), (0, 0), (BLOCK_Q, BK), (1, 0))
        kw3 = tl.make_block_ptr(ws_ptr + slot + 3 * BLOCK_Q * BK, (BLOCK_Q, BK), (BK, 1), (0, 0), (BLOCK_Q, BK), (1, 0))

        k0 = tl.load(tl.make_block_ptr(k_ptr, (N, DIM), (DIM, 1), (base + 0 * BK, 0), (BK, DIM), (1, 0)))
        tl.store(kw0, tl.dot(q, tl.trans(k0)))
        k1 = tl.load(tl.make_block_ptr(k_ptr, (N, DIM), (DIM, 1), (base + 1 * BK, 0), (BK, DIM), (1, 0)))
        tl.store(kw1, tl.dot(q, tl.trans(k1)))
        k2 = tl.load(tl.make_block_ptr(k_ptr, (N, DIM), (DIM, 1), (base + 2 * BK, 0), (BK, DIM), (1, 0)))
        tl.store(kw2, tl.dot(q, tl.trans(k2)))
        k3 = tl.load(tl.make_block_ptr(k_ptr, (N, DIM), (DIM, 1), (base + 3 * BK, 0), (BK, DIM), (1, 0)))
        tl.store(kw3, tl.dot(q, tl.trans(k3)))

        # ---- 阶段 2：逐块读回，UB 里同时只有 1 份 s ----
        t0 = tl.load(kw0)
        m0 = tl.max(t0, -1)
        t1 = tl.load(kw1)
        m1 = tl.max(t1, -1)
        t2 = tl.load(kw2)
        m2 = tl.max(t2, -1)
        t3 = tl.load(kw3)
        m3 = tl.max(t3, -1)
        acc += tl.maximum(tl.maximum(m0, m1), tl.maximum(m2, m3))
    tl.store(o_ptr + tl.arange(0, BLOCK_Q), acc)
