"""验证 fp32->fp16 的转换落在 L0C->GM(fixpipe) 还是 Vector。

两个 kernel 结构完全相同（store 到 ws -> 读回 -> 归约输出），唯一差别是
ws 的 dtype 和 store 前的显式转换：
    k_store_f32 : ws 是 fp32，直接存
    k_store_f16 : ws 是 fp16，存之前 .to(tl.float16)
"""

import triton
import triton.language as tl


@triton.jit
def k_store_f32(q_ptr, k_ptr, ws_ptr, o_ptr, N, BLOCK_Q: tl.constexpr, BK: tl.constexpr,
                DIM: tl.constexpr):
    q = tl.load(q_ptr + tl.arange(0, BLOCK_Q)[:, None] * DIM +
                tl.arange(0, DIM)[None, :])
    pid = tl.program_id(0)
    slot = pid * (4 * BLOCK_Q * BK)
    acc = tl.zeros([BLOCK_Q], tl.float32)
    for base in range(0, N, 4 * BK):
        for i in tl.static_range(4):
            kp = tl.make_block_ptr(k_ptr, (N, DIM), (DIM, 1), (base + i * BK, 0),
                                   (BK, DIM), (1, 0))
            w = tl.make_block_ptr(ws_ptr + slot + i * BLOCK_Q * BK, (BLOCK_Q, BK),
                                  (BK, 1), (0, 0), (BLOCK_Q, BK), (1, 0))
            tl.store(w, tl.dot(q, tl.trans(tl.load(kp))))
        for i in tl.static_range(4):
            r = tl.make_block_ptr(ws_ptr + slot + i * BLOCK_Q * BK, (BLOCK_Q, BK),
                                  (BK, 1), (0, 0), (BLOCK_Q, BK), (1, 0))
            acc = tl.maximum(acc, tl.max(tl.load(r), -1))
    tl.store(o_ptr + tl.arange(0, BLOCK_Q), acc)


@triton.jit
def k_store_f16(q_ptr, k_ptr, ws_ptr, o_ptr, N, BLOCK_Q: tl.constexpr, BK: tl.constexpr,
                DIM: tl.constexpr):
    q = tl.load(q_ptr + tl.arange(0, BLOCK_Q)[:, None] * DIM +
                tl.arange(0, DIM)[None, :])
    pid = tl.program_id(0)
    slot = pid * (4 * BLOCK_Q * BK)
    acc = tl.zeros([BLOCK_Q], tl.float32)
    for base in range(0, N, 4 * BK):
        for i in tl.static_range(4):
            kp = tl.make_block_ptr(k_ptr, (N, DIM), (DIM, 1), (base + i * BK, 0),
                                   (BK, DIM), (1, 0))
            w = tl.make_block_ptr(ws_ptr + slot + i * BLOCK_Q * BK, (BLOCK_Q, BK),
                                  (BK, 1), (0, 0), (BLOCK_Q, BK), (1, 0))
            tl.store(w, tl.dot(q, tl.trans(tl.load(kp))).to(tl.float16))
        for i in tl.static_range(4):
            r = tl.make_block_ptr(ws_ptr + slot + i * BLOCK_Q * BK, (BLOCK_Q, BK),
                                  (BK, 1), (0, 0), (BLOCK_Q, BK), (1, 0))
            acc = tl.maximum(acc, tl.max(tl.load(r), -1))
    tl.store(o_ptr + tl.arange(0, BLOCK_Q), acc)
