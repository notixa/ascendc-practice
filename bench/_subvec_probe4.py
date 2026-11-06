#!/usr/bin/env python3
"""探针 4：用【constexpr 偏移 + sub_vec_id 控制流】绕开"运行时偏移被忽略"，验证能否切对

设计
----
构造 q 使 `dot(q, q^T)[r, c] = 128 * r * c`（q[r,:] = r），于是
    max over c  =>  128 * r * (BLOCK_Q-1)  ∝  r
每行存 max，就能直接看出这一行是原张量的第几行。

期望（切分正确）
    实例 row_lo=0  =>  o[0..31]  ∝ 0..31
    实例 row_lo=32 =>  o[32..63] ∝ 32..63

错误形态
    A) o[32..63] 的值仍是 ∝0..31      => extract_slice 忽略运行时 offset
    B) o 里有 -1（没被写）             => row_lo 两个实例相同
    C) 值杂乱                          => extract 从 L0C 拿到了未转换的数据
"""

from __future__ import annotations

import torch
import torch_npu
import triton
import triton.language as tl
from triton.language.extra.cann import extension as al

GRID, BLOCK_Q, NVEC, D = 20, 64, 2, 64
SUB = BLOCK_Q // NVEC


@triton.jit
def k_mix(q_ptr, o_ptr, id_ptr, BLOCK_Q: tl.constexpr, NVEC: tl.constexpr,
          D: tl.constexpr):
    BLOCK_SUB_Q: tl.constexpr = BLOCK_Q // NVEC
    if NVEC == 1:
        row_lo = 0
    else:
        row_lo = (al.sub_vec_id() * BLOCK_SUB_Q).to(tl.int32)
    cols = tl.arange(0, D)
    rows = tl.arange(0, BLOCK_Q)
    # q[r, :] = r  =>  dot(q, q^T)[r, c] = D * r * c
    q = tl.full([BLOCK_Q, D], 1.0, dtype=tl.float16) * rows[:, None].to(tl.float16)
    s_full = tl.dot(q, tl.trans(q)).to(tl.float16)  # [BLOCK_Q, BLOCK_Q]
    # 变体：constexpr 偏移 + 运行时控制流（绕开"运行时偏移被忽略"）
    if NVEC == 1:
        sub = s_full
    else:
        if al.sub_vec_id() == 0:
            sub = al.extract_slice(s_full, (0, 0), (BLOCK_SUB_Q, BLOCK_Q), (1, 1))
        else:
            sub = al.extract_slice(s_full, (BLOCK_SUB_Q, 0), (BLOCK_SUB_Q, BLOCK_Q),
                                   (1, 1))
    tl.store(o_ptr + row_lo + tl.arange(0, BLOCK_SUB_Q),
             tl.max(sub, axis=-1).to(tl.float32))
    tl.store(id_ptr + tl.program_id(0), row_lo)
    _ = cols


def main():
    print(
        f"\n{'='*80}\n探针4  grid={GRID}  BLOCK_Q={BLOCK_Q}  NVEC={NVEC}  BLOCK_SUB_Q={SUB}\n{'='*80}\n",
        flush=True)
    q = torch.zeros(BLOCK_Q, D, dtype=torch.float16, device="npu")
    o = torch.full((BLOCK_Q, ), -1.0, dtype=torch.float32, device="npu")
    ids = torch.full((GRID, ), -1.0, dtype=torch.float32, device="npu")
    k_mix[(GRID, )](q, o, ids, BLOCK_Q=BLOCK_Q, NVEC=NVEC, D=D)
    torch.npu.synchronize()
    ov, iv = o.cpu(), ids.cpu()

    print(f"  实例看到的 row_lo 唯一值: {sorted(set(int(v) for v in iv.tolist()))}")
    scale = D * (BLOCK_Q - 1)
    print(f"\n  行  值        值/scale   推定来自原张量第几行")
    for r in (0, 1, 31, 32, 33, 63):
        v = ov[r].item()
        print(
            f"  {r:>2}  {v:>10.1f}  {v/scale:>8.3f}   {v/scale:>6.1f}"
            f"   {'← 期望 ' + str(r) if abs(v/scale - r) < 0.5 else '✗ 与期望 ' + str(r) + ' 不符'}"
        )
    exp = torch.arange(BLOCK_Q, dtype=torch.float32) * scale
    bad = [(r, ov[r].item() / scale, r) for r in range(BLOCK_Q)
           if abs(ov[r].item() / scale - r) > 0.5]
    print(f"\n  不符的行数: {len(bad)}/{BLOCK_Q}")
    if len(bad) == 0:
        print("  ✅ extract_slice 从 dot 结果上切分是正确的 ⇒ 我的 FA 实现另有 bug")
    elif len(bad) == BLOCK_Q // 2 and all(b[0] >= SUB for b in bad):
        print(f"  ❌ 后一半全错 ⇒ **连 constexpr 偏移 + 控制流也失效**（sub_vec_id 分支被折叠）")
    else:
        print(f"  ⚠️ 部分错，样本: {bad[:6]}")


if __name__ == "__main__":
    main()
