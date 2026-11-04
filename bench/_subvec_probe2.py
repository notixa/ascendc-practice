#!/usr/bin/env python3
"""探针：确认 `al.sub_vec_id()` / `al.extract_slice` 在【纯 SSA】kernel 里是否真的生效。

背景
----
`fa_fwd.py` 加了 AIV 子核切分（`FA_NVEC=2`）后：
    NVEC=1  o_err=0.00040  OK
    NVEC=2  o_err=1.43513  FAIL      ← 形态像"一半行没被写"
需要判断是 (A) `sub_vec_id()` 恒返回 0，还是 (B) `extract_slice` 忽略运行时 offsets，
还是 (C) 我的形状/偏移写错了。

做法
----
3 个最小 kernel，都用 `grid=(20,)`（和真实 FA 一致）：

  k_id    只把 `row_lo` 写到 GM         ⇒ 看每个核读到的 row_lo 是 0 还是 0/32 混合
  k_vec   纯 vector + extract_slice     ⇒ 看两半是否都被写（不能说明 dot 的影响）
  k_mix   带一次 tl.dot 的 mix kernel   ⇒ 和 FA 的真实结构最接近

判读
----
  k_id  输出里同时出现 0 和 32      => sub_vec_id 生效
  k_id  输出全是 0                  => **A 成立**（sub_vec_id 在纯 SSA 上无效）
  k_vec/k_mix 两半都被写            => extract_slice 的运行时 offsets 生效
  k_vec/k_mix 只有前半被写          => **B 成立**（extract_slice 忽略运行时偏移）
  两半都被写但值不对                => **C 成立**（我的偏移算法错）
"""

from __future__ import annotations

import os
import sys

import torch
import torch_npu
import triton
import triton.language as tl
from triton.language.extra.cann import extension as al

GRID = 20
BLOCK_Q = 64
NVEC = 2
SUB = BLOCK_Q // NVEC  # 32


@triton.jit
def k_id(oid_ptr, BLOCK_Q: tl.constexpr, NVEC: tl.constexpr):
    """只记录本核看到的 row_lo。"""
    BLOCK_SUB_Q: tl.constexpr = BLOCK_Q // NVEC
    if NVEC == 1:
        row_lo = 0
    else:
        row_lo = (al.sub_vec_id() * BLOCK_SUB_Q).to(tl.int32)
    tl.store(oid_ptr + tl.program_id(0), row_lo)


@triton.jit
def k_vec(o_ptr, BLOCK_Q: tl.constexpr, NVEC: tl.constexpr):
    """纯 vector：全量张量 → extract_slice → 写回本核那半。"""
    BLOCK_SUB_Q: tl.constexpr = BLOCK_Q // NVEC
    if NVEC == 1:
        row_lo = 0
    else:
        row_lo = (al.sub_vec_id() * BLOCK_SUB_Q).to(tl.int32)
    # 全量：第 r 行的值 = r * 10，切开后能直接看出是哪一段
    full = (tl.arange(0, BLOCK_Q) * 10).to(tl.float32)
    sub = al.extract_slice(full, (row_lo,), (BLOCK_SUB_Q,), (1,))
    tl.store(o_ptr + row_lo + tl.arange(0, BLOCK_SUB_Q), sub)


@triton.jit
def k_mix(q_ptr, o_ptr, BLOCK_Q: tl.constexpr, NVEC: tl.constexpr, D: tl.constexpr):
    """mix：先做一次 tl.dot（和 FA 一样有 cube 参与），再 extract_slice。"""
    BLOCK_SUB_Q: tl.constexpr = BLOCK_Q // NVEC
    if NVEC == 1:
        row_lo = 0
    else:
        row_lo = (al.sub_vec_id() * BLOCK_SUB_Q).to(tl.int32)
    cols = tl.arange(0, D)
    q = tl.load(q_ptr + tl.arange(0, BLOCK_Q)[:, None] * D + cols[None, :])
    # 让 s_full 的第 r 行 ≈ r，便于识别切到了哪一段
    s_full = tl.dot(q, tl.trans(q)).to(tl.float32)  # [BLOCK_Q, BLOCK_Q]
    s_full = s_full * 0.0 + (tl.arange(0, BLOCK_Q) * 10).to(tl.float32)[None, :]
    sub = al.extract_slice(s_full, (row_lo, 0), (BLOCK_SUB_Q, BLOCK_Q), (1, 1))
    tl.store(o_ptr + row_lo + tl.arange(0, BLOCK_SUB_Q), tl.max(sub, axis=-1))


def run(fn, pre, post, out_shape):
    """pre: 输出指针之前的实参；post: 输出指针之后的实参（constexpr）。

    注意参数顺序：k_mix 的签名是 (q_ptr, o_ptr, ...)，输出指针在【第二个】位置，
    所以用 pre/post 分开传，避免上一次那种整体错位一位的 bug。
    """
    o = torch.full(out_shape, -1.0, dtype=torch.float32, device="npu")
    fn[(GRID,)](*pre, o, *post)
    torch.npu.synchronize()
    return o.cpu()


def main():
    print(f"\n{'='*84}")
    print(f"子核切分探针  grid={GRID}  BLOCK_Q={BLOCK_Q}  NVEC={NVEC}  BLOCK_SUB_Q={SUB}")
    print(f"{'='*84}\n", flush=True)

    # ① row_lo 到底是什么
    ids = run(k_id, [], [BLOCK_Q, NVEC], (GRID,)).flatten()
    uniq = sorted(set(int(v) for v in ids.tolist()))
    print(f"① k_id: 每个核读到的 row_lo 唯一值 = {uniq}")
    print(f"   → {'✅ sub_vec_id 生效（出现 0 和 ' + str(SUB) + '）' if len(uniq) > 1 else '❌ **A 成立**：恒为 ' + str(uniq[0]) + '，纯 SSA kernel 里 sub_vec_id 不区分两个 AIV'}", flush=True)

    # ② 纯 vector + extract_slice
    ov = run(k_vec, [], [BLOCK_Q, NVEC], (BLOCK_Q,)).flatten()
    written = [i for i in range(BLOCK_Q) if ov[i].item() >= 0]
    print(f"\n② k_vec: 被写的位置数 = {len(written)}/{BLOCK_Q}"
          f"  前 4 = {[round(ov[i].item(),1) for i in range(4)]}"
          f"  后 4 = {[round(ov[BLOCK_Q-1-i].item(),1) for i in range(4)]}")
    ok_vec = len(written) == BLOCK_Q
    print(f"   → {'✅ 两半都被写，extract_slice 的运行时 offsets 生效' if ok_vec else '❌ 只有一半被写 ⇒ **B 成立**'}", flush=True)

    # ③ mix（带 dot）
    q = torch.randn(BLOCK_Q, 128, dtype=torch.float16, device="npu")
    om = run(k_mix, [q], [BLOCK_Q, NVEC, 128], (BLOCK_Q,)).flatten()
    written_m = [i for i in range(BLOCK_Q) if om[i].item() >= 0]
    print(f"\n③ k_mix: 被写的位置数 = {len(written_m)}/{BLOCK_Q}")
    print(f"   前 4 = {[round(om[i].item(),1) for i in range(4)]}"
          f"  后 4 = {[round(om[BLOCK_Q-1-i].item(),1) for i in range(4)]}")
    ok_mix = len(written_m) == BLOCK_Q
    print(f"   → {'✅ mix kernel 里也生效' if ok_mix else '❌ mix kernel 里只有一半被写'}", flush=True)

    # 结论
    print(f"\n{'='*84}")
    if len(uniq) == 1:
        print("结论：❌ **A —— sub_vec_id 在纯 SSA kernel 里恒为 0**，这是编译器/pass 行为，不是我的逻辑 bug。")
        print("      切分只能在显式 GM workspace 路径下生效（fa_fwd_manual.py 已验证可行）。")
    elif not ok_vec:
        print("结论：❌ **B —— extract_slice 忽略运行时 offsets**（sub_vec_id 本身是好的）。")
        print("      改用普通指针 + 运行时行偏移（manual 版的做法）。")
    elif not ok_mix:
        print("结论：⚠️ 纯 vector 生效但 mix kernel 不生效 ⇒ 与 tl.dot 的交互问题。")
    else:
        print("结论：✅ 机制都生效 ⇒ **C —— 我的 FA 实现里有形状/偏移 bug**，回去逐行核对。")
    print(f"{'='*84}")


if __name__ == "__main__":
    main()
