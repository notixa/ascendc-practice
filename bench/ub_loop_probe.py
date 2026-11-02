#!/usr/bin/env python3
"""测量「两段循环 vs 单循环」对 UB 峰值的影响（纯 A/B，其余代码完全一致）。

做法：用同一个 @triton.jit 的 _body，分别在 1 个和 2 个 for 循环里调用，
在固定 tile 下编译（默认 128x128，已知会 ub overflow），从 bishengir 的
"requires N bits while M bits available" 里读出 UB 需求并对比。

⇒ 两者的差值 = 「多一个循环」带来的 UB 增量，也就是：
   段1 的 k/v buffer 到底有没有被段2 复用。

用法（在 NPU 机器上跑）:
    python3 bench/ub_loop_probe.py                    # tile = 128x128
    python3 bench/ub_loop_probe.py --bq 64 --bk 256   # 换 tile
    python3 bench/ub_loop_probe.py --keep             # 顺便跑一个能编过的小 tile 做 sanity

注意：这是合成 kernel（只有 dot / exp2 / dot），UB 绝对值与真实 FA 不同，
      但【1 循环 vs 2 循环的差值】正是我们要看的量。
"""

import argparse
import re

import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def _body(q, kp, vp, o, BLOCK_K: tl.constexpr):
    """一次 KV 迭代。指针通过返回值穿线，和真实 kernel 的写法一致。"""
    k = tl.load(kp)
    v = tl.load(vp)
    s = tl.dot(q, tl.trans(k)).to(tl.float16)
    mi = tl.max(s, axis=-1).to(tl.float32)
    s = tl.exp2((s - mi[:, None].to(tl.float16)).to(tl.float16))
    o = tl.dot(s, v, o)
    kp = tl.advance(kp, (BLOCK_K, 0))
    vp = tl.advance(vp, (BLOCK_K, 0))
    return o, kp, vp


@triton.jit
def _fa_one_loop(q_ptr, k_ptr, v_ptr, o_ptr, N_k, SPLIT,
                 BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, HEAD_DIM: tl.constexpr):
    q = tl.load(q_ptr + tl.arange(0, BLOCK_Q)[:, None] * HEAD_DIM
                + tl.arange(0, HEAD_DIM)[None, :])
    kp = tl.make_block_ptr(k_ptr, (N_k, HEAD_DIM), (HEAD_DIM, 1), (0, 0),
                           (BLOCK_K, HEAD_DIM), (1, 0))
    vp = tl.make_block_ptr(v_ptr, (N_k, HEAD_DIM), (HEAD_DIM, 1), (0, 0),
                           (BLOCK_K, HEAD_DIM), (1, 0))
    o = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)
    nblk = tl.cdiv(N_k, BLOCK_K)
    for _ in range(0, nblk):
        o, kp, vp = _body(q, kp, vp, o, BLOCK_K=BLOCK_K)
    tl.store(o_ptr + tl.arange(0, BLOCK_Q)[:, None] * HEAD_DIM
             + tl.arange(0, HEAD_DIM)[None, :], o.to(tl.float16))


@triton.jit
def _fa_two_loop(q_ptr, k_ptr, v_ptr, o_ptr, N_k, SPLIT,
                 BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, HEAD_DIM: tl.constexpr):
    q = tl.load(q_ptr + tl.arange(0, BLOCK_Q)[:, None] * HEAD_DIM
                + tl.arange(0, HEAD_DIM)[None, :])
    kp = tl.make_block_ptr(k_ptr, (N_k, HEAD_DIM), (HEAD_DIM, 1), (0, 0),
                           (BLOCK_K, HEAD_DIM), (1, 0))
    vp = tl.make_block_ptr(v_ptr, (N_k, HEAD_DIM), (HEAD_DIM, 1), (0, 0),
                           (BLOCK_K, HEAD_DIM), (1, 0))
    o = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)
    nblk = tl.cdiv(N_k, BLOCK_K)
    # 段1：[0, SPLIT)
    for _ in range(0, SPLIT):
        o, kp, vp = _body(q, kp, vp, o, BLOCK_K=BLOCK_K)
    # 段2：[SPLIT, nblk)  —— 循环体与段1 完全相同
    for _ in range(SPLIT, nblk):
        o, kp, vp = _body(q, kp, vp, o, BLOCK_K=BLOCK_K)
    tl.store(o_ptr + tl.arange(0, BLOCK_Q)[:, None] * HEAD_DIM
             + tl.arange(0, HEAD_DIM)[None, :], o.to(tl.float16))


def probe(kern, q, k, v, o, n_k, split, bq, bk, d):
    try:
        kern[(1,)](q, k, v, o, n_k, split, BLOCK_Q=bq, BLOCK_K=bk, HEAD_DIM=d)
        torch.npu.synchronize()
        return None, "OK"
    except Exception as exc:                                   # noqa: BLE001
        msg = str(exc)
        m = re.search(r"requires (\d+) bits while (\d+) bits", msg)
        if m:
            return int(m.group(1)), msg
        return None, f"{type(exc).__name__}: {msg.splitlines()[0][:110]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bq", type=int, default=128)
    ap.add_argument("--bk", type=int, default=128)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--split", type=int, default=8, help="段1/段2 分界的 tile 数")
    args = ap.parse_args()

    d = 128
    n = args.n
    q = torch.randn(args.bq * d, dtype=torch.float16, device="npu")
    k = torch.randn(n * d, dtype=torch.float16, device="npu")
    v = torch.randn(n * d, dtype=torch.float16, device="npu")
    o = torch.zeros(args.bq * d, dtype=torch.float16, device="npu")

    print(f"\ntile = ({args.bq}, {args.bk})   N_k = {n}   split = {args.split} 个 tile")
    res = {}
    for tag, kern in (("1-loop", _fa_one_loop), ("2-loop", _fa_two_loop)):
        bits, info = probe(kern, q, k, v, o, n, args.split, args.bq, args.bk, d)
        res[tag] = bits
        if bits is None:
            print(f"  {tag:<8} {info}")
        else:
            print(f"  {tag:<8} requires {bits:>9,} bits = {bits / 8 / 1024:>8.1f} KB")

    a, b = res["1-loop"], res["2-loop"]
    print()
    if isinstance(a, int) and isinstance(b, int):
        d_bits = b - a
        print(f"  Δ(2-loop − 1-loop) = {d_bits:+,} bits = {d_bits / 8 / 1024:+.1f} KB")
        one_kv = args.bk * d * 2 * 2          # k + v 各一份 fp16 tile
        print(f"  一个 k/v 副本 = {one_kv / 1024:.1f} KB"
              f" ⇒ Δ 相当于 {d_bits / 8 / one_kv:+.2f} 份")
        print()
        if abs(d_bits) < one_kv * 8 * 0.1:
            print("  ⇒ 结论：两段循环几乎不额外占 UB —— 段1 的 k/v 被段2 复用了，")
            print("           「拆段吃掉 UB」这个假设不成立。")
        elif d_bits > 0:
            print("  ⇒ 结论：两段循环确实多吃了约一份 k/v —— 段1 的 buffer 没被复用。")
        else:
            print("  ⇒ 反常：两段反而更省，需要看 IR 才能解释。")
    else:
        print("  （有一个变体没拿到 UB 数字，无法比较 —— 可能是它编过了）")


if __name__ == "__main__":
    main()
