#!/usr/bin/env python3
"""Triton FA 前向：正确性 + 性能自检。

用法:
    python3 bench/triton_check.py            # 只测正确性（快）
    python3 bench/triton_check.py --perf     # 正确性 + 性能（会触发 autotune，较慢）

判据:
    o_err < 2e-2   （fp16 量化量级）

注：本版本 kernel 不再输出 LSE，只校验 O。
"""

import argparse
import math
import os
import sys

import torch
import torch_npu

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "triton_lab"))

from kernels.flash_attention.fa_fwd import fla_fwd  # noqa: E402

O_TOL = 2e-2

# (B, N, H, causal)
# N 的选择是刻意的:
#   320 / 384  -> EVEN_K=False 分支 (320,384 都不是 256 的倍数)
#   384        -> 非 causal 时是"尾部 tile"路径
#   512 / 4096 -> 整除快路径
CASES = [
    (1, 320, 4, True),
    (1, 384, 8, True),
    (1, 384, 8, False),
    (1, 512, 8, True),
    (1, 512, 8, False),
    (1, 1024, 8, False),   # 覆盖 (64,512) 的多迭代 non-causal 路径
    (1, 4096, 32, True),
]


def golden(q, k, v, causal):
    s = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) / math.sqrt(q.size(-1))
    if causal:
        n = q.size(1)
        m = torch.triu(torch.ones(n, n, dtype=torch.bool, device=s.device), 1)
        s = s.masked_fill(m, float("-inf"))
    return torch.einsum("bhqk,bkhd->bqhd", torch.softmax(s, -1), v.float())


def check_one(b, n, h, causal):
    q = torch.randn(b, n, h, 128, dtype=torch.float16, device="npu")
    o = fla_fwd(q, q, q, causal=causal)
    torch.npu.synchronize()

    ref_o = golden(q.cpu(), q.cpu(), q.cpu(), causal)
    o_err = (o.cpu().float() - ref_o).abs().max().item()

    ok = o_err < O_TOL
    print(
        f"  B{b} N{n:>5} H{h:<2} causal={str(causal):<5}"
        f"  o_err={o_err:.5f}  {'OK' if ok else 'FAIL'}"
    )
    return ok


def bench(f, warm=10, iters=20, reps=5):
    for _ in range(warm):
        f()
    torch.npu.synchronize()
    samples = []
    for _ in range(reps):
        a = torch.npu.Event(enable_timing=True)
        z = torch.npu.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            f()
        z.record()
        torch.npu.synchronize()
        samples.append(a.elapsed_time(z) / iters)
    samples.sort()
    return samples[len(samples) // 2]


def perf():
    b, n, h, causal = 1, 4096, 32, True
    q = torch.randn(b, n, h, 128, dtype=torch.float16, device="npu")
    fla_fwd(q, q, q, causal=causal)  # 触发 autotune
    torch.npu.synchronize()

    ms = bench(lambda: fla_fwd(q, q, q, causal=causal))
    flops = 4 * b * h * (n * (n + 1) / 2) * 128
    # 注意：这里【只允许】打印 [perf] 这一行。
    # 之前这里有过两行硬编码的历史对照文本（含 "4.838 ms" / "3.658 ms"），
    # 结果 `grep -oE "[0-9]+\.[0-9]+ ms"` 会把它们抓进去，
    # 让 5 次测量显示成"全是 3.658 ms、极差 0.0%"—— 纯属假数据。
    # 要留历史数字请写进 README 或注释，不要 print 到 stdout。
    print(f"\n[perf] B{b} N{n} H{h} causal={causal}:  {ms:.3f} ms   {flops / (ms * 1e-3) / 1e12:.1f} TFLOPS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perf", action="store_true", help="额外跑 N=4096 H=32 的性能")
    args = ap.parse_args()

    # 性能必须先测：正确性 case 会编译+执行 6 种不同的 constexpr 特化，
    # 之后设备状态（频率/L2/HBM 页）已非冷态 —— 实测会虚高 16%
    # （4.177 冷进程 vs 4.862 在正确性 case 之后）。
    if args.perf:
        perf()

    print("\n[correctness]")
    results = [check_one(*c) for c in CASES]

    n_fail = results.count(False)
    print(f"\n{len(results) - n_fail}/{len(results)} passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
