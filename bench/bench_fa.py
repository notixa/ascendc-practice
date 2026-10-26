"""
bench_fa.py —— CATLASS FA (catlass_op) vs Triton-Ascend FA (triton_lab) 的性能对比

设计原则（perf-loop）：
  1. 先过正确性闸门，再谈性能 —— 算错的 kernel 跑得快没有意义
  2. 同一份输入、同一 dtype、同一 layout 归一化方式
  3. warmup + 多次重复取中位数，并打印波动范围（波动 > 10% 说明测不准）
  4. 只改一个变量：backend 是唯一被比较的变量，其余全部相同

两种语义都测，分开报告：
  - full   : CATLASS 传【全 0 mask】（等价于不屏蔽）、Triton causal=False
  - causal : 两边都 causal（Triton 需先实现 causal，见 README）

用法（在 910B 机器上）：
    python3 bench/bench_fa.py
    python3 bench/bench_fa.py --only catlass
    python3 bench/bench_fa.py --skip-correctness
    python3 bench/bench_fa.py --peak-tflops 320
"""
import argparse
import math
import os
import statistics
import sys

import torch
import torch_npu

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

#   (batch, heads, N, head_dim)   —— CATLASS 要求 head_dim=128 且 N % 128 == 0
CASES = [
    (1, 8, 512, 128),
    (1, 8, 1024, 128),
    (1, 8, 2048, 128),
    (1, 8, 4096, 128),
    (4, 8, 2048, 128),
]

WARMUP = 10
ITERS = 50
REPEATS = 5

_CATLASS_MASK = {}


def catlass_mask(device, causal):
    """causal=True → 严格上三角（1=屏蔽）；False → 全 0（不屏蔽 = full attention）"""
    key = (str(device), causal)
    if key not in _CATLASS_MASK:
        base = torch.ones(128, 128, dtype=torch.float16)
        m = torch.triu(base, diagonal=1) if causal else torch.zeros_like(base)
        _CATLASS_MASK[key] = m.to(device)
    return _CATLASS_MASK[key]


# ---------------------------------------------------------------------------
# 后端封装：统一接收 [B, N, H, D]（fp16, contiguous），返回同样 shape
# ---------------------------------------------------------------------------
def make_catlass(so_path):
    torch.ops.load_library(so_path)

    def run(q_bnhd, k_bnhd, v_bnhd, causal, block_dim=0):
        return torch.ops.catlass.fat(q_bnhd, k_bnhd, v_bnhd,
                                     catlass_mask(q_bnhd.device, causal),
                                     1.0 / math.sqrt(q_bnhd.size(-1)),
                                     0, block_dim)
    return run


def make_triton():
    sys.path.insert(0, os.path.join(ROOT, "triton_lab"))
    from kernels.flash_attention.fa_fwd import fla_fwd   # noqa: E402

    def run(q_bnhd, k_bnhd, v_bnhd, causal, block_dim=0):
        # Triton 需要 [B, H, N, D]
        q = q_bnhd.permute(0, 2, 1, 3).contiguous()
        k = k_bnhd.permute(0, 2, 1, 3).contiguous()
        v = v_bnhd.permute(0, 2, 1, 3).contiguous()
        try:
            o_bhnd, _lse = fla_fwd(q, k, v, causal=causal)
        except TypeError:
            if causal:
                raise RuntimeError(
                    "fla_fwd 还不支持 causal 参数 —— 先在 triton_lab 里实现（见 bench/README.md）")
            o_bhnd, _lse = fla_fwd(q, k, v)
        return o_bhnd.permute(0, 2, 1, 3).contiguous()
    return run


# ---------------------------------------------------------------------------
# golden / 计时 / roofline
# ---------------------------------------------------------------------------
def golden(q, k, v, causal):
    qf, kf, vf = q.float(), k.float(), v.float()
    s = torch.einsum("bqhd,bkhd->bhqk", qf, kf) / math.sqrt(q.size(-1))
    if causal:
        n = q.size(1)
        m = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        s = s.masked_fill(m, float("-inf"))
    p = torch.softmax(s, dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", p, vf)


def timeit(fn, warmup=WARMUP, iters=ITERS, repeats=REPEATS):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()

    samples = []
    for _ in range(repeats):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.npu.synchronize()
        samples.append(start.elapsed_time(end) / iters)   # ms / iter

    med = statistics.median(samples)
    lo, hi = min(samples), max(samples)
    spread = (hi - lo) / med if med > 0 else float("inf")
    return med, lo, hi, spread


def flops_of(batch, heads, n, d, causal):
    # S = Q K^T : 2*N*N*D ;  O = P V : 2*N*N*D  （每个 head）
    f = 4.0 * batch * heads * n * n * d
    return f / 2.0 if causal else f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["catlass", "triton"], default=None)
    ap.add_argument("--mode", choices=["full", "causal", "both"], default="both")
    ap.add_argument("--skip-correctness", action="store_true")
    ap.add_argument("--peak-tflops", type=float, default=None,
                    help="datasheet 峰值 fp16 TFLOPS，填了才输出 %%peak")
    ap.add_argument("--so", default=os.path.join(ROOT, "catlass_lab/catlass_op/build/libcatlass_fat.so"))
    ap.add_argument("--atol", type=float, default=2e-2)
    args = ap.parse_args()

    backends = {}
    if args.only in (None, "catlass"):
        backends["catlass"] = make_catlass(args.so)
    if args.only in (None, "triton"):
        backends["triton"] = make_triton()

    modes = [False, True] if args.mode == "both" else [args.mode == "causal"]

    print(f"device: {torch.npu.get_device_name(0)}")
    print(f"warmup={WARMUP} iters={ITERS} repeats={REPEATS}（每轮取中位数）\n")

    rows = []
    for (batch, heads, n, d) in CASES:
        if d != 128:
            print(f"skip B={batch} H={heads} N={n} D={d}（CATLASS 只支持 head_dim=128）")
            continue

        torch.manual_seed(0)
        q = torch.randn(batch, n, heads, d, dtype=torch.float16).npu()
        k = torch.randn(batch, n, heads, d, dtype=torch.float16).npu()
        v = torch.randn(batch, n, heads, d, dtype=torch.float16).npu()

        print(f"── B={batch} H={heads} N={n} D={d} ──")

        for causal in modes:
            tag = "causal" if causal else "full  "
            ref = None if args.skip_correctness else golden(q.cpu(), k.cpu(), v.cpu(), causal)

            for name, fn in backends.items():
                # ---- 闸门 ----
                if ref is not None:
                    try:
                        o = fn(q, k, v, causal)
                        torch.npu.synchronize()
                    except Exception as e:                  # noqa: BLE001
                        print(f"  [{tag}] {name:<8} 跑不通: {e}")
                        continue
                    err = (o.cpu().float() - ref).abs().max().item()
                    ok = err < args.atol
                    print(f"  [{tag}] {'OK ' if ok else 'FAIL'} {name:<8} max_err={err:.5f}")
                    if not ok:
                        print(f"           → 跳过计时（算错的数字没有意义）")
                        continue

                # ---- 计时 ----
                try:
                    med, lo, hi, spread = timeit(lambda: fn(q, k, v, causal))
                except Exception as e:                      # noqa: BLE001
                    print(f"  [{tag}] {name:<8} 计时失败: {e}")
                    continue
                f = flops_of(batch, heads, n, d, causal)
                tflops = f / (med * 1e-3) / 1e12
                peak = f"  {100.0 * tflops / args.peak_tflops:5.1f}% peak" if args.peak_tflops else ""
                warn = "  ⚠ 波动大" if spread > 0.10 else ""
                print(f"  [{tag}]     {name:<8} {med:8.3f} ms  [{lo:.3f}, {hi:.3f}]  "
                      f"{tflops:7.1f} TFLOPS{peak}{warn}")
                rows.append((name, causal, batch, heads, n, med, tflops))
        print()

    # ---- 汇总 ----
    if rows:
        by_case = {}
        for name, causal, batch, heads, n, med, tflops in rows:
            by_case.setdefault((causal, batch, heads, n), {})[name] = med
        print("=" * 74)
        print(f"{'mode':<8}{'case':<20}{'catlass(ms)':>15}{'triton(ms)':>13}{'catlass/triton':>16}")
        for (causal, batch, heads, n), d_ in sorted(by_case.items(), key=lambda x: (x[0][0], x[0][3])):
            c = d_.get("catlass", float("nan"))
            t = d_.get("triton", float("nan"))
            ratio = c / t if t == t and t > 0 else float("nan")
            mode = "causal" if causal else "full"
            print(f"{mode:<8}{f'B{batch} H{heads} N{n}':<20}{c:>15.3f}{t:>13.3f}{ratio:>16.2f}x")
        print("=" * 74)
        print("ratio > 1 表示 CATLASS 更慢，< 1 表示更快")
        print("注意：causal 行两边 FLOPs 相同，但绝对 FLOPs 只有 full 的一半，不要跨行比较 TFLOPS")


if __name__ == "__main__":
    main()
