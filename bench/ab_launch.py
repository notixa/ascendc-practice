#!/usr/bin/env python3
"""对比 `enable_preload` / `enable_dynamic_cv_pipeline` 两个 CV-pipelining 开关的效果。

两个开关都是正规 NPUOptions 字段，直接作为 kernel launch 的 kwarg：
  enable_preload            -> bishengir `--enable-preload=true`
                               官方说明：Enable preload (skew mode) for CV-pipelining
  enable_dynamic_cv_pipeline -> Triton 侧 pass，内部自动配好
                               set_workspace_multibuffer=0 / enable_mixed_cv=True /
                               disable_auto_inject_block_sync=True

用法
----
    python3 bench/ab_launch.py                       # 两个 kernel × 4 个配置
    python3 bench/ab_launch.py --kernels fa          # 只测 fa_fwd.py
    python3 bench/ab_launch.py --tiles 32x256        # 强制 tile
    python3 bench/ab_launch.py --iters 10 --reps 3   # 快速模式

输出
----
    - 终端表格：每个 kernel 的配置 / o_err / median ms / TFLOPS / 相对自身基线的倍数
    - JSON：bench/ab_launch_result.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# 必须在 import kernel 之前设 FAT_TILES（模块级读取）
_PRE = argparse.ArgumentParser(add_help=False)
_PRE.add_argument("--tiles", default=None)
_pre, _ = _PRE.parse_known_args()
if _pre.tiles:
    os.environ["FAT_TILES"] = _pre.tiles

for p in (_ROOT, os.path.join(_ROOT, "triton_lab"), _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

CONFIGS = {
    "base":     {},
    "preload":  {"enable_preload": True},
    "nopreload": {"enable_preload": False},
    "dcp":      {"enable_dynamic_cv_pipeline": True},
    "both":     {"enable_preload": True, "enable_dynamic_cv_pipeline": True},
}

KERNELS = {
    "fa":     ("kernels.flash_attention.fa_fwd", 30),
    "manual": ("kernels.flash_attention.fa_fwd_manual", 43),
}

CORRECTNESS = [(1, 512, 8, True)]
PERF = (1, 4096, 32, True)
O_ERR_TOL = 2e-2


def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument("--kernels", default="fa,manual")
    ap.add_argument("--configs", default="base,preload,dcp,both")
    ap.add_argument("--tiles", default=None, help="强制 tile，如 32x256 / 64x256")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(_HERE, "ab_launch_result.json"))
    return ap.parse_args()


def golden(q, causal):
    import torch
    n = q.shape[1]
    qf = q.float().permute(0, 2, 1, 3)
    s = qf @ qf.transpose(-1, -2) * (q.shape[3] ** -0.5)
    if causal:
        m = torch.tril(torch.ones(n, n, device=q.device, dtype=torch.bool))
        s = s.masked_fill(~m, float("-inf"))
    return (torch.softmax(s, -1) @ qf).permute(0, 2, 1, 3)


def main():
    args = parse_args()
    import torch
    import torch_npu

    want_kernels = [k.strip() for k in args.kernels.split(",") if k.strip()]
    want_configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    for w in want_kernels:
        if w not in KERNELS:
            raise SystemExit(f"未知 kernel {w}，可选 {list(KERNELS)}")
    for w in want_configs:
        if w not in CONFIGS:
            raise SystemExit(f"未知配置 {w}，可选 {list(CONFIGS)}")

    tile_note = os.environ.get("FAT_TILES") or "自动 _pick_tile"
    print(f"\n{'='*104}")
    print(f"ab_launch: kernels={want_kernels} configs={want_configs}")
    print(f"           tile={tile_note}  iters={args.iters} reps={args.reps}")
    print(f"{'='*104}\n", flush=True)

    torch.manual_seed(0)
    corr = []
    for b, n, h, causal in CORRECTNESS:
        q = torch.randn(b, n, h, 128, dtype=torch.float16, device="npu")
        corr.append((q, causal, golden(q, causal)))
        del q
    pb, pn, ph, pcausal = PERF
    pq = torch.randn(pb, pn, ph, 128, dtype=torch.float16, device="npu")
    pflops = 4 * pb * ph * (pn * (pn + 1) / 2) * 128

    def timeit(fn, iters, reps, warmup):
        for _ in range(warmup):
            fn()
        torch.npu.synchronize()
        samples = []
        for _ in range(reps):
            st = torch.npu.Event(enable_timing=True)
            en = torch.npu.Event(enable_timing=True)
            st.record()
            for _ in range(iters):
                fn()
            en.record()
            torch.npu.synchronize()
            samples.append(st.elapsed_time(en) / iters)
        samples.sort()
        return samples[len(samples) // 2]

    results = []
    hdr = f"{'kernel':<8}{'config':<12}{'max o_err':>11}{'median ms':>12}{'TFLOPS':>9}{'vs base':>10}  status"
    print(hdr)
    print("-" * len(hdr), flush=True)

    for ktag in want_kernels:
        mod_name = KERNELS[ktag][0]
        try:
            mod = __import__(mod_name, fromlist=["fla_fwd"])
        except Exception:
            print(f"{ktag:<8}{'(import)':<12}  ✗ {traceback.format_exc(limit=1).strip().splitlines()[-1][:60]}")
            continue
        base_ms = None
        for ctag in want_configs:
            opts = CONFIGS[ctag]
            if hasattr(mod, "_LAUNCH"):
                mod._LAUNCH.clear()
                mod._LAUNCH.update(opts)
            worst, med, status = float("nan"), float("nan"), "OK"
            try:
                worst = 0.0
                for q, causal, ref in corr:
                    o = mod.fla_fwd(q, q, q, causal=causal)
                    torch.npu.synchronize()
                    worst = max(worst,
                                (o.float() - ref).abs().max().item() / (ref.abs().max().item() + 1e-6))
                if not (worst < O_ERR_TOL):
                    status = "★精度不合格"
                med = timeit(lambda: mod.fla_fwd(pq, pq, pq, causal=pcausal),
                             args.iters, args.reps, args.warmup)
            except Exception:
                status = "✗ " + traceback.format_exc(limit=1).strip().splitlines()[-1][:52]
            tf = pflops / (med * 1e-3) / 1e12 if med == med else float("nan")
            if ctag == "base" and med == med:
                base_ms = med
            ratio = (base_ms / med) if (base_ms and med == med) else float("nan")
            results.append({"kernel": ktag, "config": ctag, "opts": opts,
                            "max_o_err": worst, "median_ms": med, "tflops": tf,
                            "vs_base": ratio, "status": status})
            print(f"{ktag:<8}{ctag:<12}{worst:>11.5f}{med:>12.3f}{tf:>9.1f}"
                  f"{(f'{ratio:.3f}x' if ratio == ratio else '-'):>10}  {status}", flush=True)
        print("-" * len(hdr), flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n明细已写入 {args.out}")


if __name__ == "__main__":
    main()
