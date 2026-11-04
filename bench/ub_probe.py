#!/usr/bin/env python3
"""本地测 UB 需求：不同 tile × 不同 bishengir flag 下，bishengir 报的 `requires N bits`。

为什么需要这个
--------------
UB 相关的开关（`--enable-lazy-loading` / `--disable-multi-buffer-on-ub` /
`--multibuffer-mode=[(ub,N)]` …）在**同一个 tile** 上测时间是看不出效果的 ——
UB 的收益只有在**放大 tile** 时才兑现（更大的 BLOCK_K ⇒ KV 重复读更少 ⇒
mte2 更省）。所以正确的判据是：

    「原来爆 UB 的 tile，加了 flag 之后能不能编过」

本脚本用本地 bishengir 直接拿这个判据，不需要 NPU：每个 tile 只编译一次
mlirbc，然后对每组 flag 各跑一次 bishengir，抓 `requires N bits`。

用法
----
    python3 bench/ub_probe.py
    python3 bench/ub_probe.py --tiles 64x256,96x256,128x256,64x512
    python3 bench/ub_probe.py --kernel manual --tiles 32x256,64x256,64x512
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, os.path.join(_ROOT, "bench")):
    if p not in sys.path:
        sys.path.insert(0, p)

UB_BITS = 1572864  # 192 KB

FLAG_SETS = [
    ("base", []),
    ("lazy-loading", ["--enable-lazy-loading"]),
    ("no-ub-multibuf", ["--disable-multi-buffer-on-ub"]),
    ("ub=1", ["--multibuffer-mode=[(ub,1)]"]),
    ("ub=1,gm=2", ["--multibuffer-mode=[(ub,1),(gm,2)]"]),
    ("largest-first", ["--plan-memory-strategy=largest-first"]),
    ("no-tight-couple", ["--disable-tightly-coupled-buffer-reuse"]),
]

KERNELS = {
    "fa": "triton_lab/kernels/flash_attention/fa_fwd.py",
    "manual": "triton_lab/kernels/flash_attention/fa_fwd_manual.py",
}


def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("--kernel", default="fa", choices=list(KERNELS))
    ap.add_argument("--tiles", default="64x256,96x256,128x256,64x512")
    ap.add_argument("--nq", type=int, default=4096)
    ap.add_argument("--nk", type=int, default=4096)
    ap.add_argument("--nvec", type=int, default=1)
    return ap.parse_args()


def main():
    args = parse_args()
    import ir_diff  # 复用抽取 / emit_mlirbc / run_bishengir

    path = os.path.join(_ROOT, KERNELS[args.kernel])
    src = ir_diff.extract_kernel(path)
    tiles = []
    for t in args.tiles.split(","):
        bq, bk = (int(v) for v in t.split("x"))
        tiles.append((bq, bk))

    workdir = tempfile.mkdtemp(prefix="ubprobe_", dir=os.path.join(_ROOT, ".scratch"))
    ce = {"N_q": args.nq, "N_k": args.nk, "delta_qk": args.nk - args.nq,
          "HEAD_DIM": 128, "IS_CAUSAL": True, "NVEC": args.nvec}

    results = {}
    for bq, bk in tiles:
        tag = f"{bq}x{bk}"
        ce2 = dict(ce, BLOCK_Q=bq, BLOCK_K=bk)
        try:
            mlirbc, _ = ir_diff.emit_mlirbc(src, ce2, workdir, f"{args.kernel}_{tag}")
        except SystemExit as e:
            print(f"  [{tag}] mlirbc 失败: {e}")
            continue
        except Exception as e:
            print(f"  [{tag}] mlirbc 异常: {type(e).__name__}: {str(e)[:60]}")
            continue
        row = {}
        for name, flags in FLAG_SETS:
            ir = _run_bishengir(mlirbc, flags)
            m = re.search(r"requires (\d+) bits while (\d+) bits available", ir)
            if m:
                need = int(m.group(1))
                row[name] = (need, need / UB_BITS)
            elif "error" in ir.lower() and "ub overflow" not in ir.lower():
                row[name] = (None, None)  # 别的错误
            else:
                row[name] = (0, 0.0)      # 编过 ⇒ 不需要额外 UB 报告
        results[tag] = row
        print(f"  [{tag}] 采完 {len(row)} 组", flush=True)

    print(f"\n{'='*100}")
    print(f"UB 需求（bits）· kernel={args.kernel} · 上限 {UB_BITS}")
    print(f"{'='*100}")
    hdr = f"{'tile':<10}" + "".join(f"{n:>15}" for n, _ in FLAG_SETS)
    print(hdr)
    print("-" * len(hdr))
    for tag, row in results.items():
        cells = []
        for name, _ in FLAG_SETS:
            need, ratio = row[name]
            if need is None:
                cells.append(f"{'其它错误':>15}")
            elif need == 0:
                cells.append(f"{'✓ 装得下':>15}")
            else:
                cells.append(f"{need/1024:.0f}K {'✗' if ratio > 1 else '✓':<1}".rjust(15))
        print(f"{tag:<10}" + "".join(cells))
    print(f"{'='*100}")
    print("说明：✓ 装得下 = bishengir 未报 ub overflow；✗ = 超限。")
    print("      某 flag 让原本 ✗ 的 tile 变成 ✓，就是这个 flag 真实降低了 UB。")
    shutil.rmtree(workdir, ignore_errors=True)


def _run_bishengir(mlirbc: str, extra_flags: list[str]) -> str:
    import triton
    td = os.path.dirname(triton.__file__)
    bc = os.path.join(td, "backends/ascend/bishengir/bin/bishengir-compile")
    out = tempfile.mktemp(prefix="ir_", dir=os.path.join(_ROOT, ".scratch"))
    # 必须和服务器真实编译的命令行一致（否则 UB 估计差一个数量级）：
    # 服务器实测命令行含 --enable-optimized-metaop=false 与 --enable-auto-blockify-loop
    cmd = [bc, mlirbc, "--target=Ascend910B3",
           "--enable-auto-multi-buffer=True", "--enable-auto-bind-sub-block=True",
           "--enable-optimized-metaop=false", "--enable-auto-blockify-loop",
           "--enable-hfusion-compile=true", "--enable-hivm-compile=true",
           "--enable-triton-kernel-compile=true",
           "-o", out] + list(extra_flags)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    for f in glob.glob(out + "*"):
        try:
            os.remove(f)
        except OSError:
            pass
    return r.stderr + r.stdout


if __name__ == "__main__":
    main()
