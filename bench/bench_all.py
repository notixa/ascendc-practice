#!/usr/bin/env python3
"""
bench_all.py —— 三个实现（CATLASS / Triton-Ascend / vendor）横向对比，一张表出结果。

口径设计（这是本脚本存在的理由）：
  - CATLASS     : 跑 catlass_op/build/fat_bench（C++ + ACL 直调），
                  warmup 10 / iters N / repeats 5 取中位数 —— **纯 device，零 Python launcher**
  - Triton      : 跑 bench_fa.py --only triton --graph --json，
                  取它的 "device-only (纯 replay)" 那一行 —— 与 CATLASS 同口径
  - vendor      : 跑 bench_fa.py --only vendor --json（npu_fusion_attention）
                  注意 vendor 那一路带 torch launcher 开销，但它本身慢，影响相对小

用法：
    python bench/bench_all.py                        # 默认 shape 列表
    python bench/bench_all.py --shapes 4096:8:0      # 单个：N:H:causal
    python bench/bench_all.py --iters 20 --skip-vendor
    python bench/bench_all.py --markdown > out.md

输出：一张 markdown 表 + 每路的 ms / TFLOPS / 相对 vendor 的倍数。
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FAT_BENCH = os.path.join(ROOT, "catlass_lab/catlass_op/build/fat_bench")
BENCH_FA = os.path.join(HERE, "bench_fa.py")

# 默认 shape：(N, H, causal)。B 固定 1（vendor/triton 的 CASES 也只有 B=1 的 8 head 组合之外另算）
DEFAULT_SHAPES = [(512, 8, 0), (512, 8, 1), (1024, 8, 0), (1024, 8, 1),
                  (2048, 8, 0), (2048, 8, 1), (4096, 8, 0), (4096, 8, 1)]

RE_MEDIAN = re.compile(r"PERF: median = ([0-9.]+) ms/iter")
RE_TFLOPS = re.compile(r"PERF: ([0-9.]+) TFLOPS")


def run(cmd, env=None, timeout=1200, quiet=True):
    e = dict(os.environ)
    if env:
        e.update(env)
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=e)
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return 124, "[timeout]"
    except FileNotFoundError as ex:
        return 127, f"[not found] {ex}"


def bench_catlass(n, h, causal, iters, core_num, batch):
    """返回 (ms, tflops) 或 (None, None)"""
    if not os.path.isfile(FAT_BENCH):
        print(f"  [catlass] 找不到 {FAT_BENCH}，先构建 catlass_op", file=sys.stderr)
        return None, None, "?"
    cmd = [FAT_BENCH, str(n), str(h), str(causal), str(core_num), "0", str(iters)]
    rc, out = run(cmd, env={"FAT_BATCH": str(batch)})
    m = RE_MEDIAN.search(out)
    if rc != 0 or not m:
        # 精度失败也给时间（PERF 已打印），只是标一下
        tail = "\n".join(out.strip().splitlines()[-3:])
        print(f"  [catlass] N={n} H={h} c={causal} 未取到 PERF (rc={rc}): {tail}", file=sys.stderr)
        return None, None, "?"
    ms = float(m.group(1))
    t = RE_TFLOPS.search(out)
    tflops = float(t.group(1)) if t else None
    # 顺带把精度抓出来（跳过 golden 时是 "skipped"）
    mae = re.search(r"max_err = ([0-9.]+|skipped)", out)
    return ms, tflops, (mae.group(1) if mae else "?")


def bench_torch_backend(shapes, mode_label, extra_args):
    """跑一次 bench_fa.py 的一个后端，返回 {(causal,N,H,name): (ms, tflops)}"""
    cases = ",".join(f"{n}:{h}" for (n, h, _) in shapes)
    out = {}
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        jpath = tf.name
    try:
        cmd = [sys.executable, BENCH_FA, "--json", jpath, "--cases", cases,
               "--mode", "both"] + extra_args
        rc, txt = run(cmd)
        if not os.path.isfile(jpath):
            print(f"  [{mode_label}] 未产生 json (rc={rc}):\n{txt[-500:]}", file=sys.stderr)
            return out
        with open(jpath) as f:
            rows = json.load(f)
        for r in rows:
            out[(r["causal"], r["n"], r["heads"], r["name"])] = (r["ms"], r["tflops"])
    finally:
        if os.path.exists(jpath):
            os.unlink(jpath)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="*", default=None,
                    help="N:H:causal 列表，如 4096:8:0 1024:8:1；缺省用内置列表")
    ap.add_argument("--iters", type=int, default=20, help="CATLASS 每轮迭代数")
    ap.add_argument("--core-num", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1, help="CATLASS 的 batch（FAT_BATCH）")
    ap.add_argument("--skip-catlass", action="store_true")
    ap.add_argument("--skip-triton", action="store_true")
    ap.add_argument("--skip-vendor", action="store_true")
    ap.add_argument("--markdown", action="store_true", help="只输出 markdown 表")
    args = ap.parse_args()

    if args.shapes:
        shapes = []
        for sp in args.shapes:
            n, h, c = sp.split(":")
            shapes.append((int(n), int(h), int(c)))
    else:
        shapes = DEFAULT_SHAPES

    def say(*a):
        if not args.markdown:
            print(*a)

    say(f"shapes: {len(shapes)} 个；CATLASS iters={args.iters} coreNum={args.core_num} batch={args.batch}")

    # ---- CATLASS ----
    cat = {}
    if not args.skip_catlass:
        say("\n[1/3] CATLASS (fat_bench, 纯 device) ...")
        for (n, h, c) in shapes:
            ms, tf, mae = bench_catlass(n, h, c, args.iters, args.core_num, args.batch)
            if ms is not None:
                cat[(c, n, h)] = (ms, tf, mae)
                say(f"  N={n:<5} H={h} c={c}  {ms:8.3f} ms  "
                    f"{(f'{tf:6.1f} TFLOPS' if tf else '     ? TFLOPS')}  max_err={mae}")

    # ---- Triton（device-only replay）----
    tri = {}
    if not args.skip_triton:
        say("\n[2/3] Triton-Ascend (--graph，取 device-only replay) ...")
        raw = bench_torch_backend(shapes, "triton", ["--only", "triton", "--graph"])
        for (c, n, h, name), (ms, tf) in raw.items():
            if name.endswith("-device"):
                tri[(c, n, h)] = ms
                say(f"  N={n:<5} H={h} c={c}  {ms:8.3f} ms  (replay)")

    # ---- vendor ----
    ven = {}
    if not args.skip_vendor:
        say("\n[3/3] vendor (npu_fusion_attention) ...")
        raw = bench_torch_backend(shapes, "vendor", ["--only", "vendor"])
        for (c, n, h, name), (ms, tf) in raw.items():
            if not name.endswith("-device") and name == "vendor":
                ven[(c, n, h)] = ms
                say(f"  N={n:<5} H={h} c={c}  {ms:8.3f} ms")

    # ---- 汇总 ----
    def tflops_of(n, c, ms):
        if ms is None or ms <= 0:
            return None
        pairs = n * (n + 1) / 2 if c else n * n
        return 4.0 * args.batch * 8 * pairs * 128 / (ms * 1e-3) / 1e12

    print()
    print("| N | H | mode | CATLASS ms | CATLASS TFLOPS | Triton(dev) ms | Triton TFLOPS "
          "| vendor ms | vendor TFLOPS | CATLASS/vendor | Triton/vendor |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for (n, h, c) in shapes:
        k = (c, n, h)
        cm = cat.get(k, (None, None, "?"))[0]
        tm = tri.get(k)
        vm = ven.get(k)
        ct = cat.get(k, (None, None, "?"))[1] or tflops_of(n, c, cm)
        tt = tflops_of(n, c, tm)
        vt = tflops_of(n, c, vm)

        def f(v, fmt="{:.3f}"):
            return fmt.format(v) if v is not None else "—"

        def tf(v):
            return f"{v:.1f}" if v is not None else "—"

        ratio_c = f"{cm / vm:.2f}x" if (cm and vm) else "—"
        ratio_t = f"{tm / vm:.2f}x" if (tm and vm) else "—"
        print(f"| {n} | {h} | {'causal' if c else 'full'} | {f(cm)} | {tf(ct)} | {f(tm)} | "
              f"{tf(tt)} | {f(vm)} | {tf(vt)} | {ratio_c} | {ratio_t} |")

    print()
    print("口径说明：")
    print("  CATLASS  = fat_bench（C++/ACL 直调，warmup10 + iters 次 ×5 轮取中位数，零 Python launcher）")
    print("  Triton   = bench_fa.py --graph 的 device-only replay（不含 Python launcher）")
    print("  vendor   = npu_fusion_attention（带 torch launcher，但它本身慢，影响相对小）")
    print("  FLOPs    = 4 × B × H × pairs × D，pairs = N(N+1)/2 (causal) 或 N² (full)")
    print("  倍数 > 1 表示比 vendor 慢；causal 与 full 的绝对 FLOPs 不同，勿跨行比 TFLOPS")


if __name__ == "__main__":
    main()
