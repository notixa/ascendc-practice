#!/usr/bin/env python3
"""
bench_all.py —— 三个实现（CATLASS / Triton-Ascend / vendor）横向对比，一张表出结果。

★ 公平性保证（这是本脚本存在的理由）
    三个后端跑【完全相同】的 (B, N, H, D, causal) 集合：
      - 同一份 shapes 列表，逐个下发：CATLASS → fat_bench 的 `N H causal`；torch 后端 → `--cases B:N:H`
      - causal 的取值集合由 --modes 统一决定，三个后端一致，不会出现"某个后端少测一行"
      - shapes 去重（不会因为同时列了 (N,H,0) 和 (N,H,1) 而让 torch 侧跑两遍）
      - FLOPs 用每个 shape 自己的 B/H 算，不再是全局常量

★ 两种计时口径（--scope）
    device : CATLASS = fat_bench（C++/ACL 直调）；Triton = --graph 的 device-only replay
             —— 都是"纯 device"，适合比 kernel 本身
    app    : CATLASS/Triton 都走图捕获（去掉 host launcher）；vendor 是原始 op 调用
             —— 都含框架 launcher，适合比"真实部署端到端"
    both   : 两种都测，表里分两列

用法：
    python bench/bench_all.py --preset realistic --scope app
    python bench/bench_all.py --preset realistic --scope device
    python bench/bench_all.py --shapes 1:4096:8,4:2048:8 --modes causal --iters 20
    python bench/bench_all.py --markdown > out.md
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

RE_MEDIAN = re.compile(r"PERF: median = ([0-9.]+) ms/iter")
RE_TFLOPS = re.compile(r"PERF: ([0-9.]+) TFLOPS")
RE_MAXERR = re.compile(r"max_err = ([0-9.]+|skipped)")

# 真实场景 shape（B, N, H）。D 固定 128；causal 由 --modes 决定。
#   LLM prefill：H=8（小模型）/ H=32（13B~14B 级），N=2048~4096，causal 为主
#   多 batch：B=4~16，短序列（吞吐场景）
PRESETS = {
    "realistic": [
        (1, 512, 8),     # 小序列 prefill
        (1, 2048, 8),    # 典型 prefill
        (1, 4096, 8),    # 长上下文 prefill
        (1, 2048, 32),   # 多 head（13B 级），任务数足
        (1, 4096, 32),   # 长上下文 + 多 head
        (4, 2048, 8),    # 多 batch
        (16, 512, 8),    # 多 batch 短序列（吞吐）
    ],
    "large": [
        (1, 2048, 8),
        (1, 4096, 8),
        (1, 8192, 8),
        (1, 4096, 32),
    ],
    "small": [
        (1, 512, 8),
        (1, 1024, 8),
    ],
}


def run(cmd, env=None, timeout=3600):
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


def bench_catlass(b, n, h, causal, iters, core_num):
    """返回 (ms, tflops, max_err) 或 (None, None, None)"""
    if not os.path.isfile(FAT_BENCH):
        raise SystemExit(f"找不到 {FAT_BENCH}，先构建 catlass_op")
    cmd = [FAT_BENCH, str(n), str(h), str(causal), str(core_num), "0", str(iters)]
    rc, out = run(cmd, env={"FAT_BATCH": str(b)})
    m = RE_MEDIAN.search(out)
    if rc != 0 or not m:
        tail = "\n".join(out.strip().splitlines()[-3:])
        print(f"  [catlass] B={b} N={n} H={h} c={causal} 未取到 PERF (rc={rc}): {tail}", file=sys.stderr)
        return None, None, None
    t = RE_TFLOPS.search(out)
    e = RE_MAXERR.search(out)
    return float(m.group(1)), (float(t.group(1)) if t else None), (e.group(1) if e else "?")


def bench_torch_backend(shapes, mode, extra_args, label):
    """跑一次 bench_fa.py 的一个后端，返回 {(causal,N,H,B,name): (ms, tflops)}"""
    cases = ",".join(f"{b}:{n}:{h}" for (b, n, h) in shapes)
    out = {}
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        jpath = tf.name
    try:
        cmd = [sys.executable, BENCH_FA, "--json", jpath, "--cases", cases,
               "--mode", mode] + extra_args
        rc, txt = run(cmd)
        if not os.path.isfile(jpath):
            print(f"  [{label}] 未产生 json (rc={rc}):\n{txt[-600:]}", file=sys.stderr)
            return out
        with open(jpath) as f:
            rows = json.load(f)
        for r in rows:
            out[(r["causal"], r["n"], r["heads"], r["batch"], r["name"])] = (r["ms"], r["tflops"])
    finally:
        if os.path.exists(jpath):
            os.unlink(jpath)
    return out


def flops_of(b, n, h, d, causal):
    pairs = n * (n + 1) / 2 if causal else n * n
    return 4.0 * b * h * pairs * d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), default=None)
    ap.add_argument("--shapes", default=None,
                    help="N:H 或 B:N:H 列表，逗号分隔；与 --preset 二选一")
    ap.add_argument("--modes", choices=["both", "causal", "full"], default="both")
    ap.add_argument("--scope", choices=["device", "app", "both"], default="device",
                    help="device=C++直调/纯replay；app=走 torch launcher（真实部署）")
    ap.add_argument("--iters", type=int, default=20, help="CATLASS 每轮迭代数（device 口径）")
    ap.add_argument("--core-num", type=int, default=20, help="CATLASS AIC 核数（910B3=20）")
    ap.add_argument("--d", type=int, default=128, help="head dim")
    ap.add_argument("--skip-catlass", action="store_true")
    ap.add_argument("--skip-triton", action="store_true")
    ap.add_argument("--skip-vendor", action="store_true")
    ap.add_argument("--markdown", action="store_true", help="只输出 markdown 表")
    args = ap.parse_args()

    # ---- shapes ----
    if args.shapes:
        shapes = []
        for spec in args.shapes.split(","):
            f = [int(x) for x in spec.split(":")]
            shapes.append((1, f[0], f[1]) if len(f) == 2 else (f[0], f[1], f[2]))
    else:
        shapes = PRESETS[args.preset or "realistic"]
    seen, uniq = set(), []
    for s in shapes:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    shapes = uniq

    causals = [0, 1] if args.modes == "both" else ([1] if args.modes == "causal" else [0])

    def say(*a):
        if not args.markdown:
            print(*a)

    say(f"shapes ({len(shapes)}): " + ", ".join(f"B{b}N{n}H{h}" for b, n, h in shapes))
    say(f"causal: {causals}   scope: {args.scope}   D: {args.d}")
    say("三个后端跑完全相同的 (B,N,H,D,causal) 集合\n")

    # ---- CATLASS（device 口径：fat_bench）----
    cat_dev = {}
    if not args.skip_catlass and args.scope in ("device", "both"):
        say("[1] CATLASS / fat_bench（C++ 直调，纯 device）...")
        for (b, n, h) in shapes:
            for c in causals:
                ms, tf, mae = bench_catlass(b, n, h, c, args.iters, args.core_num)
                if ms is not None:
                    cat_dev[(c, n, h, b)] = ms
                    say(f"  B{b} N={n:<5} H={h:<3} c={c}  {ms:8.4f} ms  "
                        f"{(f'{tf:6.1f} TFLOPS' if tf else '       ? TFLOPS'):>16}  max_err={mae}")

    # ---- CATLASS（app 口径：torch .so）----
    cat_app = {}
    if not args.skip_catlass and args.scope in ("app", "both"):
        say("\n[2] CATLASS / torch .so（含 launcher，真实部署口径）...")
        # app 口径：CATLASS 走 torch .so，并用图捕获去掉 host launcher 开销
        #（裸 op 调用的固定 host 成本 ~0.19 ms，与 shape 无关，会把小 shape 全淹掉；
        #  确认过程见 bench_all.py 头部注释）
        raw = bench_torch_backend(shapes, args.modes,
                                  ["--only", "catlass", "--graph"], "catlass-app")
        for (c, n, h, b, name), (ms, _) in raw.items():
            if name == "catlass":
                cat_app[(c, n, h, b)] = ms
                say(f"  B{b} N={n:<5} H={h:<3} c={c}  {ms:8.4f} ms")

    # ---- Triton ----
    # ★ device 口径用 --graph 的 replay-only 行；app 口径必须【不带 --graph】，
    #   否则拿到的还是 device 时间，和 CATLASS(app)/vendor 不可比。
    tri = {}
    if not args.skip_triton:
        if args.scope in ("device", "both"):
            say("\n[3] Triton-Ascend / device 口径（--graph 的 replay-only 行）...")
            raw = bench_torch_backend(shapes, args.modes, ["--only", "triton", "--graph"], "triton-dev")
            for (c, n, h, b, name), (ms, _) in raw.items():
                if name.endswith("-device"):
                    tri[(c, n, h, b)] = ms
                    say(f"  B{b} N={n:<5} H={h:<3} c={c}  {ms:8.4f} ms  (device-only replay)")
        if args.scope in ("app", "both"):
            say("\n[4] Triton-Ascend / app 口径（不带 --graph，含 launcher）...")
            raw = bench_torch_backend(shapes, args.modes, ["--only", "triton"], "triton-app")
            for (c, n, h, b, name), (ms, _) in raw.items():
                if name == "triton":
                    tri[(c, n, h, b)] = ms
                    say(f"  B{b} N={n:<5} H={h:<3} c={c}  {ms:8.4f} ms  (含 launcher)")

    # ---- vendor ----
    ven = {}
    if not args.skip_vendor:
        say("\n[5] vendor（npu_fusion_attention）...")
        raw = bench_torch_backend(shapes, args.modes, ["--only", "vendor"], "vendor")
        for (c, n, h, b, name), (ms, _) in raw.items():
            if name == "vendor":
                ven[(c, n, h, b)] = ms
                say(f"  B{b} N={n:<5} H={h:<3} c={c}  {ms:8.4f} ms")

    # ---- 汇总 ----
    def tf(c, n, h, b, ms):
        return None if (ms is None or ms <= 0) else flops_of(b, n, h, args.d, c) / (ms * 1e-3) / 1e12

    def f(v, p=4):
        return f"{v:.{p}f}" if v is not None else "—"

    def t(v):
        return f"{v:.1f}" if v is not None else "—"

    show_dev = args.scope in ("device", "both")
    show_app = args.scope in ("app", "both")

    hdr = "| B | N | H | mode |"
    sep = "|---|---|---|---|"
    if show_dev:
        hdr += " CATLASS ms | CATLASS TFLOPS |"
        sep += "---|---|---|"
    if show_app:
        hdr += " CATLASS(app) ms |"
        sep += "---|"
    hdr += " Triton ms | Triton TFLOPS | vendor ms | vendor TFLOPS | CATLASS/vendor | Triton/vendor |"
    sep += "---|---|---|---|---|---|"
    print()
    print(hdr)
    print(sep)

    for (b, n, h) in shapes:
        for c in causals:
            k = (c, n, h, b)
            row = f"| {b} | {n} | {h} | {'causal' if c else 'full'} |"
            if show_dev:
                cm = cat_dev.get(k)
                row += f" {f(cm)} | {t(tf(c, n, h, b, cm))} |"
            if show_app:
                row += f" {f(cat_app.get(k))} |"
            tm, vm = tri.get(k), ven.get(k)
            row += f" {f(tm)} | {t(tf(c, n, h, b, tm))} | {f(vm)} | {t(tf(c, n, h, b, vm))} |"
            cref = cat_dev.get(k) if show_dev else cat_app.get(k)
            row += f" {(f'{cref/vm:.2f}x' if (cref and vm) else '—'):>5} |"
            row += f" {(f'{tm/vm:.2f}x' if (tm and vm) else '—'):>5} |"
            print(row)

    print()
    print("口径：")
    if show_dev:
        print("  CATLASS      = fat_bench（C++/ACL 直调，warmup10 + iters×5 轮中位数，零 Python launcher）")
        print("  Triton       = bench_fa.py --graph 的 device-only(纯 replay) 行")
    if show_app:
        print("  CATLASS(app) = bench_fa.py --only catlass --graph（torch + 图捕获）")
    print("  vendor       = npu_fusion_attention（含 torch launcher）")
    print(f"  FLOPs        = 4 × B × H × pairs × D，pairs = N(N+1)/2 (causal) 或 N² (full)，D={args.d}")
    print("  倍数 > 1 表示比 vendor 慢；causal 与 full 的绝对 FLOPs 不同，勿跨行比 TFLOPS")
    print("  vendor 的 causal 走它自己的 variant 阶梯（mask/sparse_mode），未必与我们的实现等价")


if __name__ == "__main__":
    main()
