#!/usr/bin/env -S uv run
"""汇总 msprof op simulator 的指令级产物（纯本地分析，不需要 NPU）。

用法:
    uv run bench/sim_summary.py                     # 默认读 ./sim_small
    uv run bench/sim_summary.py --dir ~/some/sim    # 指定目录
    uv run bench/sim_summary.py --instr-only        # 只看指令聚合
    uv run bench/sim_summary.py --trace-only        # 只看 pipe 时间线

输入是 `msprof op simulator` 产出的目录，结构：
    <dir>/OPPROF_*/simulator/coreN.veccoreM/
        coreN.veccoreM_instr_exe.csv    逐指令: instr,pipe,call_count,cycles,running_time(us),detail
        trace.json                      按 pipe 分 lane 的时间线

注意: 目前采集只包含 veccore（AIV），没有 cubecore（AIC）。
"""

import argparse
import collections
import csv
import glob
import json
import os
import sys

# 同步类指令：这些是"等"，不是"算"
SYNC_INSTRS = {"BAR", "WAIT_FLAG", "SET_FLAG", "MOV_SRC_TO_DST_ALIGN"}
# 真正的向量/标量计算（用来对照同步开销）
COMPUTE_HINT = ("VCONV", "MOVEMASK", "DIV", "MUL", "ADD", "SUB", "MIN", "MAX",
                "MAXIMUM", "EXP", "VEXP", "VMUL", "VADD")


def find_sim_root(root):
    """返回 (instr_files, trace_files)。"""
    base = os.path.join(root, "OPPROF_*", "simulator")
    instr = sorted(glob.glob(os.path.join(base, "core*.veccore*", "*_instr_exe.csv")))
    trace = sorted(glob.glob(os.path.join(base, "core*.veccore*", "trace.json")))
    if not instr and not trace:
        sys.exit(f"在 {root} 下没找到 OPPROF_*/simulator/core*.veccore*/ 产物")
    return instr, trace


def summarize_instr(files):
    agg = collections.defaultdict(lambda: [0, 0])   # instr -> [calls, cycles]
    per_core = {}
    for f in files:
        tot = 0
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                name = r["instr"]
                calls = int(r["call_count"] or 0)
                cyc = int(r["cycles"] or 0)
                agg[name][0] += calls
                agg[name][1] += cyc
                tot += cyc
        per_core[os.path.basename(os.path.dirname(f))] = tot

    total = sum(v[1] for v in agg.values())
    ncore = len(files)
    print(f"\n{'=' * 68}\n[指令聚合]  cores = {ncore}   总 cycles = {total:,}\n{'=' * 68}")
    print(f"{'instr':<28}{'calls':>8}{'cycles':>12}{'占':>8}{'cyc/call':>10}")
    for name, (c, cy) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:18]:
        print(f"{name:<28}{c:>8}{cy:>12,}{cy / total * 100:>7.1f}%{cy / c:>10.0f}")

    sync = sum(cy for n, (_, cy) in agg.items() if n in SYNC_INSTRS)
    comp = sum(cy for n, (_, cy) in agg.items() if n not in SYNC_INSTRS)
    print(f"\n  同步类(BAR/WAIT_FLAG/SET_FLAG/MOV_SRC_TO_DST_ALIGN) = {sync:>9,}  {sync / total * 100:>5.1f}%")
    print(f"  其余（计算+取数+流控）                              = {comp:>9,}  {comp / total * 100:>5.1f}%")

    for name in ("BAR", "WAIT_FLAG", "SET_FLAG", "MOV_SRC_TO_DST_ALIGN"):
        if name in agg:
            c, cy = agg[name]
            print(f"    {name:<24} {c / ncore:>6.1f} 条/核   {cy / c:>6.0f} cyc/条"
                  f"  ({cy / c / 1.8e3:>5.3f} µs @1.8GHz)")
    vals = sorted(per_core.values())
    if vals:
        print(f"\n  每核 cycles: min={vals[0]:,} 中位={vals[len(vals) // 2]:,} max={vals[-1]:,}")
    return agg, total


def summarize_trace(files):
    if not files:
        print("\n[pipe 时间线] 没有 trace.json，跳过")
        return
    busy = collections.defaultdict(float)
    cnt = collections.defaultdict(int)
    span_sum = 0.0
    covered_sum = 0.0        # 至少有一个 pipe 在忙的时间（区间并集）
    for f in files:
        with open(f) as fh:
            ev = json.load(fh).get("traceEvents", [])
        pids = {e["args"]["name"]: e["pid"] for e in ev if e.get("name") == "process_name"}
        inv = {v: k for k, v in pids.items()}
        xs = [e for e in ev if e.get("ph") == "X"]
        if not xs:
            continue
        lo = min(e["ts"] for e in xs)
        hi = max(e["ts"] + (e.get("dur") or 0) for e in xs)
        # 单位自适应：ts/dur 在不同版本里可能是 ns 也可能是 µs。
        # 本 kernel 单核跨度远小于 10^4 µs，原始值 >1e4 就判定为 ns。
        scale = 1e-3 if (hi - lo) > 1e4 else 1.0
        span_sum += (hi - lo) * scale

        for e in xs:
            busy[inv.get(e["pid"], e["pid"])] += (e.get("dur") or 0) * scale
            cnt[inv.get(e["pid"], e["pid"])] += 1

        # 区间并集 = 真正"有 pipe 在忙"的时长，剩下的才是纯空转
        ivs = sorted((e["ts"] * scale, (e["ts"] + (e.get("dur") or 0)) * scale) for e in xs)
        cov, cur_s, cur_e = 0.0, None, None
        for s, e2 in ivs:
            if cur_s is None:
                cur_s, cur_e = s, e2
            elif s <= cur_e:
                cur_e = max(cur_e, e2)
            else:
                cov += cur_e - cur_s
                cur_s, cur_e = s, e2
        if cur_s is not None:
            cov += cur_e - cur_s
        covered_sum += cov

    n = len(files)
    tot = sum(busy.values())
    print(f"\n{'=' * 68}\n[pipe 时间线]  cores = {n}   平均 span = {span_sum / n:.3f} µs/核\n{'=' * 68}")
    print(f"{'pipe':<14}{'events':>8}{'busy(µs)':>11}{'占 span':>9}")
    for k, v in sorted(busy.items(), key=lambda kv: -kv[1]):
        print(f"{k:<14}{cnt[k]:>8}{v:>11.3f}{v / span_sum * 100:>8.1f}%")
    print(f"{'SUM':<14}{sum(cnt.values()):>8}{tot:>11.3f}{tot / span_sum * 100:>8.1f}%")
    print(f"\n  ⇒ pipe 并行度 = Σpipe_busy / span = {tot / span_sum * 100:.0f}%"
          f"（100% = 完全串行，越高越并行）")
    idle = span_sum - covered_sum
    print(f"  ⇒ 至少一个 pipe 在忙 = {covered_sum / span_sum * 100:.1f}% of span")
    print(f"  ⇒ 纯空转（所有 pipe 都闲着）= {idle / span_sum * 100:.1f}%"
          f"  ≈ {idle / n:.3f} µs/核  ← 这部分就是被同步/等待吃掉的")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="sim_small", help="simulator 产物目录（默认 ./sim_small）")
    ap.add_argument("--instr-only", action="store_true")
    ap.add_argument("--trace-only", action="store_true")
    ap.add_argument("--top", type=int, default=18)
    args = ap.parse_args()

    root = os.path.expanduser(args.dir)
    instr, trace = find_sim_root(root)
    if not args.trace_only:
        summarize_instr(instr)
    if not args.instr_only:
        summarize_trace(trace)


if __name__ == "__main__":
    main()
