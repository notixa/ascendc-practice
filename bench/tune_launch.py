#!/usr/bin/env python3
"""扫描 `fa_fwd.py` 的 launch 级调参，逐个配置做【正确性 + 性能】验证。

为什么是这些参数
----------------
官方 triton-ascend-kernels 仓库（gitcode.com/Ascend/triton-ascend-kernels）的
调优手段里，launch 级参数的调用频次是：

    num_warps                                     25 次  ← 最高频
    num_stages                                    18 次
    multibuffer                                    5 次
    set_workspace_multibuffer                      2 次
    limit_auto_multi_buffer_only_for_local_buffer   2 次
    inject_barrier_all                             1 次（同步问题诊断）

而 `al.scope` / `al.sync_block_*` / `bl.alloc` 的使用次数是 **0**。

官方文档对 multibuffer 的解释（docs/sources/best-practices/chunk_scaled_dot_kkt.md）：
    "multibuffer=True 是 NPU 侧的软件流水开关（类比 GPU 侧 num_stages 的作用），
     把下一个 K 块的 GM→UB load 与当前块的 tl.dot 重叠，掩盖 load 时延。"

注意：这里的 num_stages 是【launch 级编译选项】，不是 tl.range(num_stages=)
那个循环级参数（后者实测无效）。

用法
----
    # 默认网格（num_warps × num_stages × multibuffer）
    python3 bench/tune_launch.py

    # 自定义网格
    python3 bench/tune_launch.py --nw 2,4,8 --ns 1,2,3 --mb 1

    # 加一组 workspace / local-buffer 组合
    python3 bench/tune_launch.py --extra "set_workspace_multibuffer=4" \
                                 --extra "limit_auto_multi_buffer_only_for_local_buffer=0"

    # 强制 tile（UB 紧张时）
    python3 bench/tune_launch.py --tiles 64x256

    # 只测正确性不测性能（快）
    python3 bench/tune_launch.py --no-perf

输出
----
    - 终端表格：配置 / o_err 最大值 / median ms / TFLOPS / 状态
    - JSON：bench/tune_launch_result.json（按 median 升序）
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import traceback

# ── 路径：让本脚本既能从仓库根跑，也能从 bench/ 里跑 ─────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, os.path.join(_ROOT, "triton_lab"), _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)


# ── 正确性用的小 case（N=4096 的 golden 太占内存，只用来计时）─────────────────
CORRECTNESS_CASES = [
    (1, 512, 8, True),
    (1, 512, 8, False),
    (1, 384, 8, True),
]
PERF_CASE = (1, 4096, 32, True)
O_ERR_TOL = 2e-2


def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument("--nw", default="2,4,8",
                    help="num_warps 候选，逗号分隔（默认 2,4,8）")
    ap.add_argument("--ns", default="1,2,3",
                    help="num_stages 候选的 launch 级取值（默认 1,2,3）")
    ap.add_argument("--mb", default="1",
                    help="multibuffer 候选（默认 1=True；写 1,0 两组都测）")
    ap.add_argument("--extra", action="append", default=[],
                    metavar="KEY=VAL",
                    help="额外固定 kwarg，可重复。值按 int/bool/str 自动解析")
    ap.add_argument("--tiles", default=None,
                    help="强制 tile，如 64x256（等价于 FAT_TILES）")
    ap.add_argument("--no-perf", action="store_true", help="只做正确性，不计时")
    ap.add_argument("--iters", type=int, default=20, help="每次采样内层迭代数")
    ap.add_argument("--reps", type=int, default=5, help="采样次数（取中位数）")
    ap.add_argument("--warmup", type=int, default=3, help="预热次数")
    ap.add_argument("--out", default=os.path.join(_HERE, "tune_launch_result.json"))
    return ap.parse_args()


def _parse_val(s: str):
    low = s.strip().lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def _golden(q, causal):
    """与 fa_fwd 同口径的参考实现：scale = d**-0.5，causal 用 tril。"""
    import torch  # 局部导入：脚本顶层不 import torch，避免 --help 时初始化 NPU driver

    n = q.shape[1]
    qf = q.float().permute(0, 2, 1, 3)              # [b,h,n,d]
    s = qf @ qf.transpose(-1, -2) * (q.shape[3] ** -0.5)
    if causal:
        # 与 kernel 的 `k_idx <= q_idx + delta_qk` 一致；delta_qk=0（N_q==N_k）
        m = torch.tril(torch.ones(n, n, device=q.device, dtype=torch.bool))
        s = s.masked_fill(~m, float("-inf"))
    ref = torch.softmax(s, -1) @ qf
    return ref.permute(0, 2, 1, 3)                  # [b,n,h,d]


def main():
    args = parse_args()

    # FAT_TILES 必须在 import fa_fwd 之前设好（模块级读取）
    if args.tiles:
        os.environ["FAT_TILES"] = args.tiles

    import torch
    import torch_npu
    from kernels.flash_attention import fa_fwd as fa

    try:
        from torch_npu.npu import get_device_properties
        n_cores = get_device_properties(0).cube_core_num
    except Exception:
        n_cores = 20

    fixed = {}
    for item in args.extra:
        k, _, v = item.partition("=")
        fixed[k.strip()] = _parse_val(v)

    grid_nw = [int(x) for x in args.nw.split(",") if x.strip()]
    grid_ns = [int(x) for x in args.ns.split(",") if x.strip()]
    grid_mb = [bool(int(x)) for x in args.mb.split(",") if x.strip()]

    combos = []
    for nw, ns, mb in itertools.product(grid_nw, grid_ns, grid_mb):
        opts = dict(fixed)
        opts.update({"num_warps": nw, "num_stages": ns, "multibuffer": mb})
        combos.append(opts)

    fn = fa.fla_fwd
    print(f"\n{'='*92}")
    print(f"tune_launch: {len(combos)} 个配置 | 设备 cube 核数 {n_cores} | "
          f"tile={'FAT_TILES 强制 ' + os.environ['FAT_TILES'] if args.tiles else '自动 _pick_tile'}")
    print(f"正确性 case: {[(f'N{n}', 'causal' if c else 'full') for _, n, _, c in CORRECTNESS_CASES]}")
    print(f"性能   case: B{PERF_CASE[0]} N{PERF_CASE[1]} H{PERF_CASE[2]} causal={PERF_CASE[3]}")
    print(f"{'='*92}\n", flush=True)

    # ── 预先造数据 + golden（只算一次，各配置复用）────────────────────────────
    torch.manual_seed(0)
    correct_data = []
    for b, n, h, causal in CORRECTNESS_CASES:
        q = torch.randn(b, n, h, 128, dtype=torch.float16, device="npu")
        correct_data.append((q, causal, _golden(q, causal)))

    perf_b, perf_n, perf_h, perf_causal = PERF_CASE
    perf_q = None
    if not args.no_perf:
        perf_q = torch.randn(perf_b, perf_n, perf_h, 128, dtype=torch.float16, device="npu")
    perf_flops = 4 * perf_b * perf_h * (perf_n * (perf_n + 1) / 2) * 128

    def time_ms(fn_, iters, reps, warmup):
        for _ in range(warmup):
            fn_()
        torch.npu.synchronize()
        samples = []
        for _ in range(reps):
            s = torch.npu.Event(enable_timing=True)
            e = torch.npu.Event(enable_timing=True)
            s.record()
            for _ in range(iters):
                fn_()
            e.record()
            torch.npu.synchronize()
            samples.append(s.elapsed_time(e) / iters)
        samples.sort()
        return samples[len(samples) // 2]

    results = []
    hdr = f"{'#':>3} {'config':<58} {'max o_err':>10} {'median ms':>11} {'TFLOPS':>8}  status"
    print(hdr)
    print("-" * len(hdr), flush=True)

    for idx, opts in enumerate(combos, 1):
        cfg_str = " ".join(f"{k}={v}" for k, v in opts.items())
        fa._LAUNCH.clear()
        fa._LAUNCH.update(opts)
        max_err = float("nan")
        median_ms = float("nan")
        status = "OK"
        try:
            # ① 正确性
            worst = 0.0
            for q, causal, ref in correct_data:
                o = fn(q, q, q, causal=causal)
                torch.npu.synchronize()
                denom = ref.abs().max().item() + 1e-6
                worst = max(worst, (o.float() - ref).abs().max().item() / denom)
            max_err = worst
            if not (max_err < O_ERR_TOL):
                status = "★精度不合格"
            # ② 性能
            if not args.no_perf:
                median_ms = time_ms(lambda: fn(perf_q, perf_q, perf_q, causal=perf_causal),
                                    args.iters, args.reps, args.warmup)
        except Exception:
            status = "✗ " + traceback.format_exc(limit=1).strip().splitlines()[-1][:40]
        tf = perf_flops / (median_ms * 1e-3) / 1e12 if median_ms == median_ms else float("nan")
        results.append({
            "config": opts,
            "max_o_err": max_err,
            "median_ms": median_ms,
            "tflops": tf,
            "status": status,
        })
        print(f"{idx:>3} {cfg_str:<58} {max_err:>10.5f} {median_ms:>11.3f} {tf:>8.1f}  {status}",
              flush=True)

    # ── 汇总 ────────────────────────────────────────────────────────────────
    ok = [r for r in results if r["status"] == "OK" and r["median_ms"] == r["median_ms"]]
    ok.sort(key=lambda r: r["median_ms"])
    print(f"\n{'='*92}")
    print(f"合格配置 {len(ok)}/{len(results)}（o_err < {O_ERR_TOL}）")
    if ok:
        best = ok[0]
        print(f"\n最优: {best['config']}")
        print(f"      median {best['median_ms']:.3f} ms / {best['tflops']:.1f} TFLOPS "
              f"/ max o_err {best['max_o_err']:.5f}")
        print(f"\nTop 10（按 median 升序）：")
        for r in ok[:10]:
            cfg = " ".join(f"{k}={v}" for k, v in r["config"].items())
            print(f"  {r['median_ms']:>8.3f} ms  {r['tflops']:>7.1f} TF  {cfg}")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n明细已写入 {args.out}")
    print(f"复现最优: FA_NW={ok[0]['config']['num_warps']} FA_NS={ok[0]['config']['num_stages']} "
          f"FA_MB={int(ok[0]['config']['multibuffer'])} python3 bench/triton_check.py --perf"
          if ok else "", flush=True)


if __name__ == "__main__":
    main()
