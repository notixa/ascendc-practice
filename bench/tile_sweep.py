#!/usr/bin/env python3
"""扫 (BLOCK_Q, BLOCK_K) 组合：triton-ascend 只要求 16 的倍数，不必是 2 的幂。

每个 tile 在【独立子进程】里跑（FAT_TILES 在 import 时读取，且 UB overflow
会污染进程状态），父进程汇总成一张按耗时排序的表。

用法（NPU 机器上）:
    python3 bench/tile_sweep.py                        # 默认候选表，N=4096 H=32 causal
    python3 bench/tile_sweep.py --n 512 --h 8          # 小 shape
    python3 bench/tile_sweep.py --tiles 64x256,96x128,80x192
    python3 bench/tile_sweep.py --causal 0             # 非 causal

子进程模式（内部用，不要直接调）:
    python3 bench/tile_sweep.py --one 64x256 --n 4096 --h 32 --causal 1
"""

import argparse
import os
import re
import subprocess
import sys

# 全部是 16 的倍数；刻意混入大量非 2 的幂
DEFAULT_TILES = [
    # 面积 16384（当前 UB 天花板附近）
    "64x256",   # 当前最优
    "128x128",
    "112x144",
    "96x160",
    "80x192",
    "144x112",
    # 面积 12288~14336
    "96x128",   # 旧基线
    "112x128",
    "128x112",
    "64x224",
    "48x256",
    "80x160",
    "96x144",
    # 面积 8192~12288
    "64x192",
    "48x192",
    "32x256",
    # 面积更大（大概率 UB 超限，用来探天花板）
    "96x192",
    "128x160",
    "80x224",
]


# ---------------------------------------------------------------- child mode
def run_one(tile, n, h, causal, iters, reps):
    os.environ["FAT_TILES"] = tile                       # 必须在 import 之前
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "triton_lab"))
    import torch                                        # noqa: PLC0415
    import torch_npu                                    # noqa: E402,F401
    from kernels.flash_attention.fa_fwd import fla_fwd  # noqa: PLC0415

    bq, bk = (int(v) for v in tile.split("x"))
    q = torch.randn(1, n, h, 128, dtype=torch.float16, device="npu")
    try:
        o = fla_fwd(q, q, q, causal=bool(causal))
        torch.npu.synchronize()
    except Exception as exc:                            # noqa: BLE001
        msg = str(exc)
        m = re.search(r"requires (\d+) bits while (\d+) bits", msg)
        if m:
            print(f"RESULT {bq} {bk} UB {int(m.group(1))} 0 0")
        else:
            print(f"RESULT {bq} {bk} ERR 0 0 0 {type(exc).__name__}: {msg.splitlines()[0][:80]}")
        return

    # 正确性（拿第 0 行前 8 列粗查，避免父进程再算 golden）
    import math                                         # noqa: PLC0415
    s = torch.einsum("bqhd,bkhd->bhqk", q.float(), q.float()) / math.sqrt(128)
    if causal:
        m2 = torch.triu(torch.ones(n, n, dtype=torch.bool, device=s.device), 1)
        s = s.masked_fill(m2, float("-inf"))
    ref = torch.einsum("bhqk,bkhd->bqhd", torch.softmax(s, -1), q.float())
    err = (o.float() - ref).abs().max().item()

    def bench(f, warm=5, it=iters, rp=reps):
        for _ in range(warm):
            f()
        torch.npu.synchronize()
        ts = []
        for _ in range(rp):
            a = torch.npu.Event(enable_timing=True)
            z = torch.npu.Event(enable_timing=True)
            a.record()
            for _ in range(it):
                f()
            z.record()
            torch.npu.synchronize()
            ts.append(a.elapsed_time(z) / it)
        ts.sort()
        return ts[len(ts) // 2]

    ms = bench(lambda: fla_fwd(q, q, q, causal=bool(causal)))
    print(f"RESULT {bq} {bk} OK 0 {ms:.4f} {err:.5f}")


# --------------------------------------------------------------- parent mode
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default=None, help="逗号分隔，如 64x256,96x128")
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--h", type=int, default=32)
    ap.add_argument("--causal", type=int, default=1)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--one", default=None, help="（内部）子进程模式，跑单个 tile")
    args = ap.parse_args()

    if args.one:
        run_one(args.one, args.n, args.h, args.causal, args.iters, args.reps)
        return

    tiles = [t.strip() for t in args.tiles.split(",")] if args.tiles else DEFAULT_TILES
    flops = 4 * 1 * args.h * (args.n * (args.n + 1) / 2 if args.causal else args.n * args.n) * 128

    rows = []
    for i, t in enumerate(tiles, 1):
        cmd = [sys.executable, os.path.abspath(__file__), "--one", t,
               "--n", str(args.n), "--h", str(args.h), "--causal", str(args.causal),
               "--iters", str(args.iters), "--reps", str(args.reps)]
        print(f"[{i}/{len(tiles)}] {t:<10}", end=" ", flush=True)
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=900).stdout
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
            rows.append((t, "TIMEOUT", None, None, ""))
            continue
        line = next((ln for ln in out.splitlines() if ln.startswith("RESULT")), None)
        if not line:
            print("无结果")
            rows.append((t, "NORESULT", None, None, ""))
            continue
        p = line.split()
        bq, bk, st = p[1], p[2], p[3]
        if st == "UB":
            print(f"UB {int(p[4]) / 8 / 1024:.1f} KB")
            rows.append((t, "UB", int(p[4]), None, ""))
        elif st == "ERR":
            print(f"ERR  {' '.join(p[6:])[:60]}")
            rows.append((t, "ERR", None, None, " ".join(p[6:])[:60]))
        else:
            ms, err = float(p[5]), float(p[6])
            print(f"{ms:.3f} ms  {flops / (ms * 1e-3) / 1e12:.1f} TF  err={err:.4f}")
            rows.append((t, "OK", None, ms, err))

    print("\n" + "=" * 74)
    print(f"shape: B1 N{args.n} H{args.h} causal={bool(args.causal)}   "
          f"(按耗时排序，16 的倍数即可，不必是 2 的幂)")
    print("=" * 74)
    print(f"{'tile':<10}{'BQ*BK':>8}{'状态':>8}{'ms':>10}{'TFLOPS':>9}{'max_err':>10}")
    ok = [r for r in rows if r[1] == "OK"]
    bad = [r for r in rows if r[1] != "OK"]
    for t, st, ub, ms, err in sorted(ok, key=lambda r: r[3]):
        bq, bk = (int(v) for v in t.split("x"))
        print(f"{t:<10}{bq * bk:>8}{'OK':>8}{ms:>10.3f}"
              f"{flops / (ms * 1e-3) / 1e12:>9.1f}{err:>10.5f}")
    for t, st, ub, ms, err in bad:
        bq, bk = (int(v) for v in t.split("x"))
        extra = f"{ub / 8 / 1024:.1f} KB" if ub else err
        print(f"{t:<10}{bq * bk:>8}{st:>8}{'—':>10}{'—':>9}{extra:>10}")

    if ok:
        best = min(ok, key=lambda r: r[3])
        print(f"\n⇒ 最快: {best[0]}  {best[3]:.3f} ms  "
              f"{flops / (best[3] * 1e-3) / 1e12:.1f} TFLOPS  (max_err={best[4]:.5f})")
        print(f"   把它加进 _pick_tile 前，记得确认它在其它 shape 上也 OK。")


if __name__ == "__main__":
    main()
