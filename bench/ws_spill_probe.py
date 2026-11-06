#!/usr/bin/env python3
"""验证「显式 workspace 落盘能否真的让 s_i 的 UB 生命周期提前终止」。

只比较两个最小 kernel，两者 dot 数量和形状完全一致，唯一差别是
**每个 dot 之后是否立刻 tl.store 到 workspace**：

    k_hold   : 4 个 dot 全部产出后才消费（s0..s3 同时存活）
    k_spill  : 每个 dot 之后立刻 store 到 ws，再逐块 load 回来消费

判据（决定这个方案能不能成立）：

    k_hold  UB  ~ 4 份 s  (BQ=64, BK=256 ⇒ ≈256 KB ⇒ 报 ub overflow)
    k_spill UB  ~ 1 份 s  (≈64 KB + acc + q ≈ 110~150 KB ⇒ 能编过)

    ⇒ k_spill 能编过、且 requires ≈ 1M bits 量级 ⇒ **live range 确实被 store 终止，方案成立**
    ⇒ 两个都报 256KB 量级 ⇒ PlanMemory 没有提前释放，方案失效

用法（NPU 机器上）:
    python3 bench/ws_spill_probe.py
"""

import os
import re
import subprocess
import sys

D = 128
BQ = 64
BK = 256
STACK = 4
SW = BK * STACK  # 1024 列 / 轮


def run_child(tag, which, bk):
    import torch  # noqa: PLC0415
    import torch_npu  # noqa: E402,F401

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _ws_spill_kernels import k_hold, k_spill  # noqa: PLC0415
    kern = k_hold if which == "k_hold" else k_spill
    q = torch.randn(BQ * D, dtype=torch.float16, device="npu")
    k = torch.randn(4096 * D, dtype=torch.float16, device="npu")
    o = torch.zeros(BQ, dtype=torch.float32, device="npu")
    ws = torch.zeros(20 * 4 * BQ * 512, dtype=torch.float32, device="npu")
    try:
        kern[(20, )](q, k, ws, o, 4096, BLOCK_Q=BQ, BK=bk, DIM=D)
        torch.npu.synchronize()
        print(f"RESULT {tag} OK 0")
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        m = re.search(r"requires (\d+) bits", msg)
        if m:
            kind = "CC" if "cc overflow" in msg else "UB"
            print(f"RESULT {tag} {kind} {int(m.group(1))}")
        else:
            print(f"RESULT {tag} ERR 0")
            print(f"MSG {type(exc).__name__}: {msg.splitlines()[0][:110]}")


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--one":
        run_child(sys.argv[2], sys.argv[3], int(sys.argv[4]))
        return

    print("=" * 78)
    print("k_spill 扫 BK：落盘后 UB 峰值只含 1 份 s，看能撑到多大列宽")
    print(f"BQ={BQ}  STACK=4  1 份 s = BQ*BK*4B  L0C = BQ*BK*4B (上限 128 KB)")
    print("=" * 78)
    print(f"{'BK':>5}{'每轮列宽':>9}{'1 份 s':>9}{'L0C':>8}   {'结果':<28}")
    for bk in (256, 320, 384, 448, 512):
        res = _probe("spill", "k_spill", bk)
        sw = bk * 4
        s_kb = BQ * bk * 4 / 1024
        print(f"{bk:>5}{sw:>9}{s_kb:>8.0f}K{s_kb:>7.0f}K   {res:<28}")

    print("\n  k_hold 参照（不落盘，4 份 s 同时活）：")
    for bk in (256, 512):
        print(
            f"{bk:>5}{bk * 4:>9}{BQ * bk * 4 / 1024:>8.0f}K{BQ * bk * 4 / 1024:>7.0f}K   {_probe('hold', 'k_hold', bk):<28}"
        )


def _probe(tag, which, bk):
    cmd = [sys.executable, os.path.abspath(__file__), "--one", tag, which, str(bk)]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=1800).stdout
    res = next((ln for ln in out.splitlines() if ln.startswith("RESULT")),
               "RESULT ? ? 0")
    msg = next((ln for ln in out.splitlines() if ln.startswith("MSG")), "")
    p = res.split()
    if p[2] == "OK":
        return "编过 ✓"
    if p[2] in ("UB", "CC"):
        return f"{p[2]} overflow {int(p[3]) / 8 / 1024:.1f} KB"
    return f"ERR {msg[:44]}"


if __name__ == "__main__":
    main()
