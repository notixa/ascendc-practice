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
SW = BK * STACK          # 1024 列 / 轮



def run_child(tag, which):
    import torch          # noqa: PLC0415
    import torch_npu      # noqa: E402,F401

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _ws_spill_kernels import k_hold, k_spill          # noqa: PLC0415
    kern = k_hold if which == "k_hold" else k_spill
    q = torch.randn(BQ * D, dtype=torch.float16, device="npu")
    k = torch.randn(4096 * D, dtype=torch.float16, device="npu")
    o = torch.zeros(BQ, dtype=torch.float32, device="npu")
    ws = torch.zeros(20 * 4 * BQ * BK, dtype=torch.float32, device="npu")
    try:
        kern[(20,)](q, k, ws, o, 4096, BLOCK_Q=BQ, BK=BK, DIM=D)
        torch.npu.synchronize()
        print(f"RESULT {tag} OK 0")
    except Exception as exc:                                # noqa: BLE001
        m = re.search(r"requires (\d+) bits", str(exc))
        print(f"RESULT {tag} {'UB' if m else 'ERR'} {int(m.group(1)) if m else 0}")
        if not m:
            print(f"MSG {type(exc).__name__}: {str(exc).splitlines()[0][:110]}")


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--one":
        run_child(sys.argv[2], "k_hold" if sys.argv[2] == "hold" else "k_spill")
        return

    print("=" * 72)
    print(f"BQ={BQ}  BK={BK}  STACK={STACK}  每轮总列宽={SW}  1 份 s = 64 KB")
    print("=" * 72)
    for tag, which in (("hold", "k_hold"), ("spill", "k_spill")):
        cmd = [sys.executable, os.path.abspath(__file__), "--one", tag]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=1800).stdout
        res = next((ln for ln in out.splitlines() if ln.startswith("RESULT")), "RESULT ? ? ?")
        msg = next((ln for ln in out.splitlines() if ln.startswith("MSG")), "")
        p = res.split()
        label = "k_hold  (4 份 s 同时活)" if tag == "hold" else "k_spill (每个 dot 后落盘)"
        if p[2] == "OK":
            print(f"  {label:<28} 编过")
        elif p[2] == "UB":
            bits = int(p[3])
            print(f"  {label:<28} UB {bits / 8 / 1024:>7.1f} KB   ({bits / 8 / 1024 / 64:.1f} 份 s)")
        else:
            print(f"  {label:<28} ERR  {msg[:70]}")
    print("\n判据：spill 的 UB 若 ≈1 份 s(64KB)+acc+q ≈ 110~150 KB ⇒ 方案成立（live range 被终止）")
    print("     若 spill 仍 ≈4 份 s(256KB) ⇒ PlanMemory 没释放，方案失效")


if __name__ == "__main__":
    main()
