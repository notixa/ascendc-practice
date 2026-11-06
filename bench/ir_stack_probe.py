#!/usr/bin/env python3
"""验证「方法一（4 路 grouping）」与「方法二（放大 BLOCK_N）」是否等价。

核心思路：只写两个 kernel（`k_seq` 单 dot / `k_group4` 四 dot），用三种配置对比：

    配置               等价于
    k_seq   BN=64     现状（每迭代处理 64 列）
    k_group4 BN=64    方法一（4 个 64 列小 dot 攒在一起）
    k_seq   BN=256    方法二（一个 256 列大 dot）

如果 **k_group4(BN=64) 的 UB == k_seq(BN=256) 的 UB**，就证明：
    方法一 ≡ 放大 BLOCK_N  ⇒ 那篇文「方法一不会爆 UB」的说法不成立，
    因为它和「被认为会爆 UB」的方法二占用完全一样。

EXP-A（小 tile，全部能编过）: 数 IR 里的跨核同步算子，回答「4 个 dot 能不能合并成 1 次 flag」
EXP-B（大 tile，故意让它爆）: 读 bishengir 的 "requires N bits"，回答 UB 峰值

用法（NPU 机器上）:
    python3 bench/ir_stack_probe.py            # A + B
    python3 bench/ir_stack_probe.py --exp a    # 只数同步算子
    python3 bench/ir_stack_probe.py --exp b    # 只看 UB
"""

import argparse
import os
import re
import subprocess
import sys

D = 128  # HEAD_DIM

PATTERNS = {
    "sync_block": r"sync_block",
    "set_flag": r"set_flag|SetFlag",
    "wait_flag": r"wait_flag|WaitFlag",
    "tt.dot": r"tt\.dot|ttng\.dot",
    "scf.for": r"scf\.for",
}

# ------------------------------------------------------------------ kernels
KERNELS = r'''
import triton
import triton.language as tl


@triton.jit
def k_seq(q_ptr, k_ptr, v_ptr, o_ptr, N,
          BLOCK_Q: tl.constexpr, BN: tl.constexpr, DIM: tl.constexpr):
    q = tl.load(q_ptr + tl.arange(0, BLOCK_Q)[:, None] * DIM
                + tl.arange(0, DIM)[None, :])
    acc = tl.zeros([BLOCK_Q, DIM], tl.float32)
    mi = tl.full([BLOCK_Q], -1e38, tl.float32)
    li = tl.zeros([BLOCK_Q], tl.float32)
    kp = tl.make_block_ptr(k_ptr, (N, DIM), (DIM, 1), (0, 0), (BN, DIM), (1, 0))
    vp = tl.make_block_ptr(v_ptr, (N, DIM), (DIM, 1), (0, 0), (BN, DIM), (1, 0))
    for _ in range(0, N, BN):
        k = tl.load(kp)
        v = tl.load(vp)
        s = tl.dot(q, tl.trans(k))
        m_prev = mi
        mi = tl.maximum(m_prev, tl.max(s, -1))
        p = tl.exp(s - mi[:, None].to(tl.float16)).to(tl.float16)
        f = tl.exp(m_prev - mi)
        acc = f[:, None] * acc
        acc = tl.dot(p, v, acc)
        li = f * li + tl.sum(p.to(tl.float32), -1)
        kp = tl.advance(kp, (BN, 0))
        vp = tl.advance(vp, (BN, 0))
    tl.store(o_ptr + tl.arange(0, BLOCK_Q)[:, None] * DIM
             + tl.arange(0, DIM)[None, :], acc.to(tl.float16))


@triton.jit
def k_group4(q_ptr, k_ptr, v_ptr, o_ptr, N,
             BLOCK_Q: tl.constexpr, BN: tl.constexpr, DIM: tl.constexpr):
    q = tl.load(q_ptr + tl.arange(0, BLOCK_Q)[:, None] * DIM
                + tl.arange(0, DIM)[None, :])
    acc = tl.zeros([BLOCK_Q, DIM], tl.float32)
    mi = tl.full([BLOCK_Q], -1e38, tl.float32)
    li = tl.zeros([BLOCK_Q], tl.float32)
    step = 4 * BN
    for base in range(0, N, step):
        k0 = tl.load(tl.make_block_ptr(k_ptr, (N, DIM), (DIM, 1), (base + 0 * BN, 0), (BN, DIM), (1, 0)))
        k1 = tl.load(tl.make_block_ptr(k_ptr, (N, DIM), (DIM, 1), (base + 1 * BN, 0), (BN, DIM), (1, 0)))
        k2 = tl.load(tl.make_block_ptr(k_ptr, (N, DIM), (DIM, 1), (base + 2 * BN, 0), (BN, DIM), (1, 0)))
        k3 = tl.load(tl.make_block_ptr(k_ptr, (N, DIM), (DIM, 1), (base + 3 * BN, 0), (BN, DIM), (1, 0)))
        s0 = tl.dot(q, tl.trans(k0))
        s1 = tl.dot(q, tl.trans(k1))
        s2 = tl.dot(q, tl.trans(k2))
        s3 = tl.dot(q, tl.trans(k3))
        m_prev = mi
        m_batch = tl.maximum(tl.maximum(tl.max(s0, -1), tl.max(s1, -1)),
                             tl.maximum(tl.max(s2, -1), tl.max(s3, -1)))
        mi = tl.maximum(m_prev, m_batch)
        p0 = tl.exp(s0 - mi[:, None].to(tl.float16)).to(tl.float16)
        p1 = tl.exp(s1 - mi[:, None].to(tl.float16)).to(tl.float16)
        p2 = tl.exp(s2 - mi[:, None].to(tl.float16)).to(tl.float16)
        p3 = tl.exp(s3 - mi[:, None].to(tl.float16)).to(tl.float16)
        f = tl.exp(m_prev - mi)
        acc = f[:, None] * acc
        li = f * li + (tl.sum(p0.to(tl.float32), -1) + tl.sum(p1.to(tl.float32), -1)
                       + tl.sum(p2.to(tl.float32), -1) + tl.sum(p3.to(tl.float32), -1))
        v0 = tl.load(tl.make_block_ptr(v_ptr, (N, DIM), (DIM, 1), (base + 0 * BN, 0), (BN, DIM), (1, 0)))
        v1 = tl.load(tl.make_block_ptr(v_ptr, (N, DIM), (DIM, 1), (base + 1 * BN, 0), (BN, DIM), (1, 0)))
        v2 = tl.load(tl.make_block_ptr(v_ptr, (N, DIM), (DIM, 1), (base + 2 * BN, 0), (BN, DIM), (1, 0)))
        v3 = tl.load(tl.make_block_ptr(v_ptr, (N, DIM), (DIM, 1), (base + 3 * BN, 0), (BN, DIM), (1, 0)))
        acc = tl.dot(p0, v0, acc)
        acc = tl.dot(p1, v1, acc)
        acc = tl.dot(p2, v2, acc)
        acc = tl.dot(p3, v3, acc)
    tl.store(o_ptr + tl.arange(0, BLOCK_Q)[:, None] * DIM
             + tl.arange(0, DIM)[None, :], acc.to(tl.float16))
'''


# ------------------------------------------------------------- child runner
def run_child(tag, which, bn, bq, n):
    """子进程：编译一次，抓 stderr 里的 IR 统计 + UB 数字。"""
    import torch  # noqa: PLC0415
    import torch_npu  # noqa: E402,F401
    import triton  # noqa: PLC0415

    ns = {}
    exec(compile(KERNELS, "<probe>", "exec"), ns)  # noqa: S102
    kern = ns[which]

    q = torch.randn(bq * D, dtype=torch.float16, device="npu")
    k = torch.randn(n * D, dtype=torch.float16, device="npu")
    v = torch.randn(n * D, dtype=torch.float16, device="npu")
    o = torch.zeros(bq * D, dtype=torch.float16, device="npu")
    try:
        kern[(1, )](q, k, v, o, n, BLOCK_Q=bq, BN=bn, DIM=D)
        torch.npu.synchronize()
        print(f"RESULT {tag} OK 0")
    except Exception as exc:  # noqa: BLE001
        m = re.search(r"requires (\d+) bits", str(exc))
        flag = int(m.group(1)) if m else -1
        print(f"RESULT {tag} {'UB' if m else 'ERR'} {flag}")
        if not m:
            print(f"MSG {type(exc).__name__}: {str(exc).splitlines()[0][:110]}")


# ------------------------------------------------------------ parent runner
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="both", choices=["a", "b", "both"])
    ap.add_argument("--one", nargs=4, metavar=("TAG", "KERN", "BN", "BQ"), default=None)
    ap.add_argument("--n", type=int, default=1024)
    args = ap.parse_args()

    if args.one:
        run_child(args.one[0], args.one[1], int(args.one[2]), int(args.one[3]), args.n)
        return

    if args.exp in ("a", "both"):
        print("=" * 74)
        print("EXP-A  小 tile（全部编得过）⇒ 数 IR 里的跨核同步算子")
        print("=" * 74)
        cfgs = [("seq_BN64", "k_seq", 64, 64), ("grp4_BN64", "k_group4", 64, 64),
                ("seq_BN256", "k_seq", 256, 64)]
        for tag, kern, bn, bq in cfgs:
            cmd = [
                sys.executable,
                os.path.abspath(__file__), "--one", tag, kern,
                str(bn),
                str(bq)
            ]
            env = dict(os.environ, MLIR_ENABLE_DUMP="1", TRITON_ALWAYS_COMPILE="1")
            out = subprocess.run(cmd,
                                 capture_output=True,
                                 text=True,
                                 timeout=1800,
                                 env=env)
            res = next(
                (ln for ln in out.stdout.splitlines() if ln.startswith("RESULT")),
                "RESULT ? ? ?")
            counts = {k: len(re.findall(p, out.stderr)) for k, p in PATTERNS.items()}
            print(f"\n  {tag:<12} {res.split(maxsplit=3)[-1]}")
            for k, v in counts.items():
                print(f"      {k:<12} {v}")

    if args.exp in ("b", "both"):
        print("\n" + "=" * 74)
        print("EXP-B  大 tile（故意让它爆）⇒ 读 bishengir 的 requires N bits")
        print("=" * 74)
        cfgs = [("seq_BN64", "k_seq", 64, 64), ("grp4_BN64(=256列)", "k_group4", 64, 64),
                ("seq_BN256", "k_seq", 256, 64), ("seq_BN512", "k_seq", 512, 64)]
        for tag, kern, bn, bq in cfgs:
            cmd = [
                sys.executable,
                os.path.abspath(__file__), "--one", tag, kern,
                str(bn),
                str(bq)
            ]
            env = dict(os.environ, TRITON_ALWAYS_COMPILE="1")
            out = subprocess.run(cmd,
                                 capture_output=True,
                                 text=True,
                                 timeout=1800,
                                 env=env)
            res = next(
                (ln for ln in out.stdout.splitlines() if ln.startswith("RESULT")),
                "RESULT ? ? ?")
            msg = next((ln for ln in out.stdout.splitlines() if ln.startswith("MSG")),
                       "")
            p = res.split()
            if p[2] == "OK":
                print(f"  {tag:<20} 编过")
            elif p[2] == "UB":
                print(f"  {tag:<20} UB {int(p[3]) / 8 / 1024:.1f} KB")
            else:
                print(f"  {tag:<20} ERR  {msg[:80]}")


if __name__ == "__main__":
    main()
