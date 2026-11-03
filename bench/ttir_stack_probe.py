#!/usr/bin/env python3
"""本地（无 NPU）用 TTIR 验证「方法一（4 路 grouping）」与「放大 BLOCK_N」的关系。

原理：triton.compile 走到 bishengir 之前，Triton 自己的 pass 已经全部跑完，
      并把 .ttir / .ttadapter 落盘。bishengir 那步会失败（本机缺 torch 头/版本不匹配），
      但我们需要的证据在 TTIR 里就已经齐了：

  V1  4 个 qk 的 live range 是否重叠   -> 看一个 reduce 之前有几个 tt.dot 的结果
  V2  group4 是否等价于 放大 BLOCK_N    -> 看每轮循环里 reduce/exp 的条数
  V3  dot 的数量与结构                 -> 数 tt.dot / scf.for

用法（必须用 triton_lab 的 venv）:
    TRITON_CACHE_DIR=$PWD/.triton-cache \\
    triton_lab/.venv/bin/python bench/ttir_stack_probe.py
"""

import glob
import os
import re
import shutil
import sys
import tempfile

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

DIM = 128
SIG = {"q_ptr": "*fp16", "k_ptr": "*fp16", "v_ptr": "*fp16", "o_ptr": "*fp16",
       "N": "i32", "BLOCK_Q": "constexpr", "BN": "constexpr", "DIM": "constexpr"}


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


PATTERNS = {
    "tt.dot":            r"tt\.dot\b",
    "tt.reduce(max)":    r"tt\.reduce",
    "math.exp":          r"math\.exp",
    "tt.load":           r"tt\.load\b",
    "scf.for":           r"scf\.for",
    "arith.mulf":        r"arith\.mulf",
    "convert fp16":      r"arith\.truncf|arith\.extf",
}


def compile_to_ttir(fn, bq, bn, name):
    dumpdir = tempfile.mkdtemp(prefix=f"dump_{name}_", dir=os.getcwd())
    os.environ["TRITON_KERNEL_DUMP"] = "1"
    os.environ["TRITON_DUMP_DIR"] = dumpdir
    src = ASTSource(fn=fn, signature=SIG,
                    constexprs={"BLOCK_Q": bq, "BN": bn, "DIM": DIM})
    try:
        triton.compile(src, target=GPUTarget("npu", "Ascend910B3", 0))
        note = "编译全程成功"
    except Exception as exc:                              # noqa: BLE001
        note = f"{type(exc).__name__}（预期：bishengir 阶段本机过不去）"

    ttirs = glob.glob(os.path.join(dumpdir, "**", "*.ttir"), recursive=True)
    text = open(ttirs[0]).read() if ttirs else ""
    counts = {k: len(re.findall(p, text)) for k, p in PATTERNS.items()}
    shutil.rmtree(dumpdir, ignore_errors=True)
    return counts, len(text), note


def main():
    cfgs = [("k_seq  BN=64", k_seq, 64, 64, "现状：每轮 64 列"),
            ("k_group4 BN=64", k_group4, 64, 64, "方法一：4 个 64 列小 dot 攒一起（总 256 列）"),
            ("k_seq  BN=256", k_seq, 256, 64, "方法二：1 个 256 列大 dot（总 256 列）")]

    rows = []
    for label, fn, bn, bq, _desc in cfgs:
        counts, n, note = compile_to_ttir(fn, bq, bn, label.split()[0] + str(bn))
        rows.append((label, bn, counts, n, note))

    print("\n" + "=" * 92)
    print("TTIR 结构对比（每轮循环内的算子条数，全部由 Triton 自己生成，未经 bishengir）")
    print("=" * 92)
    keys = list(PATTERNS)
    print(f"{'配置':<16}{'列宽':>5}" + "".join(f"{k:>16}" for k in keys))
    for label, bn, counts, _n, _note in rows:
        print(f"{label:<16}{bn:>5}" + "".join(f"{counts[k]:>16}" for k in keys))

    print("\n说明：")
    for label, _bn, _c, n, note in rows:
        print(f"  {label:<16} TTIR {n:>6} 字符   {note}")

    a = rows[1][2]   # group4
    b = rows[2][2]   # seq BN=256
    print("\n" + "-" * 92)
    print("判据 V2：方法一 vs 方法二（两者每轮总列宽都是 256）")
    for k in keys:
        same = "相同" if a[k] == b[k] else f"不同 ({a[k]} vs {b[k]})"
        print(f"  {k:<16} {same}")
    print("\n  ⇒ 若上面绝大多数“相同”，说明【方法一 ≡ 放大 BLOCK_N】——")
    print("     两者每轮的向量算子条数一样，所以 UB 峰值也一样，")
    print("     「方法一不会爆 UB」的说法不成立。")
    print("  ⇒ 若 group4 的 reduce/exp 条数 ≈ 4×BN256 的 4 倍，说明 group4 反而更差。")


if __name__ == "__main__":
    main()
