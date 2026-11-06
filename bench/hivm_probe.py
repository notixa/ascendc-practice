#!/usr/bin/env python3
"""本地（无 NPU）生成并分析华为 HIVM IR：直接数跨核同步、mma、fixpipe、UB 峰值。

链路：
    Triton AST → TTIR/ttadapter/mlirbc   （本地 triton 3.6.0 完成）
    mlirbc     → HIVM IR                 （后端自带的 bishengir 2.0.0，本地完成）
    HIVM IR    → 各项统计

用法（必须用 triton_lab 的 venv）:
    TRITON_CACHE_DIR=$PWD/.triton-cache \\
    triton_lab/.venv/bin/python bench/hivm_probe.py
"""

import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ttir_stack_probe import SIG, k_group4, k_seq  # noqa: E402

BISHENGIR = os.path.join(
    os.path.dirname(triton.__file__),
    "backends",
    "ascend",
    "bishengir",
    "bin",
    "bishengir-compile",
)
BS_FLAGS = [
    "--target=Ascend910B3",
    "--enable-hfusion-compile=true",
    "--enable-hivm-compile=true",
    "--enable-triton-kernel-compile=true",
    "--enable-auto-multi-buffer=True",
    "--enable-auto-bind-sub-block=True",
    "--bishengir-print-ir-after=hivm-graph-sync-solver",
]

COUNTERS = {
    "sync_block": r"hivm\.hir\.sync_block_(?:wait|set)",
    "wait_flag": r"hivm\.hir\.wait_flag",
    "set_flag": r"hivm\.hir\.set_flag",
    "pipe_barrier": r"hivm\.hir\.pipe_barrier",
    "mma_tile": r"mma_tile",
    "fixpipe": r"fixpipe",
    "vector_op":
    r"hivm\.hir\.(?:vadd|vmul|vmax|vmin|vexp|vln|vcmp|vsel|vbrc|vconv|vsub)",
}


def emit_mlirbc(fn, bq, bn, tag):
    """本地跑到 bishengir 之前，拿到 .mlirbc（triton.compile 会在 bishengir 阶段报错，忽略）。"""
    dumpdir = tempfile.mkdtemp(prefix=f"dump_{tag}_", dir=os.getcwd())
    os.environ["TRITON_KERNEL_DUMP"] = "1"
    os.environ["TRITON_DUMP_DIR"] = dumpdir
    src = ASTSource(fn=fn,
                    signature=SIG,
                    constexprs={
                        "BLOCK_Q": bq,
                        "BN": bn,
                        "DIM": 128
                    })
    try:
        triton.compile(src, target=GPUTarget("npu", "Ascend910B3", 0))
    except Exception:  # noqa: BLE001
        pass  # 预期：本机缺 torch 头
    files = glob.glob(os.path.join(dumpdir, "**", "*.mlirbc"), recursive=True)
    return files[0] if files else None, dumpdir


def run_bishengir(mlirbc, tag):
    """喂给后端自带的 bishengir，返回 HIVM IR 文本。"""
    outdir = tempfile.mkdtemp(prefix=f"hivm_{tag}_", dir=os.getcwd())
    out = os.path.join(outdir, "k")
    r = subprocess.run([BISHENGIR, mlirbc] + BS_FLAGS + ["-o", out],
                       capture_output=True,
                       text=True,
                       timeout=1800)
    shutil.rmtree(outdir, ignore_errors=True)
    return r.stderr


def split_funcs(ir):
    """把 HIVM IR 按 _mix_aic / _mix_aiv 切开。"""
    parts = re.split(r"^func\.func @\w+_mix_(\w+)\(", ir, flags=re.M)
    out = {}
    for i in range(1, len(parts) - 1, 2):
        out[parts[i]] = parts[i + 1]
    return out


def loop_body(text):
    """取最外层 scf.for 的循环体（大括号配平）。只有体内的 sync 站点才随迭代重复。"""
    m = re.search(r"scf\.for[^\n]*\{\s*$", text, re.M)
    if not m:
        return ""
    i, depth = m.end(), 1
    while i < len(text) and depth:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[m.end():i - 1]


def ub_peak_kb(func_text):
    """UB 峰值 = pointer_cast 立即数偏移的最大值 + 该分配自身大小（近似取其元素字节数）。"""
    best = 0
    for m in re.finditer(
            r"hivm\.hir\.pointer_cast\(%c(\d+)_i64\) : memref<(\d+)x([a-z0-9]+), #hivm\.address_space<ub>>",
            func_text):
        off, n, dt = int(m.group(1)), int(m.group(2)), m.group(3)
        elem = {"f16": 2, "f32": 4, "i32": 4, "i64": 8, "i8": 1}.get(dt, 4)
        best = max(best, off + n * elem)
    if best == 0:  # 退路：只看最大偏移
        for m in re.finditer(r"pointer_cast\(%c(\d+)_i64\)[^\n]*address_space<ub>",
                             func_text):
            best = max(best, int(m.group(1)))
    return best / 1024


def main():
    cfgs = [
        ("k_seq  BN=64", k_seq, 64, 64, 64),  # 基线：每轮 64 列
        ("k_group4 BN=64", k_group4, 64, 64, 256),  # 方法一：每轮 256 列
        ("k_seq  BN=256", k_seq, 256, 64, 256),  # 方法二：每轮 256 列
        ("k_seq  BN=384", k_seq, 384, 64, 384),  # 探 UB 天花板
        ("k_seq  BN=512", k_seq, 512, 64, 512),  # 探 UB 天花板
    ]
    rows = []
    for label, fn, bn, bq, width in cfgs:
        tag = label.split()[0] + str(bn)
        mlirbc, dumpdir = emit_mlirbc(fn, bq, bn, tag)
        if not mlirbc:
            rows.append((label, width, None, f"没拿到 .mlirbc"))
            shutil.rmtree(dumpdir, ignore_errors=True)
            continue
        ir = run_bishengir(mlirbc, tag)
        shutil.rmtree(dumpdir, ignore_errors=True)
        funcs = split_funcs(ir)
        if not funcs:
            rows.append((
                label, width, None,
                f"bishengir 没产出 HIVM IR: {ir.strip().splitlines()[-1][:60] if ir.strip() else '空'}"
            ))
            continue
        stats = {}
        for core, body in funcs.items():
            for k, pat in COUNTERS.items():
                stats[f"{core}.{k}"] = len(re.findall(pat, body))
        aic = funcs.get("aic", "")
        stats["_body_code"] = len(re.findall(r"hivm\.hir\.sync_block", loop_body(aic)))
        stats["_ops"] = sorted(set(re.findall(r"hivm\.hir\.([a-z_0-9]+)", aic)))
        stats["_ub_aiv"] = ub_peak_kb(funcs.get("aiv", ""))
        stats["_ub_aic"] = ub_peak_kb(aic)
        rows.append((label, width, stats, None))

    keys = [f"aic.{k}" for k in COUNTERS] + ["aic.ub_KB", "aiv.ub_KB"]
    print("\n" + "=" * 100)
    print("HIVM IR 统计（后端自带 bishengir 2.0.0，本地生成，未上 NPU）")
    print("=" * 100)
    for label, width, stats, err in rows:
        print(f"\n── {label}   （每轮 {width} 列）")
        if err:
            print(f"   ✗ {err}")
            continue
        for k in COUNTERS:
            print(
                f"   aic.{k:<12} {stats.get(f'aic.{k}', 0):>4}     aiv.{k:<12} {stats.get(f'aiv.{k}', 0):>4}"
            )

    ok = [r for r in rows if r[2]]
    print("\n" + "=" * 100)
    print("归一化：只有【循环体内】的 sync 站点会随迭代重复（循环外的是 prologue/epilogue）")
    print("=" * 100)
    print(f"{'配置':<16}{'名义列宽':>9}{'体内sync':>10}{'每256列 sync':>14}{'相对基线':>10}")
    base = None
    for label, width, stats, _e in ok:
        per = stats["_body_code"] * 256.0 / width
        if base is None:
            base = per
        print(
            f"{label:<16}{width:>9}{stats['_body_code']:>10}{per:>14.1f}{base / per:>9.2f}x"
            f"    UB(AIV) {stats['_ub_aiv']:>6.1f} KB  UB(AIC) {stats['_ub_aic']:>6.1f} KB"
        )

    print("\n  AIC 算子清单（三档一致说明 lowering 路径相同，差别只在数量）:")
    print("   ", ", ".join(sorted(ok[0][2]["_ops"])) if ok else "—")

    print("\n判据：")
    print("  · 若 k_group4 与 k_seq_BN64 的【归一化 sync】相同 ⇒ 分组一个同步都省不掉")
    print("  · 若 k_seq_BN256 的【归一化 sync】明显更小 ⇒ 只有放大 BLOCK_N 有效")


if __name__ == "__main__":
    main()
