#!/usr/bin/env python3
"""本地对比 `fa_fwd.py` 和 `fa_fwd_manual.py` 的 IR，定位"显式 GM workspace"为什么错。

做法
----
1. 把两个文件里的 `@triton.jit def fla_fwd_kernel(...)` **原文抽出来**，
   写到只含 triton import 的临时模块里（不 import torch / torch_npu）。
2. 用 `ASTSource` + `triton.compile` 编到 `.mlirbc`（bishengir 阶段会报错，忽略）。
3. 用 bishengir 跑出 HIVM IR（text）。
4. 统计并对比：`sync_block_set/wait`、`mmad`、`fixpipe`、GM load/store 数量，
   以及"store 与紧随其后的 load 之间有没有 sync"。

用法
----
    python3 bench/ir_diff.py                 # 默认 32x256, NVEC=1
    python3 bench/ir_diff.py --bq 64 --bk 256
"""

from __future__ import annotations

import argparse
import ast
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

KERNEL_RE = re.compile(
    r"^@triton\.jit\ndef fla_fwd_kernel\(.*?(?=^@triton\.jit|^def |\Z)", re.S | re.M)

HDR = """\
import os
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
import triton.extension.buffer.language as bl
"""


def extract_kernel(path: str) -> str:
    src = open(path, encoding="utf-8").read()
    m = KERNEL_RE.search(src)
    if not m:
        raise SystemExit(f"在 {path} 里找不到 @triton.jit def fla_fwd_kernel")
    return m.group(0)


def build_signature(kernel_src: str, constexpr_vals: dict) -> tuple[dict, dict]:
    """从 AST 推出每个形参的 Triton 类型。"""
    tree = ast.parse(kernel_src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    fp16_ptrs = {"q_ptr", "k_ptr", "v_ptr", "o_ptr", "s_ptr", "p_ptr"}
    fp32_ptrs = {"otmp_ptr"}
    fp32_scalars = {"scale"}
    sig, cexpr = {}, {}
    for name in [a.arg for a in fn.args.args]:
        if name in fp16_ptrs:
            sig[name] = "*fp16"
        elif name in fp32_ptrs:
            sig[name] = "*fp32"
        elif name in fp32_scalars:
            sig[name] = "fp32"
        elif name in constexpr_vals:
            sig[name] = "constexpr"
            cexpr[name] = constexpr_vals[name]
        else:
            sig[name] = "i32"
    return sig, cexpr


def emit_mlirbc(kernel_src: str, constexpr_vals: dict, workdir: str, tag: str):
    """把 kernel 落到临时模块里编译，拿到 .mlirbc。"""
    mod_dir = os.path.join(workdir, tag)
    os.makedirs(mod_dir, exist_ok=True)
    mod_path = os.path.join(mod_dir, f"k_{tag}.py")
    with open(mod_path, "w", encoding="utf-8") as f:
        f.write(HDR + "\n\n" + kernel_src)
    sys.path.insert(0, mod_dir)
    for m in list(sys.modules):
        if m.startswith("k_"):
            del sys.modules[m]
    mod = __import__(f"k_{tag}")
    fn = mod.fla_fwd_kernel

    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    sig, cexpr = build_signature(kernel_src, constexpr_vals)
    dump_dir = os.path.join(workdir, f"dump_{tag}")
    os.environ["TRITON_KERNEL_DUMP"] = "1"
    os.environ["TRITON_DUMP_DIR"] = dump_dir
    os.environ["TRITON_CACHE_DIR"] = os.path.join(workdir, "cache")
    try:
        triton.compile(ASTSource(fn=fn, signature=sig, constexprs=cexpr),
                       target=GPUTarget("npu", "Ascend910B3", 0))
    except Exception:
        pass  # bishengir 阶段必然报错（本地没有 NPU 侧链接环境），只要 mlirbc 落盘即可
    hits = glob.glob(os.path.join(dump_dir, "**", "*.mlirbc"), recursive=True)
    if not hits:
        raise SystemExit(f"{tag}: 没拿到 .mlirbc（dump 目录 {dump_dir}）")
    return hits[0], mod_path


def run_bishengir(mlirbc: str, workdir: str, tag: str) -> str:
    import triton
    triton_dir = os.path.dirname(triton.__file__)
    bc = os.path.join(triton_dir, "backends/ascend/bishengir/bin/bishengir-compile")
    outdir = os.path.join(workdir, f"ir_{tag}")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "k")
    cmd = [
        bc, mlirbc, "--target=Ascend910B3", "--enable-hfusion-compile=true",
        "--enable-hivm-compile=true", "--enable-triton-kernel-compile=true",
        "--enable-auto-multi-buffer=True", "--enable-auto-bind-sub-block=True", "-o",
        out
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    shutil.rmtree(outdir, ignore_errors=True)
    return r.stderr


def summarize(ir: str) -> dict:

    def cnt(pat):
        return len(re.findall(pat, ir))

    return {
        "IR 行数": len(ir.splitlines()),
        "func.func": cnt(r"func\.func"),
        "sync_block_set": cnt(r"hivm\.hir\.sync_block_set"),
        "sync_block_wait": cnt(r"hivm\.hir\.sync_block_wait"),
        "sync_block_all": cnt(r"hivm\.hir\.sync_block_all"),
        "mmad": cnt(r"hivm\.hir\.mmad"),
        "fixpipe": cnt(r"hivm\.hir\.fixpipe"),
        "nd2nz": cnt(r"hivm\.hir\.nd2nz"),
        "vsel(where)": cnt(r"hivm\.hir\.vsel"),
        "vexp": cnt(r"hivm\.hir\.vexp"),
        "copy/mov": cnt(r"hivm\.hir\.(?:copy|mov_ub_to_ub|load|store)\b"),
        "multi_buffer": cnt(r"multi_buffer_counter"),
    }


def show_sync_context(ir: str, max_lines: int = 26, tag: str = ""):
    """把 sync/converter 相关的关键行按顺序列出来，看 store→load 之间有没有 sync。"""
    keys = ("sync_block_set", "sync_block_wait", "fixpipe", "hivm.hir.mmad", "nd2nz",
            "multi_buffer_counter")
    print(f"\n  ── {tag}: 关键指令序列（去重连续相同项）──")
    print("     （只显示 aic 或 aiv 中指令更多的那个 function）")
    # 取 sync 数最多的那个 function
    funcs = re.split(r"(?=func\.func)", ir)
    funcs = [f for f in funcs if "func.func" in f]
    if not funcs:
        print("      （没有 func.func）")
        return
    funcs.sort(key=lambda f: -len(re.findall(r"hivm\.hir\.", f)))
    body = funcs[0]
    seq = []
    for line in body.splitlines():
        for k in keys:
            if k in line:
                label = k
                m = re.search(r"(?:cora_id|event_id)\s*=\s*(\d+)", line)
                if m:
                    label += f"(id={m.group(1)})"
                if not seq or seq[-1] != label:
                    seq.append(label)
                break
    print("      " + " -> ".join(seq[:max_lines]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bq", type=int, default=32)
    ap.add_argument("--bk", type=int, default=256)
    ap.add_argument("--nq", type=int, default=4096)
    ap.add_argument("--nk", type=int, default=4096)
    ap.add_argument("--nvec", type=int, default=1)
    a = ap.parse_args()

    files = {
        "fa_fwd":
        os.path.join(_ROOT, "triton_lab/kernels/flash_attention/fa_fwd.py"),
        "manual":
        os.path.join(_ROOT, "triton_lab/kernels/flash_attention/fa_fwd_manual.py"),
    }
    constexpr_vals = {
        "N_q": a.nq,
        "N_k": a.nk,
        "delta_qk": a.nk - a.nq,
        "BLOCK_Q": a.bq,
        "BLOCK_K": a.bk,
        "HEAD_DIM": 128,
        "IS_CAUSAL": True,
        "NVEC": a.nvec,
    }
    print(f"\n{'='*100}")
    print(
        f"配置: BLOCK_Q={a.bq} BLOCK_K={a.bk} N_q={a.nq} N_k={a.nk} causal=True NVEC={a.nvec}"
    )
    print(f"{'='*100}")

    workdir = tempfile.mkdtemp(prefix="irdiff_", dir=os.path.join(_ROOT, ".scratch"))
    results = {}
    for tag, path in files.items():
        print(f"\n[{tag}] 抽取并编译 ...", flush=True)
        src = extract_kernel(path)
        mlirbc, mod_path = emit_mlirbc(src, constexpr_vals, workdir, tag)
        ir = run_bishengir(mlirbc, workdir, tag)
        if not ir.strip():
            print(f"  ✗ {tag}: bishengir 无输出")
            continue
        results[tag] = summarize(ir)
        print(f"  ✓ {tag}: {len(ir.splitlines())} 行 IR", flush=True)
        show_sync_context(ir, tag=tag)

    if len(results) == 2:
        print(f"\n{'='*100}")
        print(f"{'指标':<20}{'fa_fwd.py (SSA)':>22}{'manual (显式 GM)':>22}{'差':>10}")
        print("-" * 74)
        for k in results["fa_fwd"]:
            x, y = results["fa_fwd"][k], results["manual"][k]
            print(f"{k:<20}{x:>22}{y:>22}{y - x:>+10}")
        print(f"{'='*100}")
    shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
