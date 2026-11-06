#!/usr/bin/env python3
"""本地 dump bishengir 全流水 IR（不需要 NPU，不需要服务器）。

前置条件（一次性）
------------------
只装了 triton-ascend 但没有 torch 的环境，需要做三件事：
  1. 造一个假的 torch_npu dist-info（编译路径会读它的版本号）
       DIR=<venv>/lib/python3.11/site-packages/torch_npu-2.7.1.post4.dist-info
       printf 'Metadata-Version: 2.1\\nName: torch_npu\\nVersion: 2.7.1.post4\\n' > $DIR/METADATA
  2. driver.py 的 NPUUtils.__init__ 支持 TRITON_SKIP_SO_BUILD 跳过 C 扩展构建
  3. driver.py 的 get_arch 支持 TRITON_ARCH_OVERRIDE（无设备时返回 Ascend910B3）
  4. compiler.py 两处 `cmd_list = (...)` 之后注入 FA_BH2（本仓库 .scratch/venv-ir 已打）

  另外需要一个 CPU 版 torch（`NPUUtils` 找 include 路径）：
       uv pip install --python <venv>/bin/python --no-deps "torch==2.7.1+cpu" \
           --index-url https://download.pytorch.org/whl/cpu

用法
----
    python3 bench/ir_dump.py --kernel fa --bq 64 --bk 512
    python3 bench/ir_dump.py --kernel manual --bq 64 --bk 256 --stats

输出
----
    .scratch/irdump-<kernel>-<bq>x<bk>/  下的数百个 stage IR 文件
    --stats 时打印关键统计（sub-block 分区、DMA、同步数）
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import re
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

KERNELS = {
    "fa": "triton_lab/kernels/flash_attention/fa_fwd.py",
    "manual": "triton_lab/kernels/flash_attention/fa_fwd_manual.py",
}
VEC_OPS = r"hivm\.hir\.(vadd|vmul|vmax|vexp|vsel|vcmp|vcast|vrec|vln|vabs|vdiv|vsub)"


def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument("--kernel", default="fa", choices=list(KERNELS))
    ap.add_argument("--bq", type=int, default=64)
    ap.add_argument("--bk", type=int, default=512)
    ap.add_argument("--nq", type=int, default=4096)
    ap.add_argument("--nk", type=int, default=4096)
    ap.add_argument("--nvec", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=0, help="核内 tiling 粒度；0 = 等于 BLOCK_K")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--keep", action="store_true", help="保留临时目录")
    return ap.parse_args()


def main():
    a = parse_args()
    import ir_diff  # 复用抽取 / 签名 / emit_mlirbc

    os.environ["TRITON_SKIP_SO_BUILD"] = "1"
    os.environ["TRITON_ARCH_OVERRIDE"] = "Ascend910B3"

    outdir = os.path.join(_ROOT, ".scratch", f"irdump-{a.kernel}-{a.bq}x{a.bk}")
    shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(outdir, exist_ok=True)
    os.environ[
        "FA_BH2"] = f"--mlir-print-ir-after-all --mlir-print-ir-tree-dir={outdir}"

    src = ir_diff.extract_kernel(os.path.join(_ROOT, KERNELS[a.kernel]))
    ce = {
        "N_q": a.nq,
        "N_k": a.nk,
        "delta_qk": a.nk - a.nq,
        "BLOCK_Q": a.bq,
        "BLOCK_K": a.bk,
        "HEAD_DIM": 128,
        "IS_CAUSAL": True,
        "NVEC": a.nvec,
        "CHUNK": a.chunk or a.bk,
        # 显式 GM workspace 的 stride（fa_fwd_manual.py 里标成了 tl.constexpr）
        # 必须填，否则 build_signature 会退化成 i32 运行时参数，
        # 编出来的 IR 就不是新版的（会重现 ×16 交叉布局）
        "stride_sc": a.bq * a.bk,
        "stride_sq": a.bk,
        "stride_sk": 1,
        "stride_pc": a.bq * a.bk,
        "stride_pq": a.bk,
        "stride_pk": 1,
        "stride_otmp_c": a.bq * 128,
        "stride_otmp_q": 128,
        "stride_otmp_d": 1
    }
    sig, cexpr = ir_diff.build_signature(src, ce)

    work = tempfile.mkdtemp(prefix="irdump_", dir=os.path.join(_ROOT, ".scratch"))
    mod = os.path.join(work, "k.py")
    with open(mod, "w") as f:
        f.write(ir_diff.HDR + "\n\n" + src)
    spec = importlib.util.spec_from_file_location("irdump_k", mod)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    os.environ["TRITON_CACHE_DIR"] = os.path.join(work, "cache")

    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    ok = True
    try:
        triton.compile(ASTSource(fn=m.fla_fwd_kernel, signature=sig, constexprs=cexpr),
                       target=GPUTarget("npu", "Ascend910B3", 0))
        print("编译成功")
    except Exception as e:
        ok = False
        print("编译失败（IR 通常已落盘）:", str(e)[:160].replace("\n", " "))

    files = sorted(glob.glob(os.path.join(outdir, "**", "*.mlir"), recursive=True))
    print(f"\n落盘 {len(files)} 个 stage IR -> {outdir}")
    if not a.keep:
        shutil.rmtree(work, ignore_errors=True)

    if not a.stats:
        return 0 if ok else 1

    # ── 关键统计 ──
    def pick(substr, prefer_last=False):
        hits = [f for f in files if substr in os.path.basename(f)]
        return hits[-1] if (prefer_last and hits) else (hits[0] if hits else None)

    print(f"\n{'='*96}")
    print(
        f"{'stage':<44}{'行数':>7}{'vec_op':>8}{'mmad':>6}{'sync_set':>10}{'sync_wait':>11}"
    )
    print("-" * 96)
    targets = [
        "partition-and-bind-sub-block", "hivm-bind-sub-block", "inject-block-sync",
        "hivm-plan-memory"
    ]
    for t in targets:
        f = pick(t)
        if not f:
            print(f"{t:<44}{'（无此 pass）':>30}")
            continue
        txt = open(f).read()
        print(f"{os.path.basename(f)[:44]:<44}{len(txt.splitlines()):>7}"
              f"{len(re.findall(VEC_OPS, txt)):>8}"
              f"{len(re.findall(r'hivm.hir.mmad', txt)):>6}"
              f"{len(re.findall(r'sync_block_set', txt)):>10}"
              f"{len(re.findall(r'sync_block_wait', txt)):>11}")

    part = pick("partition-and-bind-sub-block")
    if part:
        txt = open(part).read()
        print(f"\n① 子核切分：core_type 标记 {len(re.findall(r'core_type = <', txt))} 处"
              f"，sub_block_id {len(re.findall(r'sub_block_id', txt))} 处")
        print(
            f"   vector op {len(re.findall(VEC_OPS, txt))} 个 —— "
            f"{'✅ 已按子块分裂（每个 op 出现 2 份或带 sub_block_id）' if len(re.findall('sub_block_id', txt)) else '❌ 未分裂（每个 op 只有一份，无 sub_block_id）'}"
        )
        core = re.findall(r"core_type = <([A-Z]+)>", txt)
        from collections import Counter
        print(f"   core_type 分布: {dict(Counter(core))}")
        for m2 in re.finditer(r"hivm\.hir\.load[^\n]*core_type = <([A-Z]+)>", txt):
            sh = re.search(r"outs\(([^)]*)\)", m2.group(0))
            if sh:
                s = re.search(r"tensor<([0-9x]+x[fi][0-9]+)>", sh.group(1))
                if s:
                    print(f"   DMA: {s.group(1):<16} core={m2.group(1)}")
    print(f"{'='*96}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
