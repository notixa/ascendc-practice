"""FlashDecoding 精度门禁（split-KV decode）。

对比 ``kernels.flash_decoding.flash_decoding.flash_decoding`` 与 fp32 参考实现：

    scores = Q @ K^T * D^-0.5
    O      = softmax(scores) @ V

输入布局（和 kernel 的契约一致）：

- ``Q:   [B, H_kv, G, D]``  G = 每个 KV head 对应的 Q head 数
- ``K/V: [B, H_kv, S, D]``

覆盖点：

- ``G = 1``（MHA）、``G = 4/8``（常见 GQA）、``G = 5``（非 2 的幂）、``G = 16``；
- ``S = 128/1000/1536``，其中 1536 会走 ``NUM_SPLITS=3 -> PAD=4`` 的 split 路径；
- ``--long`` 追加 4096；
- 默认 fp16（当前 kernel 只支持 fp16；bf16 会显式报错）。

用法（在 910B 机器上）::

    python scripts/fd_precision_gate.py --quick
    python scripts/fd_precision_gate.py
    python scripts/fd_precision_gate.py --long
    python scripts/fd_precision_gate.py --group 5 8 --seq 1000 1536

退出码：0 全过；1 有 case 超差/异常；2 环境不可用。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

DTYPE_MAP = {
    "fp16": "float16",
    "bf16": "bfloat16",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FlashDecoding precision gate")
    parser.add_argument("--device", type=int, default=0, help="NPU device id（默认 0）")
    parser.add_argument(
        "--dtype",
        choices=["fp16", "bf16"],
        default="fp16",
        help="输入 dtype（默认 fp16；当前 kernel 对 bf16 支持不完整）",
    )
    parser.add_argument(
        "--group",
        type=int,
        nargs="+",
        default=None,
        help="G 列表；默认 1 4 5 8 16（--quick 时只有 8）",
    )
    parser.add_argument(
        "--seq",
        type=int,
        nargs="+",
        default=None,
        help="KV 长度列表；默认 128 1000 1536（--quick 时只有 1024）",
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        nargs="+",
        default=[128],
        help="head_dim 列表（默认只测 128）",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads-kv", type=int, default=2)
    parser.add_argument(
        "--split-s",
        type=int,
        default=512,
        help="每个 split 的 KV 长度（默认 512；必须 >= block_s）",
    )
    parser.add_argument(
        "--block-s",
        type=int,
        default=128,
        help="内层 KV tile（默认 128；必须是 2 的幂）",
    )
    parser.add_argument("--quick", action="store_true", help="只跑 G=8、S=1024 的最小 case")
    parser.add_argument("--long", action="store_true", help="追加 S=4096")
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def build_cases(args: argparse.Namespace) -> list[tuple[int, int, int, int, int]]:
    """返回 (B, H_kv, G, S, D) 列表。"""
    if args.group is not None:
        groups = list(args.group)
    elif args.quick:
        groups = [8]
    else:
        groups = [1, 4, 5, 8, 16]

    if args.seq is not None:
        seqs = list(args.seq)
    elif args.quick:
        seqs = [1024]
    else:
        seqs = [128, 1000, 1536]

    if args.long and 4096 not in seqs:
        seqs = [*seqs, 4096]

    cases: list[tuple[int, int, int, int, int]] = []
    for seq in seqs:
        for group in groups:
            for dim in args.head_dim:
                cases.append((args.batch, args.heads_kv, group, seq, dim))
    return cases


def reference_flash_decoding(torch, q, k, v):
    """fp32 参考实现，返回 [B, H_kv, G, D]。"""
    qf = q.float()
    kf = k.float()
    vf = v.float()
    scale = q.size(-1) ** -0.5

    scores = torch.einsum("bhgd,bhsd->bhgs", qf, kf) * scale
    p = torch.softmax(scores, dim=-1)
    return torch.einsum("bhgs,bhsd->bhgd", p, vf)


def _max_rel(diff, ref, eps: float = 1e-6) -> float:
    rel = diff / ref.abs().clamp_min(eps)
    return float(rel.max().item())


def run_case(
    torch,
    flash_decoding,
    args: argparse.Namespace,
    batch: int,
    heads_kv: int,
    group: int,
    seq: int,
    dim: int,
) -> bool:
    dtype = getattr(torch, DTYPE_MAP[args.dtype])
    device = torch.device(f"npu:{args.device}")

    shape_q = (batch, heads_kv, group, dim)
    shape_kv = (batch, heads_kv, seq, dim)

    seed = args.seed + batch * 101 + heads_kv * 53 + group * 31 + seq * 17 + dim
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(shape_q, generator=gen).to(device=device, dtype=dtype).contiguous()
    k = torch.randn(shape_kv, generator=gen).to(device=device, dtype=dtype).contiguous()
    v = torch.randn(shape_kv, generator=gen).to(device=device, dtype=dtype).contiguous()

    out = flash_decoding(
        q,
        k,
        v,
        split_s=args.split_s,
        block_s=args.block_s,
    )
    torch.npu.synchronize()

    ref = reference_flash_decoding(torch, q, k, v)
    torch.npu.synchronize()

    if tuple(out.shape) != tuple(ref.shape):
        raise RuntimeError(f"输出形状不一致: kernel={tuple(out.shape)} ref={tuple(ref.shape)}")

    out_f = out.float()
    finite = bool(torch.isfinite(out_f).all().item())
    diff = (out_f - ref).abs()
    max_abs = float(diff.max().item())
    max_rel = _max_rel(diff, ref)
    ok = finite and torch.allclose(out_f, ref, atol=args.atol, rtol=args.rtol)

    status = "PASS" if ok else "FAIL"
    print(
        f"[{status}] dtype={args.dtype:4s} B={batch} H_kv={heads_kv:2d} "
        f"G={group:2d} S={seq:4d} D={dim:3d} "
        f"max|dO|={max_abs:.3e} maxRel={max_rel:.3e}"
    )
    if not finite:
        print("       !! O 含 NaN/Inf")
    if not ok:
        print(f"       !! O 超差（atol={args.atol}, rtol={args.rtol}）")

    del q, k, v, out, ref, out_f, diff
    torch.npu.empty_cache()
    return ok


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        import torch
        import torch_npu  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"[SKIP] 无法导入 torch/torch_npu：{type(exc).__name__}: {exc}")
        return 2

    if not torch.npu.is_available():
        print("[SKIP] 当前机器没有可用 NPU")
        return 2

    if args.dtype == "bf16":
        print("[WARN] 当前 flash_decoding kernel 对 bf16 支持不完整，失败属预期")

    torch.npu.set_device(args.device)
    if args.device != 0:
        print("[WARN] flash_decoding 内部固定读取 device 0 的核数，--device != 0 时 grid 可能不准")

    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(SCRIPT_DIR))

    from kernels.flash_decoding.flash_decoding import flash_decoding

    cases = build_cases(args)
    total = len(cases)
    passed = 0
    failed = 0
    errored = 0

    print(
        f"device=npu:{args.device} cases={total} "
        f"atol={args.atol} rtol={args.rtol} dtype={args.dtype} "
        f"split_s={args.split_s} block_s={args.block_s}"
    )
    print("-" * 110)

    for idx, (batch, heads_kv, group, seq, dim) in enumerate(cases, 1):
        print(f"({idx}/{total}) running ...", end=" ", flush=True)
        try:
            ok = run_case(torch, flash_decoding, args, batch, heads_kv, group, seq, dim)
        except Exception as exc:  # noqa: BLE001
            errored += 1
            print(f"[ERROR] {type(exc).__name__}: {exc}")
            continue
        if ok:
            passed += 1
        else:
            failed += 1

    print("-" * 110)
    print(f"summary: total={total} pass={passed} fail={failed} error={errored}")
    return 0 if (failed == 0 and errored == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
