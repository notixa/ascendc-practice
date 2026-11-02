"""单 tile 隔离探针：绕过 autotune，固定 BLOCK_Q/BLOCK_K 跑一个 case。

用途：autotune 会在同一进程里依次试多个 tile，某个 tile 触发 aicore
异常后设备被污染，后续 case 全部报 507015，无法判断是哪个配置。
本脚本每次只跑一个 tile，配合 shell 循环，每个配置一个进程，崩溃互不影响。

示例：

    python scripts/fa_tile_probe.py --seq 1000 --causal 0 --dtype fp16 \
        --block-q 96 --block-k 128

    # 逐个 tile 隔离（某个崩了不影响后面的）
    for cfg in "96 128" "64 256" "128 256" "128 128" "64 128" "32 256"; do
      set -- $cfg
      echo "== BQ=$1 BK=$2"
      python scripts/fa_tile_probe.py --seq 1000 --causal 0 --dtype fp16 \
        --block-q "$1" --block-k "$2" || echo "CRASH/FAIL BQ=$1 BK=$2"
    done
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FA single-tile isolation probe")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seq", type=int, default=1000, help="S_q = S_k")
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument(
        "--causal", choices=["0", "1"], default="0", help="0=非 causal, 1=causal"
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-q", type=int, required=True)
    parser.add_argument("--block-k", type=int, required=True)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-check", action="store_true", help="只跑 kernel，不比对")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    causal = args.causal == "1"

    try:
        import torch
        import torch_npu  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"[SKIP] 无法导入 torch/torch_npu：{type(exc).__name__}: {exc}")
        return 2

    if not torch.npu.is_available():
        print("[SKIP] 当前机器没有可用 NPU")
        return 2

    sys.path.insert(0, str(SCRIPT_DIR))
    sys.path.insert(0, str(REPO_ROOT))

    import triton
    from fa_precision_gate import DTYPE_MAP, _max_rel, reference_attention
    from torch_npu.npu import get_device_properties
    from triton.compiler.errors import MLIRCompilationError

    from kernels.flash_attention.fa_fwd import fla_fwd_kernel

    torch.npu.set_device(args.device)

    raw_kernel = getattr(fla_fwd_kernel, "fn", fla_fwd_kernel)
    dtype = getattr(torch, DTYPE_MAP[args.dtype])
    device = torch.device(f"npu:{args.device}")

    sq = sk = args.seq
    hd = args.head_dim
    bq = args.block_q
    bk = args.block_k

    gen = torch.Generator(device="cpu").manual_seed(args.seed + sq * 131 + hd)
    q = (
        torch.randn((args.batch, sq, args.heads, hd), generator=gen)
        .to(device=device, dtype=dtype)
        .contiguous()
    )
    k = (
        torch.randn((args.batch, sk, args.heads, hd), generator=gen)
        .to(device=device, dtype=dtype)
        .contiguous()
    )
    v = (
        torch.randn((args.batch, sk, args.heads, hd), generator=gen)
        .to(device=device, dtype=dtype)
        .contiguous()
    )

    o = torch.empty_like(q)
    l = torch.empty((args.batch, args.heads, sq), dtype=torch.float32, device=device)
    scale = float(hd) ** -0.5

    cores = get_device_properties(0).cube_core_num
    grid = (min(args.batch * args.heads * triton.cdiv(sq, bq), cores),)

    print(
        f"BQ={bq} BK={bk} seq={sq} causal={int(causal)} dtype={args.dtype} grid={grid}",
        flush=True,
    )

    try:
        raw_kernel[grid](
            q,
            k,
            v,
            o,
            l,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            l.stride(0),
            l.stride(1),
            l.stride(2),
            args.batch,
            args.heads,
            scale,
            N_q=sq,
            N_k=sk,
            delta_qk=sk - sq,
            HEAD_DIM=hd,
            IS_CAUSAL=causal,
            BLOCK_Q=bq,
            BLOCK_K=bk,
        )
    except MLIRCompilationError as exc:
        first = str(exc).strip().splitlines()
        print("[COMPILE_FAIL] " + (first[0] if first else "MLIRCompilationError"))
        return 3
    torch.npu.synchronize()
    print("kernel done", flush=True)

    if args.no_check:
        return 0

    o_ref, lse_ref = reference_attention(torch, q, k, v, causal, scale)
    torch.npu.synchronize()

    o_kf = o.permute(0, 2, 1, 3).float()
    diff = (o_kf - o_ref).abs()
    max_abs = float(diff.max().item())
    max_rel = _max_rel(diff, o_ref)
    finite_o = bool(torch.isfinite(o_kf).all().item())
    ok_o = finite_o and torch.allclose(o_kf, o_ref, atol=args.atol, rtol=args.rtol)

    l_kf = l.float()
    ldiff = (l_kf - lse_ref).abs()
    max_lse = float(ldiff.max().item())
    finite_l = bool(torch.isfinite(l_kf).all().item())
    ok_l = finite_l and torch.allclose(l_kf, lse_ref, atol=args.atol, rtol=args.rtol)

    status = "PASS" if (ok_o and ok_l) else "FAIL"
    print(
        f"[{status}] max|dO|={max_abs:.3e} maxRel={max_rel:.3e} max|dLSE|={max_lse:.3e}"
    )
    return 0 if (ok_o and ok_l) else 1


if __name__ == "__main__":
    sys.exit(main())
