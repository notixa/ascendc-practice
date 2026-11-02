"""FA forward 精度闸门（Step 0）。

对比 ``fla_fwd`` 与 fp32 参考实现（显式 QK^T -> softmax -> PV），
同时校验输出 O 和 LSE L。参考实现按 (b, h) 循环，避免物化
[B, H, S_q, S_k] 的大张量。

参考语义与 kernel 对齐：
- 输入布局 BSHD，``delta = S_k - S_q``；
- causal 采用 bottom-right 对齐：``k_idx <= q_idx + delta``；
  因此 causal 只测 ``S_k >= S_q`` 的组合（保证每行至少有一个可见 key）。

用法（在 910B 机器上）::

    python scripts/fa_precision_gate.py --quick
    python scripts/fa_precision_gate.py
    python scripts/fa_precision_gate.py --long
    python scripts/fa_precision_gate.py --dtype fp16 --seq 128 1024 --causal both

退出码：0 全过；1 有 case 超差/异常；2 环境不可用。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DTYPE_MAP = {
    "fp16": "float16",
    "bf16": "bfloat16",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FlashAttention forward precision gate"
    )
    parser.add_argument("--device", type=int, default=0, help="NPU device id（默认 0）")
    parser.add_argument(
        "--dtype",
        choices=["fp16", "bf16", "both"],
        default="both",
        help="输入 dtype（默认两者都测）",
    )
    parser.add_argument(
        "--seq",
        type=int,
        nargs="+",
        default=None,
        help="S_q = S_k 的序列长度列表；默认 128 1000 1024（--quick 时只有 128）",
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        nargs="+",
        default=[64, 128],
        help="head_dim 列表（默认 64 128）",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument(
        "--causal",
        choices=["both", "0", "1"],
        default="both",
        help="causal 开关（默认两种都测）",
    )
    parser.add_argument("--quick", action="store_true", help="只跑 128x128 的最小矩阵")
    parser.add_argument("--long", action="store_true", help="追加 4096x4096")
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-lse", action="store_true", help="跳过 LSE 校验")
    return parser.parse_args(argv)


def build_cases(args: argparse.Namespace) -> list[tuple[int, int, int, bool, str]]:
    """返回 (S_q, S_k, head_dim, causal, dtype_name) 列表。"""
    if args.seq is not None:
        seqs = list(args.seq)
    elif args.quick:
        seqs = [128]
    else:
        seqs = [128, 1000, 1024]

    pairs: list[tuple[int, int]] = [(s, s) for s in seqs]
    # 非等长：非 causal 与 causal（只保留 S_k >= S_q）；quick 模式不跑
    if not args.quick:
        pairs.append((128, 256))
    if args.long:
        pairs.append((4096, 4096))

    causal_opts = [False, True] if args.causal == "both" else [args.causal == "1"]
    dtype_names = ["fp16", "bf16"] if args.dtype == "both" else [args.dtype]

    cases: list[tuple[int, int, int, bool, str]] = []
    for sq, sk in pairs:
        for hd in args.head_dim:
            for causal in causal_opts:
                if causal and sq > sk:
                    continue
                for name in dtype_names:
                    cases.append((sq, sk, hd, causal, name))
    return cases


def reference_attention(torch, q, k, v, causal: bool, scale: float):
    """fp32 参考实现，返回 (O [B,H,Sq,D], LSE [B,H,Sq])。"""
    b, sq, h, d = q.shape
    sk = k.shape[1]

    qf = q.permute(0, 2, 1, 3).contiguous().float()
    kf = k.permute(0, 2, 1, 3).contiguous().float()
    vf = v.permute(0, 2, 1, 3).contiguous().float()

    idx_q = torch.arange(sq, device=q.device).view(sq, 1)
    idx_k = torch.arange(sk, device=q.device).view(1, sk)
    keep = idx_k <= idx_q + (sk - sq)  # bottom-right causal

    o = torch.empty((b, h, sq, d), dtype=torch.float32, device=q.device)
    lse = torch.empty((b, h, sq), dtype=torch.float32, device=q.device)

    for bi in range(b):
        for hi in range(h):
            s = torch.matmul(qf[bi, hi], kf[bi, hi].transpose(0, 1)) * scale
            if causal:
                s = s.masked_fill(~keep, float("-inf"))
            lse[bi, hi] = torch.logsumexp(s, dim=-1)
            p = torch.softmax(s, dim=-1)
            o[bi, hi] = torch.matmul(p, vf[bi, hi])
    return o, lse


def _max_rel(diff, ref, eps: float = 1e-6) -> float:
    rel = diff / ref.abs().clamp_min(eps)
    return float(rel.max().item())


def run_case(
    torch, fla_fwd, args, sq: int, sk: int, hd: int, causal: bool, dtype_name: str
) -> bool:
    dtype = getattr(torch, DTYPE_MAP[dtype_name])
    device = torch.device(f"npu:{args.device}")

    shape_q = (args.batch, sq, args.heads, hd)
    shape_k = (args.batch, sk, args.heads, hd)

    gen = torch.Generator(device="cpu").manual_seed(args.seed + sq * 131 + sk * 17 + hd)
    q = torch.randn(shape_q, generator=gen).to(device=device, dtype=dtype).contiguous()
    k = torch.randn(shape_k, generator=gen).to(device=device, dtype=dtype).contiguous()
    v = torch.randn(shape_k, generator=gen).to(device=device, dtype=dtype).contiguous()

    scale = float(hd) ** -0.5

    o_k, l_k = fla_fwd(q, k, v, causal=causal)
    torch.npu.synchronize()

    o_ref, lse_ref = reference_attention(torch, q, k, v, causal, scale)
    torch.npu.synchronize()

    o_kf = o_k.permute(0, 2, 1, 3).float()
    finite_o = bool(torch.isfinite(o_kf).all().item())
    diff = (o_kf - o_ref).abs()
    max_abs = float(diff.max().item())
    max_rel = _max_rel(diff, o_ref)
    ok_o = finite_o and torch.allclose(o_kf, o_ref, atol=args.atol, rtol=args.rtol)

    ok_l = True
    max_lse = float("nan")
    if not args.no_lse:
        l_kf = l_k.float()
        finite_l = bool(torch.isfinite(l_kf).all().item())
        ldiff = (l_kf - lse_ref).abs()
        max_lse = float(ldiff.max().item())
        ok_l = finite_l and torch.allclose(
            l_kf, lse_ref, atol=args.atol, rtol=args.rtol
        )

    status = "PASS" if (ok_o and ok_l) else "FAIL"
    lse_msg = "skip" if args.no_lse else f"{max_lse:.3e}"
    print(
        f"[{status}] dtype={dtype_name:4s} causal={int(causal)} "
        f"Sq={sq:4d} Sk={sk:4d} D={hd:3d} B={args.batch} H={args.heads} "
        f"max|dO|={max_abs:.3e} maxRel={max_rel:.3e} max|dLSE|={lse_msg}"
    )
    if not finite_o:
        print("       !! O 含 NaN/Inf")
    if not ok_o:
        print(f"       !! O 超差（atol={args.atol}, rtol={args.rtol}）")
    if not args.no_lse and not ok_l:
        print(f"       !! LSE 超差（atol={args.atol}, rtol={args.rtol}）")

    del q, k, v, o_k, l_k, o_ref, lse_ref, o_kf, diff
    torch.npu.empty_cache()
    return ok_o and ok_l


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

    torch.npu.set_device(args.device)
    if args.device != 0:
        print(
            "[WARN] fla_fwd 内部固定读取 device 0 的核数，--device != 0 时 grid 可能不准"
        )
    sys.path.insert(0, str(REPO_ROOT))
    from kernels.flash_attention.fa_fwd import fla_fwd

    cases = build_cases(args)
    total = len(cases)
    passed = 0
    failed = 0
    errored = 0

    print(
        f"device=npu:{args.device} cases={total} "
        f"atol={args.atol} rtol={args.rtol} check_lse={not args.no_lse}"
    )
    print("首次调用会触发 autotune（每个 key 8 组配置），耗时会明显高于后续 case")
    print("-" * 100)

    for idx, (sq, sk, hd, causal, dtype_name) in enumerate(cases, 1):
        print(f"({idx}/{total}) running ...", end=" ", flush=True)
        try:
            ok = run_case(torch, fla_fwd, args, sq, sk, hd, causal, dtype_name)
        except Exception as exc:  # noqa: BLE001
            errored += 1
            print(f"[ERROR] {type(exc).__name__}: {exc}")
            continue
        if ok:
            passed += 1
        else:
            failed += 1

    print("-" * 100)
    print(f"summary: total={total} pass={passed} fail={failed} error={errored}")
    return 0 if (failed == 0 and errored == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
