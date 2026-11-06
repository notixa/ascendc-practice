"""FlashDecoding 性能基准（split-KV decode）。

对比：

- 被测：``kernels.flash_decoding.flash_decoding.flash_decoding``（两段式 split-KV）
- 基线：
  - ``eager``：PyTorch ``softmax(QK^T * scale) @ V``（softmax 用 fp32）
  - ``fia``：``torch_npu.npu_fused_infer_attention_score``（CANN 官方
    FusedInferAttentionScore，支持 GQA/MQA；decode 场景内部的 FlashDecoding）
  - ``sdpa``：``torch.nn.functional.scaled_dot_product_attention(enable_gqa=True)``
  - ``none``：不测基线

计时用 ``torch.npu.Event``；没有 Event 时回退到 ``time.perf_counter`` + synchronize。
每个 case 先 warmup，再跑 ``trials`` 轮，每轮 ``iters`` 次，取中位数。

输入布局：

- ``Q:   [B, H_kv, G, D]``
- ``K/V: [B, H_kv, S, D]``

用法::

    python scripts/fd_bench.py --preset quick
    python scripts/fd_bench.py
    python scripts/fd_bench.py --preset long --baseline none
    python scripts/fd_bench.py --warmup 20 --iters 50 --trials 7

退出码：0 全部跑完且 check 通过；1 有 case 报错或 check 失败；2 环境不可用。
"""

from __future__ import annotations

import argparse
import inspect
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# (label, B, H_kv, G, S, D)
DEFAULT_CASES: list[tuple[str, int, int, int, int, int]] = [
    ("llama3-8b-g8", 1, 8, 8, 1024, 128),
    ("llama3-8b-g4", 1, 8, 4, 2048, 128),
    ("llama3.1-405b", 1, 8, 16, 4096, 128),
    ("mha-g1", 1, 8, 1, 1024, 128),
    ("mqa-g32", 1, 1, 32, 1024, 128),
    ("non-pow2-g5", 1, 2, 5, 1536, 128),
    ("batched-b4", 4, 8, 8, 1024, 128),
]

DTYPE_MAP = {
    "fp16": "float16",
    "bf16": "bfloat16",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FlashDecoding benchmark")
    parser.add_argument("--device", type=int, default=0, help="NPU device id（默认 0）")
    parser.add_argument(
        "--dtype",
        choices=["fp16", "bf16"],
        default="fp16",
        help="输入 dtype（默认 fp16；当前 kernel 对 bf16 支持不完整）",
    )
    parser.add_argument(
        "--preset",
        choices=["default", "quick", "long"],
        default="default",
        help="default=7 个代表 case；quick=前 2 个；long=追加 S=8192",
    )
    parser.add_argument(
        "--baseline",
        choices=["eager", "fia", "sdpa", "none"],
        default="eager",
        help="baseline：eager / fia（官方 FIA）/ sdpa / none",
    )
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
    parser.add_argument(
        "--reuse-workspace",
        action="store_true",
        help="复用 mid_o/mid_lse/out，排除每次调用的分配开销",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="用 torch.npu.NPUGraph 捕获 flash_decoding（自动复用 workspace），"
        "测 graph replay 延迟",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="不跑精度 check，只测时间",
    )
    return parser.parse_args(argv)


def build_cases(preset: str) -> list[tuple[str, int, int, int, int, int]]:
    cases = list(DEFAULT_CASES)
    if preset == "quick":
        cases = cases[:2]
    elif preset == "long":
        cases = cases + [
            ("long-g8-8k", 1, 8, 8, 8192, 128),
            ("long-g16-8k", 1, 8, 16, 8192, 128),
        ]
    return cases


def eager_attention(torch, q, k, v):
    """朴素 decode 基线：QK^T -> fp32 softmax -> PV。"""
    scale = q.size(-1) ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    p = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(p, v)


def fia_attention(torch, q, k, v):
    """CANN 官方 FusedInferAttentionScore（FIA）基线。

    q: [B, H_kv, G, D] -> BNSD [B, H_q, 1, D]
    k/v: [B, H_kv, S, D]
    """
    import torch_npu

    fn = getattr(torch_npu, "npu_fused_infer_attention_score", None)
    if fn is None:
        raise RuntimeError(
            "torch_npu.npu_fused_infer_attention_score 不可用；"
            "请确认 torch_npu/CANN 版本支持 FIA"
        )

    batch, heads_kv, group, dim = q.shape
    heads_q = heads_kv * group
    q_bnsd = q.reshape(batch, heads_q, 1, dim)
    seq_len = k.size(2)

    kwargs = {
        "num_heads": heads_q,
        "num_key_value_heads": heads_kv,
        "input_layout": "BNSD",
        "scale": dim**-0.5,
        # Q_S=1 时 actual_seq_lengths 不生效，官方 decode 示例只传 kv 长度。
        "actual_seq_lengths_kv": [seq_len] * batch,
    }
    try:
        params = inspect.signature(fn).parameters
        filtered = {name: value for name, value in kwargs.items() if name in params}
        if "num_heads" in filtered:
            kwargs = filtered
    except (TypeError, ValueError):
        pass

    result = fn(q_bnsd, k, v, **kwargs)
    out = result[0] if isinstance(result, (tuple, list)) else result
    return out.reshape(batch, heads_kv, group, dim)


def sdpa_attention(torch, q, k, v):
    """PyTorch SDPA GQA 基线（部分版本会下沉到 FIA）。"""
    batch, heads_kv, group, dim = q.shape
    q_bnsd = q.reshape(batch, heads_kv * group, 1, dim)
    out = torch.nn.functional.scaled_dot_product_attention(
        q_bnsd,
        k,
        v,
        dropout_p=0.0,
        is_causal=False,
        enable_gqa=True,
    )
    return out.reshape(batch, heads_kv, group, dim)


def check_output(torch, run_kernel, q, k, v, atol: float, rtol: float) -> tuple[bool, float]:
    out = run_kernel()
    ref = eager_attention(torch, q, k, v)
    torch.npu.synchronize()

    # NPU 的 aclnnIsClose 要求两边 dtype 一致；统一在 fp32 上比较。
    out_f = out.float()
    ref_f = ref.float()
    finite = bool(torch.isfinite(out_f).all().item())
    max_abs = float((out_f - ref_f).abs().max().item())
    ok = finite and torch.allclose(out_f, ref_f, atol=atol, rtol=rtol)

    del out, ref, out_f, ref_f
    return ok, max_abs


def bench_ms(torch, fn, warmup: int, iters: int, trials: int) -> tuple[float, float]:
    """返回 (median_ms, min_ms)。"""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()

    times: list[float] = []
    for _ in range(trials):
        event_cls = getattr(torch.npu, "Event", None)
        if event_cls is not None:
            try:
                start = event_cls(enable_timing=True)
                end = event_cls(enable_timing=True)
                start.record()
                for _ in range(iters):
                    fn()
                end.record()
                torch.npu.synchronize()
                times.append(float(start.elapsed_time(end)) / iters)
                continue
            except Exception:  # noqa: BLE001
                # 某些 torch_npu 版本 Event 不可用，回退到 wall clock
                pass

        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.npu.synchronize()
        times.append((time.perf_counter() - t0) * 1e3 / iters)

    times.sort()
    return times[len(times) // 2], times[0]


def _bytes_and_flops(torch, q, k, batch: int, heads_kv: int, group: int, seq: int, dim: int) -> tuple[int, int]:
    itemsize = q.element_size()
    kv_bytes = 2 * batch * heads_kv * seq * dim * itemsize
    qo_bytes = 2 * batch * heads_kv * group * dim * itemsize
    total_bytes = kv_bytes + qo_bytes
    # QK + PV 各 2 * M * N * K，合计 4 * B * H_kv * G * S * D
    flops = 4 * batch * heads_kv * group * seq * dim
    return total_bytes, flops


def run_case(
    torch,
    flash_decoding,
    args: argparse.Namespace,
    label: str,
    batch: int,
    heads_kv: int,
    group: int,
    seq: int,
    dim: int,
) -> tuple[bool, bool]:
    """返回 (ran_ok, check_ok)。"""
    dtype = getattr(torch, DTYPE_MAP[args.dtype])
    device = torch.device(f"npu:{args.device}")

    seed = args.seed + batch * 101 + heads_kv * 53 + group * 31 + seq * 17 + dim
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn((batch, heads_kv, group, dim), generator=gen).to(device=device, dtype=dtype).contiguous()
    k = torch.randn((batch, heads_kv, seq, dim), generator=gen).to(device=device, dtype=dtype).contiguous()
    v = torch.randn((batch, heads_kv, seq, dim), generator=gen).to(device=device, dtype=dtype).contiguous()

    workspace = None
    if args.reuse_workspace or args.graph:
        num_splits = (seq + args.split_s - 1) // args.split_s
        workspace = (
            torch.empty(
                (batch, heads_kv, num_splits, group, dim),
                dtype=torch.float16,
                device=device,
            ),
            torch.empty(
                (batch, heads_kv, num_splits, group),
                dtype=torch.float32,
                device=device,
            ),
            torch.empty(
                (batch, heads_kv, group, dim),
                dtype=torch.float16,
                device=device,
            ),
        )

    if args.graph:
        if workspace is None:
            raise RuntimeError("--graph 需要 workspace")

        # 捕获前先 warmup，确保 Triton 编译/device 属性查询都已完成；
        # 捕获期间不能再有 torch.empty 之类的分配。
        for _ in range(3):
            flash_decoding(
                q,
                k,
                v,
                split_s=args.split_s,
                block_s=args.block_s,
                workspace=workspace,
            )
        torch.npu.synchronize()

        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            flash_decoding(
                q,
                k,
                v,
                split_s=args.split_s,
                block_s=args.block_s,
                workspace=workspace,
            )
        torch.npu.synchronize()

        def run_kernel():
            graph.replay()
            return workspace[2]

    else:

        def run_kernel():
            return flash_decoding(
                q,
                k,
                v,
                split_s=args.split_s,
                block_s=args.block_s,
                workspace=workspace,
            )

    check_ok = True
    max_abs = float("nan")
    if not args.no_check:
        check_ok, max_abs = check_output(torch, run_kernel, q, k, v, args.atol, args.rtol)

    ms, ms_min = bench_ms(
        torch,
        run_kernel,
        args.warmup,
        args.iters,
        args.trials,
    )

    base_ms = float("nan")
    if args.baseline == "eager":
        base_fn = lambda: eager_attention(torch, q, k, v)
    elif args.baseline == "fia":
        base_fn = lambda: fia_attention(torch, q, k, v)
    elif args.baseline == "sdpa":
        base_fn = lambda: sdpa_attention(torch, q, k, v)
    else:
        base_fn = None

    if base_fn is not None:
        if args.graph:
            # 和 kernel 同口径：baseline 也捕获成 graph，避免 host 开销差异。
            for _ in range(3):
                base_fn()
            torch.npu.synchronize()

            base_graph = torch.npu.NPUGraph()
            with torch.npu.graph(base_graph):
                base_out = base_fn()
            torch.npu.synchronize()

            def base_run():
                base_graph.replay()
                return base_out

        else:
            base_run = base_fn

        base_ms, _ = bench_ms(
            torch,
            base_run,
            args.warmup,
            args.iters,
            args.trials,
        )

    total_bytes, flops = _bytes_and_flops(torch, q, k, batch, heads_kv, group, seq, dim)
    gbs = total_bytes / (ms * 1e-3) / 1e9
    tflops = flops / (ms * 1e-3) / 1e12
    speedup = base_ms / ms if base_ms == base_ms and base_ms > 0 else float("nan")

    print(
        f"{label:14s} B={batch} H_kv={heads_kv:2d} G={group:2d} S={seq:5d} D={dim:3d} "
        f"| {ms:9.4f} ms (min {ms_min:8.4f}) "
        f"| {gbs:7.1f} GB/s | {tflops:7.2f} TFLOPS "
        f"| base {base_ms:9.4f} ms | x{speedup:6.2f} "
        f"| max|dO|={max_abs:.3e} {'OK' if check_ok else 'CHECK_FAIL'}"
    )

    del q, k, v
    torch.npu.empty_cache()
    return True, check_ok


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
    from kernels.flash_decoding.flash_decoding import flash_decoding

    cases = build_cases(args.preset)
    print(
        f"device=npu:{args.device} dtype={args.dtype} preset={args.preset} "
        f"split_s={args.split_s} block_s={args.block_s} "
        f"baseline={args.baseline} "
        f"reuse_workspace={int(args.reuse_workspace)} graph={int(args.graph)} "
        f"warmup={args.warmup} iters={args.iters} trials={args.trials}"
    )
    if args.graph and args.baseline != "none":
        print("[NOTE] kernel 和 baseline 都走 graph replay，x 倍速同口径")
    print("-" * 150)

    errored = 0
    check_failed = 0
    for label, batch, heads_kv, group, seq, dim in cases:
        try:
            _, ok = run_case(
                torch,
                flash_decoding,
                args,
                label,
                batch,
                heads_kv,
                group,
                seq,
                dim,
            )
            if not ok:
                check_failed += 1
        except Exception as exc:  # noqa: BLE001
            errored += 1
            print(f"{label:14s} [ERROR] {type(exc).__name__}: {exc}")

    print("-" * 150)
    print(f"summary: cases={len(cases)} error={errored} check_fail={check_failed}")
    return 0 if (errored == 0 and check_failed == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
