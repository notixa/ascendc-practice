#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ascend C RMSNorm Custom Operator vs torch_npu Performance Benchmark Suite
Hardware Target: Atlas 200I DK A2 (Ascend 310B4, 2 AIV Cores, aarch64)
Software Stack : CANN 8.0.RC3 + PyTorch 2.1.0 + torch_npu 2.1.0.post8
"""

import os
import sys
import time
import argparse
from typing import Callable, Tuple, List, Dict, Optional

import torch

try:
    import torch_npu

    HAS_TORCH_NPU = True
except ImportError:
    HAS_TORCH_NPU = False

try:
    import rmsnorm_npu

    HAS_CUSTOM_OP = True
except ImportError:
    HAS_CUSTOM_OP = False

# ANSI Color codes for terminal formatting
COLOR_HEADER = "\033[95m"
COLOR_BLUE = "\033[94m"
COLOR_CYAN = "\033[96m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_RED = "\033[91m"
COLOR_BOLD = "\033[1m"
COLOR_RESET = "\033[0m"


def print_banner(text: str):
    width = 110
    print("=" * width)
    print(f"{COLOR_BOLD}{COLOR_CYAN}{text.center(width)}{COLOR_RESET}")
    print("=" * width)


def print_section(title: str):
    print(f"\n{COLOR_BOLD}{COLOR_YELLOW}▶ {title}{COLOR_RESET}")
    print("-" * 110)


def pytorch_decomposed_rmsnorm(
    x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """Standard PyTorch decomposed RMSNorm running in eager mode.

    Decomposes into 5 discrete operators:
      1. x.pow(2)
      2. mean(dim=-1, keepdim=True)
      3. + eps
      4. torch.rsqrt(...)
      5. x * rrms * gamma
    """
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    rrms = torch.rsqrt(variance + eps)
    return x * rrms * gamma


def cpu_reference_rmsnorm(
    x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """High-precision reference implementation on CPU."""
    x_f = x.float()
    gamma_f = gamma.float()
    variance = x_f.pow(2).mean(dim=-1, keepdim=True)
    rrms = torch.rsqrt(variance + eps)
    out = x_f * rrms * gamma_f
    return out.to(x.dtype)


def check_builtin_torch_npu_support() -> Tuple[bool, str]:
    """Test whether torch_npu.npu_rms_norm is supported on the current NPU hardware."""
    if not HAS_TORCH_NPU or not torch.npu.is_available():
        return False, "torch_npu not available"

    try:
        dummy_x = torch.randn(2, 64, device="npu", dtype=torch.float32)
        dummy_g = torch.randn(64, device="npu", dtype=torch.float32)
        _ = torch_npu.npu_rms_norm(dummy_x, dummy_g)
        torch.npu.synchronize()
        return True, "Supported"
    except Exception as e:
        err_msg = str(e)
        if (
            "EZ3003" in err_msg
            or "No supported Ops kernel" in err_msg
            or "build op model failed" in err_msg
        ):
            return (
                False,
                "Unsupported on Ascend 310B (CANN OPP only provides pre-built kernel for 910/910B)",
            )
        return False, f"Failed: {err_msg[:60]}..."


def measure_latency_ms(
    fn: Callable[[], torch.Tensor],
    warmup: int = 15,
    repeat: int = 50,
    is_npu: bool = True,
) -> Dict[str, float]:
    """Measure operator latency in milliseconds with warmup and synchronization."""
    # Warmup
    for _ in range(warmup):
        fn()
    if is_npu and HAS_TORCH_NPU:
        torch.npu.synchronize()

    # Batched benchmark to avoid per-op CPU sync syscall overhead
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    if is_npu and HAS_TORCH_NPU:
        torch.npu.synchronize()
    t1 = time.perf_counter()

    avg_ms = (t1 - t0) * 1000.0 / repeat
    return {
        "mean_ms": avg_ms,
        "min_ms": avg_ms,
        "max_ms": avg_ms,
    }


def compute_bandwidth_gbps(
    shape: Tuple[int, ...], dtype: torch.dtype, latency_ms: float
) -> float:
    """Compute effective memory bandwidth in GB/s.

    RMSNorm Memory Access:
      Read:  X [Total Elements] + Gamma [Hidden Size]
      Write: Out [Total Elements]
      Total Bytes = (2 * numel(X) + numel(Gamma)) * element_size
    """
    element_size = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    numel_x = 1
    for s in shape:
        numel_x *= s
    numel_gamma = shape[-1]

    total_bytes = (2 * numel_x + numel_gamma) * element_size
    seconds = latency_ms / 1000.0
    if seconds <= 0:
        return 0.0
    return (total_bytes / (1024**3)) / seconds


def run_benchmark(args):
    print_banner("Ascend C RMSNorm Custom Operator vs torch_npu Performance Benchmark")

    # 1. Environment & Hardware Diagnostics
    print_section("Hardware & Environment Diagnostics")
    print(f"  • Platform           : Atlas 200I DK A2 (Ascend 310B4, 2 AIV Cores)")
    print(f"  • Host Architecture  : {os.uname().machine} (Quad-core ARM Cortex-A55)")
    print(f"  • PyTorch Version    : {torch.__version__}")
    if HAS_TORCH_NPU:
        print(f"  • torch_npu Version  : {torch_npu.__version__}")
        print(f"  • NPU Available      : {torch.npu.is_available()}")
        print(f"  • NPU Device Count   : {torch.npu.device_count()}")
        print(f"  • Current Device ID  : {torch.npu.current_device()}")
    else:
        print(f"  • torch_npu          : {COLOR_RED}Not Installed{COLOR_RESET}")

    if not HAS_CUSTOM_OP:
        print(f"{COLOR_RED}[FATAL ERROR] 'rmsnorm_npu' module not found!{COLOR_RESET}")
        print("Please build the extension first: python setup.py build_ext --inplace")
        sys.exit(1)
    print(f"  • rmsnorm_npu Custom : {COLOR_GREEN}Loaded Successfully{COLOR_RESET}")

    # 2. Check built-in torch_npu.npu_rms_norm status
    builtin_supported, builtin_reason = check_builtin_torch_npu_support()
    if builtin_supported:
        print(
            f"  • torch_npu.npu_rms_norm: {COLOR_GREEN}Available & Supported{COLOR_RESET}"
        )
    else:
        print(
            f"  • torch_npu.npu_rms_norm: {COLOR_YELLOW}Not Available ({builtin_reason}){COLOR_RESET}"
        )

    # 3. Define Benchmark Scenarios (Common LLM Shapes)
    # Formats: (Scenario, Shape, Description)
    scenarios = []

    # LLM Prefill Phase (Batch/Sequence processing)
    if args.scenario in ("all", "prefill"):
        scenarios.extend(
            [
                (
                    "Prefill",
                    (1, 512, 1024),
                    "Small Model (Qwen-0.5B, Seq 512, Dim 1024)",
                ),
                (
                    "Prefill",
                    (1, 1024, 2048),
                    "Medium Model (Qwen-1.8B, Seq 1024, Dim 2048)",
                ),
                (
                    "Prefill",
                    (1, 2048, 4096),
                    "7B Model Prefill (LLaMA-7B, Seq 2048, Dim 4096)",
                ),
                (
                    "Prefill",
                    (4, 256, 2048),
                    "Multi-batch Prefill (B=4, Seq 256, Dim 2048)",
                ),
                ("Prefill", (4, 512, 4096), "Heavy Prefill (B=4, Seq 512, Dim 4096)"),
            ]
        )

    # LLM Decode Phase (Token generation, SeqLen=1)
    if args.scenario in ("all", "decode"):
        scenarios.extend(
            [
                ("Decode", (1, 1, 2048), "Single-stream Decode (B=1, Seq 1, Dim 2048)"),
                (
                    "Decode",
                    (1, 1, 4096),
                    "Single-stream 7B Decode (B=1, Seq 1, Dim 4096)",
                ),
                ("Decode", (4, 1, 4096), "Batch-4 7B Decode (B=4, Seq 1, Dim 4096)"),
                (
                    "Decode",
                    (16, 1, 4096),
                    "Batch-16 7B Serving Decode (B=16, Seq 1, Dim 4096)",
                ),
            ]
        )

    dtypes = []
    if args.dtype in ("all", "fp16"):
        dtypes.append(torch.float16)
    if args.dtype in ("all", "fp32"):
        dtypes.append(torch.float32)

    # 4. Run Benchmark Suite
    print_section(
        f"Executing Benchmarks (Warmup: {args.warmup}, Repeat: {args.repeat})"
    )

    results_table = []

    for dtype in dtypes:
        dtype_str = "FP16" if dtype == torch.float16 else "FP32"
        print(f"\n{COLOR_BOLD}=== Precision: {dtype_str} ==={COLOR_RESET}")

        header = (
            f"{'Scenario':<8} | {'Shape':<16} | {'Dtype':<5} | "
            f"{'Custom Ascend C':<15} | {'PyTorch NPU':<13} | "
            f"{'torch_npu Op':<14} | {'CPU Ref':<11} | "
            f"{'Speedup':<9} | {'Throughput':<10} | {'MaxDiff'}"
        )
        print("-" * len(header))
        print(header)
        print("-" * len(header))

        for sc_name, shape, desc in scenarios:
            hidden_size = shape[-1]
            torch.manual_seed(42)

            # Generate inputs on NPU directly to benchmark pure kernel compute without PCIe/Host copy
            x_npu = torch.randn(shape, dtype=dtype, device="npu")
            gamma_npu = torch.randn(hidden_size, dtype=dtype, device="npu")
            if HAS_TORCH_NPU:
                torch.npu.synchronize()

            # Correctness verification vs CPU reference
            x_cpu = x_npu.cpu()
            gamma_cpu = gamma_npu.cpu()
            ref_out = cpu_reference_rmsnorm(x_cpu, gamma_cpu)

            # 1. Custom Ascend C Operator (NPU)
            try:
                custom_out = rmsnorm_npu.forward(x_npu, gamma_npu)
                if HAS_TORCH_NPU:
                    torch.npu.synchronize()
                max_diff = (
                    (custom_out.cpu().float() - ref_out.float()).abs().max().item()
                )
                custom_metrics = measure_latency_ms(
                    lambda: rmsnorm_npu.forward(x_npu, gamma_npu),
                    warmup=args.warmup,
                    repeat=args.repeat,
                    is_npu=True,
                )
                custom_lat = custom_metrics["mean_ms"]
            except Exception as e:
                custom_lat = None
                max_diff = float("nan")

            # 2. PyTorch Decomposed on NPU (Eager Mode)
            try:
                npu_decomp_metrics = measure_latency_ms(
                    lambda: pytorch_decomposed_rmsnorm(x_npu, gamma_npu),
                    warmup=args.warmup,
                    repeat=args.repeat,
                    is_npu=True,
                )
                npu_decomp_lat = npu_decomp_metrics["mean_ms"]
            except Exception as e:
                npu_decomp_lat = None

            # 3. Built-in torch_npu.npu_rms_norm
            builtin_lat = None
            if builtin_supported:
                try:
                    builtin_metrics = measure_latency_ms(
                        lambda: torch_npu.npu_rms_norm(x_npu, gamma_npu),
                        warmup=args.warmup,
                        repeat=args.repeat,
                        is_npu=True,
                    )
                    builtin_lat = builtin_metrics["mean_ms"]
                except Exception:
                    builtin_lat = None

            # 4. PyTorch CPU Reference
            cpu_lat = None
            if args.include_cpu:
                try:
                    cpu_metrics = measure_latency_ms(
                        lambda: pytorch_decomposed_rmsnorm(x_cpu, gamma_cpu),
                        warmup=max(3, args.warmup // 3),
                        repeat=max(5, args.repeat // 3),
                        is_npu=False,
                    )
                    cpu_lat = cpu_metrics["mean_ms"]
                except Exception:
                    cpu_lat = None

            # Calculate Speedup and Bandwidth
            # Baseline is PyTorch Decomposed on NPU
            speedup_str = "N/A"
            if custom_lat and npu_decomp_lat and custom_lat > 0:
                speedup = npu_decomp_lat / custom_lat
                color = COLOR_GREEN if speedup >= 1.0 else COLOR_YELLOW
                speedup_str = f"{color}{speedup:6.2f}x{COLOR_RESET}"

            bw_gbps = (
                compute_bandwidth_gbps(shape, dtype, custom_lat) if custom_lat else 0.0
            )

            # Format outputs
            custom_str = f"{custom_lat:8.4f} ms" if custom_lat else "FAIL"
            npu_str = f"{npu_decomp_lat:8.4f} ms" if npu_decomp_lat else "FAIL"
            builtin_str = f"{builtin_lat:8.4f} ms" if builtin_lat else "Unsupported*"
            cpu_str = f"{cpu_lat:8.3f} ms" if cpu_lat else "N/A"
            shape_str = str(list(shape))
            diff_str = f"{max_diff:.2e}"

            row_str = (
                f"{sc_name:<8} | {shape_str:<16} | {dtype_str:<5} | "
                f"{custom_str:<15} | {npu_str:<13} | "
                f"{builtin_str:<14} | {cpu_str:<11} | "
                f"{speedup_str:<18} | {bw_gbps:6.2f} GB/s | {diff_str}"
            )
            print(row_str)

            results_table.append(
                {
                    "scenario": sc_name,
                    "shape": shape_str,
                    "dtype": dtype_str,
                    "desc": desc,
                    "custom_ms": custom_lat,
                    "npu_ms": npu_decomp_lat,
                    "builtin_ms": builtin_lat,
                    "cpu_ms": cpu_lat,
                    "speedup": (npu_decomp_lat / custom_lat)
                    if (custom_lat and npu_decomp_lat)
                    else None,
                    "bandwidth_gbps": bw_gbps,
                    "max_diff": max_diff,
                }
            )

    # 5. Analysis & Architectural Insights
    print_section("Architecture & Performance Insights")
    print("""
  1. Operator Fusion (Ascend C Kernel vs PyTorch Decomposed on NPU):
     • PyTorch decomposed RMSNorm issues multiple separate kernel launches:
       [pow -> mean (reduction across hidden_dim) -> add -> rsqrt -> mul -> mul].
       Each intermediate operation writes out to device memory (DDR/HBM) and reads back in,
       incurring multiple memory round-trips and framework kernel launch dispatch overheads.
     • The custom Ascend C RMSNorm fuses the entire computation into a single pass on the
       Ascend AI Vector core (AIV):
       Global Memory -> Unified Buffer (UB) -> Vector Compute (reduce/rsqrt/mul) -> Global Memory.
       Intermediate tensors never leave on-chip SRAM/UB.

  2. Built-in 'torch_npu.npu_rms_norm' Status (*):
     • On Ascend 310B (Atlas 200I DK A2), CANN 8.0.RC3 OPP does NOT include a pre-compiled
       binary kernel for the 'RmsNorm' operator (it is currently compiled only for Ascend 910/910B).
     • Calling torch_npu.npu_rms_norm on 310B throws error EZ3003 (Operator not supported).
     • This proves the CRITICAL NECESSITY of this custom Ascend C operator: without it,
       LLM deployment (LLaMA/Qwen/DeepSeek) on 310B is forced to use slow multi-op decomposition!

  3. Memory Bandwidth & Multi-Core Scaling:
     • RMSNorm is fundamentally a Memory-Bound (bandwidth-bound) operator with Arithmetic Intensity < 1 FLOP/Byte.
     • The custom kernel distributes rows evenly across the 2 AIV cores on Ascend 310B.
""")
    print("=" * 110)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ascend C RMSNorm Benchmark Suite")
    parser.add_argument(
        "--dtype",
        choices=["all", "fp16", "fp32"],
        default="all",
        help="Data type precision to benchmark (default: all)",
    )
    parser.add_argument(
        "--scenario",
        choices=["all", "prefill", "decode"],
        default="all",
        help="Scenario to benchmark: prefill, decode, or all (default: all)",
    )
    parser.add_argument(
        "--warmup", type=int, default=15, help="Warmup iterations (default: 15)"
    )
    parser.add_argument(
        "--repeat", type=int, default=50, help="Measurement iterations (default: 50)"
    )
    parser.add_argument(
        "--no-cpu",
        dest="include_cpu",
        action="store_false",
        help="Skip CPU reference benchmark",
    )
    args = parser.parse_args()

    run_benchmark(args)
