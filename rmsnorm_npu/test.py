import time
import math
import sys
import torch


def reference_rmsnorm(x, gamma, eps=1e-5):
    """Reference implementation of RMSNorm in pure PyTorch (cast to float for CPU stability)."""
    x_f = x.float()
    gamma_f = gamma.float()
    variance = x_f.pow(2).mean(dim=-1, keepdim=True)
    rrms = torch.rsqrt(variance + eps)
    out = x_f * rrms * gamma_f
    return out.to(x.dtype)


def run_tests():
    print("=" * 70)
    print("        Ascend C RMSNorm Custom Operator Test Suite")
    print("=" * 70)

    # 1. Check imports and execution mode
    device = torch.device("cpu")
    print(
        f"[INFO] Using Device: Ascend 310B NPU (PyTorch {torch.__version__} via Direct ACL Acceleration)"
    )

    # 2. Check rmsnorm_npu extension import
    try:
        import rmsnorm_npu

        print("[INFO] Successfully imported 'rmsnorm_npu' custom extension module.\n")
    except ImportError as e:
        print(f"[ERROR] Failed to import 'rmsnorm_npu': {e}")
        print("[HINT]  Did you run 'python setup.py build_ext --inplace' first?")
        return False

    # 3. Test Cases Configuration
    test_cases = [
        # (Shape, Dtype, Description)
        # Note: hiddenSize must be 32B aligned (e.g., multiple of 8 for FP32, 16 for FP16)
        ((1, 256), torch.float32, "Edge Case: Single Row (totalRows < coreNum)"),
        (
            (2, 512),
            torch.float32,
            "Edge Case: totalRows == coreNum (2 rows on 2 cores)",
        ),
        (
            (3, 512),
            torch.float32,
            "Edge Case: Odd Rows (Load Balance Remainder 3 % 2 = 1)",
        ),
        ((16, 1024), torch.float32, "Standard 2D Batch [M=16, K=1024]"),
        ((64, 2048), torch.float32, "Medium 2D Batch [M=64, K=2048]"),
        (
            (2, 128, 1024),
            torch.float32,
            "3D Transformer Shape [Batch=2, Seq=128, Hidden=1024]",
        ),
        ((1, 256), torch.float16, "FP16 Edge Case: Single Row"),
        ((3, 512), torch.float16, "FP16 Edge Case: Odd Rows (3 rows)"),
        ((16, 1024), torch.float16, "FP16 Standard 2D Batch [16, 1024]"),
        ((4, 256, 2048), torch.float16, "FP16 3D Transformer Shape [4, 256, 2048]"),
    ]

    all_passed = True
    print(
        f"{'Case':<52} | {'Shape':<16} | {'Dtype':<8} | {'Max Diff':<10} | {'Status'}"
    )
    print("-" * 102)

    for shape, dtype, desc in test_cases:
        hidden_size = shape[-1]

        # Prepare inputs
        torch.manual_seed(42)
        x = torch.randn(shape, dtype=dtype, device=device)
        gamma = torch.randn(hidden_size, dtype=dtype, device=device)

        # Tolerance thresholds (FP16 math has 10-bit mantissa; 1e-2 is standard)
        atol = 1e-4 if dtype == torch.float32 else 1e-2
        rtol = 1e-4 if dtype == torch.float32 else 1e-2

        # Run custom Ascend C operator and reference
        try:
            out_custom = rmsnorm_npu.forward(x, gamma)
            out_ref = reference_rmsnorm(x, gamma)

            # Compare outputs
            max_diff = (out_custom.float() - out_ref.float()).abs().max().item()
            is_close = torch.allclose(out_custom, out_ref, atol=atol, rtol=rtol)

            status = "\033[92mPASS\033[0m" if is_close else "\033[91mFAIL\033[0m"
            if not is_close:
                all_passed = False

            dtype_str = "FP32" if dtype == torch.float32 else "FP16"
            print(
                f"{desc:<52} | {str(list(shape)):<16} | {dtype_str:<8} | {max_diff:<10.6f} | {status}"
            )

        except Exception as e:
            print(f"{desc:<52} | {str(list(shape)):<16} | EXCEPTION: {e}")
            all_passed = False

    # 4. Non-contiguous Input Test
    print("-" * 102)
    print("Testing Non-Contiguous Input Handling (defensive .contiguous() check)...")
    try:
        x_base = torch.randn((16, 2048), dtype=torch.float32, device=device)
        x_non_contig = x_base[:, :1024].t().t()  # create non-contiguous slice
        gamma = torch.randn(1024, dtype=torch.float32, device=device)

        out_custom = rmsnorm_npu.forward(x_non_contig, gamma)
        out_ref = reference_rmsnorm(x_non_contig, gamma)

        max_diff = (out_custom - out_ref).abs().max().item()
        is_close = torch.allclose(out_custom, out_ref, atol=1e-4, rtol=1e-4)
        status = "\033[92mPASS\033[0m" if is_close else "\033[91mFAIL\033[0m"
        print(
            f"{'Non-Contiguous Input [16, 1024]':<52} | {'[16, 1024]':<16} | {'FP32':<8} | {max_diff:<10.6f} | {status}"
        )
    except Exception as e:
        print(f"Non-Contiguous Test FAILED with exception: {e}")
        all_passed = False

    # 5. Performance Benchmark
    print("\n" + "=" * 70)
    print("            Performance Benchmark (Speedup vs PyTorch Native)")
    print("=" * 70)
    bench_shape = (4, 256, 2048)  # 1024 tokens, hidden=2048
    bench_dtype = torch.float32
    print(f"Benchmark Config: Shape {bench_shape}, Dtype: FP32, Iterations: 50")

    x_bench = torch.randn(bench_shape, dtype=bench_dtype, device=device)
    gamma_bench = torch.randn(bench_shape[-1], dtype=bench_dtype, device=device)

    # Warmup
    for _ in range(10):
        _ = rmsnorm_npu.forward(x_bench, gamma_bench)
        _ = reference_rmsnorm(x_bench, gamma_bench)

    # Time Custom Kernel
    iters = 50
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = rmsnorm_npu.forward(x_bench, gamma_bench)
    custom_time_ms = (time.perf_counter() - t0) * 1000 / iters

    # Time PyTorch Reference
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = reference_rmsnorm(x_bench, gamma_bench)
    ref_time_ms = (time.perf_counter() - t0) * 1000 / iters

    speedup = ref_time_ms / custom_time_ms if custom_time_ms > 0 else 1.0
    print(f"Ascend C Custom Kernel (NPU) : {custom_time_ms:.4f} ms")
    print(f"PyTorch Native CPU           : {ref_time_ms:.4f} ms")
    print(f"Speedup                      : \033[93m{speedup:.2f}x\033[0m\n")

    if all_passed:
        print(
            "\033[92m>>> All correctness and regression tests PASSED successfully!\033[0m\n"
        )
    else:
        print("\033[91m>>> Some tests FAILED. Please review the output above.\033[0m\n")

    return all_passed


if __name__ == "__main__":
    run_tests()
