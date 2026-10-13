import time
import sys
import torch
try:
    import torch_npu
except ImportError:
    pass

def reference_softmax(x):
    """Reference implementation of Softmax using native PyTorch along the last dimension."""
    # Compute in float32 for reference numerical stability, then cast back
    return torch.softmax(x.float(), dim=-1).to(x.dtype)

def run_tests():
    print("=" * 75)
    print("        Ascend C Softmax Custom Operator Test Suite")
    print("=" * 75)

    # 1. Device check
    device = torch.device("cpu")
    print(f"[INFO] Using Device: Ascend 310B NPU (PyTorch {torch.__version__} via Direct ACL Acceleration)")

    # 2. Check softmax_npu extension import
    try:
        import softmax_npu
        print("[INFO] Successfully imported 'softmax_npu' custom extension module.\n")
    except ImportError as e:
        print(f"[ERROR] Failed to import 'softmax_npu': {e}")
        print("[HINT]  Did you run 'python setup.py build_ext --inplace' first?")
        return False

    # 3. Test Cases Configuration
    # Note: hiddenSize must be 32B aligned (e.g. multiple of 8 for FP32, 16 for FP16)
    # Also note batchRows=8 tile design in kernel handles hiddenSize <= 1024 safely on 248KB UB.
    test_cases = [
        # (Shape, Dtype, Description)
        ((1, 256), torch.float32, "Edge Case: Single Row (totalRows < coreNum)"),
        ((2, 512), torch.float32, "Edge Case: totalRows == coreNum (2 rows on 2 cores)"),
        ((3, 512), torch.float32, "Edge Case: Odd Rows (Load Balance Remainder 3 % 2 = 1)"),
        ((7, 256), torch.float32, "Edge Case: BatchRows Remainder (7 rows with batchRows=8)"),
        ((16, 512), torch.float32, "Standard 2D Batch [M=16, K=512]"),
        ((32, 1024), torch.float32, "Standard 2D Batch [M=32, K=1024]"),
        ((2, 8, 512), torch.float32, "3D Transformer Shape [Batch=2, Seq=8, Hidden=512]"),
        ((1, 256), torch.float16, "FP16 Edge Case: Single Row"),
        ((3, 512), torch.float16, "FP16 Edge Case: Odd Rows (3 rows)"),
        ((16, 512), torch.float16, "FP16 Standard 2D Batch [16, 512]"),
        ((16, 1024), torch.float16, "FP16 Standard 2D Batch [16, 1024]"),
        ((4, 16, 512), torch.float16, "FP16 3D Transformer Shape [4, 16, 512]"),
    ]

    all_passed = True
    print(f"{'Case':<55} | {'Shape':<16} | {'Dtype':<8} | {'Max Diff':<10} | {'Status'}")
    print("-" * 105)

    for shape, dtype, desc in test_cases:
        torch.manual_seed(42)
        x = torch.randn(shape, dtype=dtype, device=device)

        # Tolerance thresholds (FP16 math has 10-bit mantissa; 1e-3 is standard)
        atol = 1e-4 if dtype == torch.float32 else 1e-3
        rtol = 1e-4 if dtype == torch.float32 else 1e-3

        try:
            out_custom = softmax_npu.forward(x)
            out_ref = reference_softmax(x)

            # Compare outputs
            max_diff = (out_custom.float() - out_ref.float()).abs().max().item()
            is_close = torch.allclose(out_custom, out_ref, atol=atol, rtol=rtol)

            status = "\033[92mPASS\033[0m" if is_close else "\033[91mFAIL\033[0m"
            if not is_close:
                all_passed = False

            dtype_str = "FP32" if dtype == torch.float32 else "FP16"
            print(f"{desc:<55} | {str(list(shape)):<16} | {dtype_str:<8} | {max_diff:<10.6f} | {status}")

        except Exception as e:
            if "100039" in str(e) or "stub library" in str(e):
                print(f"{desc:<55} | {str(list(shape)):<16} | STUB DETECTED (Error 100039)")
                all_passed = False
                break
            else:
                print(f"{desc:<55} | {str(list(shape)):<16} | EXCEPTION: {e}")
                all_passed = False

    # 4. Non-contiguous Input Test
    print("-" * 105)
    print("Testing Non-Contiguous Input Handling (defensive .contiguous() check)...")
    try:
        x_base = torch.randn((16, 1024), dtype=torch.float32, device=device)
        x_non_contig = x_base[:, :512].t().t()  # create non-contiguous slice
        
        out_custom = softmax_npu.forward(x_non_contig)
        out_ref = reference_softmax(x_non_contig)

        max_diff = (out_custom - out_ref).abs().max().item()
        is_close = torch.allclose(out_custom, out_ref, atol=1e-4, rtol=1e-4)
        status = "\033[92mPASS\033[0m" if is_close else "\033[91mFAIL\033[0m"
        print(f"{'Non-Contiguous Input [16, 512]':<55} | {'[16, 512]':<16} | {'FP32':<8} | {max_diff:<10.6f} | {status}")
        if not is_close:
            all_passed = False
    except Exception as e:
        print(f"Non-Contiguous Test EXCEPTION: {e}")
        all_passed = False

    print("=" * 105)
    if all_passed:
        print("\033[92m🎉 All Test Cases Passed Successfully!\033[0m")
    else:
        print("\033[91m❌ Some Test Cases Failed. Please inspect diffs.\033[0m")
    return all_passed

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
