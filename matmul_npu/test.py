import os
import sys
import torch

try:
    import torch_npu
    has_npu = torch.npu.is_available()
except ImportError:
    has_npu = False

try:
    import matmul_npu
except ImportError:
    print("[ERROR] Cannot import matmul_npu module. Please compile via: python3 setup.py build_ext --inplace")
    sys.exit(1)

def run_test_case(name, M, K, N, atol=2e-3, rtol=1e-3):
    device = "npu:0" if has_npu else "cpu"
    
    # 初始化输入
    torch.manual_seed(42)
    a = torch.randn((M, K), dtype=torch.half, device=device)
    b = torch.randn((K, N), dtype=torch.half, device=device)

    # 自定义算子计算
    c_custom = matmul_npu.forward(a, b)

    # PyTorch 原生算子作为黄金标杆 (Golden)
    c_golden = torch.matmul(a, b)

    # 误差对比
    c_custom_cpu = c_custom.cpu().float()
    c_golden_cpu = c_golden.cpu().float()

    max_diff = torch.max(torch.abs(c_custom_cpu - c_golden_cpu)).item()
    passed = torch.allclose(c_custom_cpu, c_golden_cpu, atol=atol, rtol=rtol)

    status = "PASS" if passed else "FAIL"
    shape_str = f"[{M}, {K}] x [{K}, {N}]"
    print(f"{name:<45} | {shape_str:<25} | Max Diff: {max_diff:8.6f} | Status: {status}")
    if not passed:
        print(f"  [FAIL DETAILS] Custom sample: {c_custom_cpu[0, :5].tolist()}")
        print(f"                 Golden sample: {c_golden_cpu[0, :5].tolist()}")
    return passed

def main():
    print("=" * 95)
    print("        Ascend C MatMul Custom Operator Test Suite (FP16 GEMM)")
    print("=" * 95)
    print(f"[INFO] Using Device: {'Ascend NPU' if has_npu else 'CPU (Direct ACL Host Launch)'}")
    print(f"[INFO] Successfully loaded 'matmul_npu' module.\n")

    test_cases = [
        ("Base Block [M=64, K=64, N=64]", 64, 64, 64),
        ("Small Aligned [M=32, K=48, N=32]", 32, 48, 32),
        ("Tall Matrix [M=128, K=64, N=64]", 128, 64, 64),
        ("Wide Matrix [M=64, K=64, N=128]", 64, 64, 128),
        ("Deep K Reduction [M=64, K=256, N=64]", 64, 256, 64),
        ("Medium GEMM [M=128, K=128, N=128]", 128, 128, 128),
        ("Square GEMM [M=256, K=256, N=256]", 256, 256, 256),
        ("Transformer FFN Shape [M=128, K=512, N=1024]", 128, 512, 1024),
    ]

    all_passed = True
    for name, M, K, N in test_cases:
        passed = run_test_case(name, M, K, N)
        if not passed:
            all_passed = False

    print("=" * 95)
    if all_passed:
        print("🎉 All Test Cases Passed Successfully!")
    else:
        print("❌ Some test cases failed. Please inspect logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
