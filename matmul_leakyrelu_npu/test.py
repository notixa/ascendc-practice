import os
import sys
import torch
import torch.nn.functional as F

try:
    import torch_npu
    has_npu = torch.npu.is_available()
except ImportError:
    has_npu = False

try:
    import matmul_leakyrelu_npu
except ImportError:
    print("[ERROR] Cannot import matmul_leakyrelu_npu module. Please compile via: python3 setup.py build_ext --inplace")
    sys.exit(1)

def run_test_case(name, M, K, N, negative_slope=0.01, atol=2e-3, rtol=1e-3):
    device = "npu:0" if has_npu else "cpu"
    
    # 初始化输入（包含正数和负数以充分测试 LeakyReLU 分支）
    torch.manual_seed(42)
    a = torch.randn((M, K), dtype=torch.half, device=device)
    b = torch.randn((K, N), dtype=torch.half, device=device)

    # 自定义 Cube-Vector 融合算子计算
    c_custom = matmul_leakyrelu_npu.forward(a, b, negative_slope)

    # PyTorch 原生算子作为黄金标杆 (Golden)
    # PyTorch: leaky_relu(matmul(A, B), negative_slope)
    c_golden = F.leaky_relu(torch.matmul(a, b), negative_slope=negative_slope)

    # 误差对比
    c_custom_cpu = c_custom.cpu().float()
    c_golden_cpu = c_golden.cpu().float()

    max_diff = torch.max(torch.abs(c_custom_cpu - c_golden_cpu)).item()
    passed = torch.allclose(c_custom_cpu, c_golden_cpu, atol=atol, rtol=rtol)

    status = "PASS" if passed else "FAIL"
    shape_str = f"[{M}, {K}] x [{K}, {N}] (slope={negative_slope})"
    print(f"{name:<45} | {shape_str:<32} | Max Diff: {max_diff:8.6f} | Status: {status}")
    if not passed:
        print(f"  [FAIL DETAILS] Custom sample: {c_custom_cpu[0, :5].tolist()}")
        print(f"                 Golden sample: {c_golden_cpu[0, :5].tolist()}")
    return passed

def main():
    print("=" * 105)
    print("   Ascend C MatMul + LeakyReLU Fusion Operator Test Suite (FP16 Cube-Vector GEMM)")
    print("=" * 105)
    print(f"[INFO] Using Device: {'Ascend NPU' if has_npu else 'CPU (Direct ACL Host Launch)'}")
    print(f"[INFO] Successfully loaded 'matmul_leakyrelu_npu' module.\n")

    test_cases = [
        ("Base Block [M=64, K=64, N=64]", 64, 64, 64, 0.01),
        ("Custom Slope 0.1 [M=64, K=64, N=64]", 64, 64, 64, 0.1),
        ("Small Aligned [M=32, K=48, N=32]", 32, 48, 32, 0.01),
        ("Tall Matrix [M=128, K=64, N=64]", 128, 64, 64, 0.01),
        ("Wide Matrix [M=64, K=64, N=128]", 64, 64, 128, 0.01),
        ("Deep K Reduction [M=64, K=256, N=64]", 64, 256, 64, 0.01),
        ("Medium GEMM [M=128, K=128, N=128]", 128, 128, 128, 0.02),
        ("Square GEMM [M=256, K=256, N=256]", 256, 256, 256, 0.01),
        ("Transformer FFN Shape [M=128, K=512, N=1024]", 128, 512, 1024, 0.01),
    ]

    all_passed = True
    for name, M, K, N, slope in test_cases:
        passed = run_test_case(name, M, K, N, negative_slope=slope)
        if not passed:
            all_passed = False

    print("=" * 105)
    if all_passed:
        print("🎉 All Test Cases Passed Successfully!")
    else:
        print("❌ Some test cases failed. Please inspect logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
