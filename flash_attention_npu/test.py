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
    import flash_attention_npu
except ImportError:
    print("[ERROR] Cannot import flash_attention_npu module. Please compile via: python3 setup.py build_ext --inplace")
    sys.exit(1)

def golden_flash_attention(q, k, v, scale=None):
    d = q.shape[-1]
    if scale is None:
        scale = 1.0 / (d ** 0.5)
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    scores = torch.matmul(q_f, k_f.transpose(-1, -2)) * scale
    p = F.softmax(scores, dim=-1)
    o = torch.matmul(p, v_f).half()
    l = torch.logsumexp(scores, dim=-1).half()
    return o, l

def run_test_case(name, N_q, N_k, d, scale=None, atol=1e-2, rtol=1e-2):
    device = "npu:0" if has_npu else "cpu"
    
    torch.manual_seed(42)
    q = torch.randn((N_q, d), dtype=torch.half, device=device)
    k = torch.randn((N_k, d), dtype=torch.half, device=device)
    v = torch.randn((N_k, d), dtype=torch.half, device=device)

    # 自定义 FlashAttention 算子计算
    if scale is not None:
        o_custom, l_custom = flash_attention_npu.forward(q, k, v, scale)
    else:
        o_custom, l_custom = flash_attention_npu.forward(q, k, v)

    # PyTorch 原生算子作为黄金标杆 (Golden)
    o_golden, l_golden = golden_flash_attention(q, k, v, scale)

    # 误差对比
    o_custom_cpu = o_custom.cpu().float()
    o_golden_cpu = o_golden.cpu().float()
    l_custom_cpu = l_custom.cpu().float()
    l_golden_cpu = l_golden.cpu().float()

    max_diff_o = torch.max(torch.abs(o_custom_cpu - o_golden_cpu)).item()
    passed_o = torch.allclose(o_custom_cpu, o_golden_cpu, atol=atol, rtol=rtol)

    max_diff_l = torch.max(torch.abs(l_custom_cpu - l_golden_cpu)).item()
    passed_l = torch.allclose(l_custom_cpu, l_golden_cpu, atol=atol, rtol=rtol)

    passed = passed_o and passed_l
    status = "PASS" if passed else "FAIL"
    shape_str = f"Q:[{N_q},{d}], K:[{N_k},{d}]"
    print(f"{name:<45} | {shape_str:<25} | MaxDiff O: {max_diff_o:7.4f} | MaxDiff L: {max_diff_l:7.4f} | Status: {status}")
    if not passed:
        print(f"  [FAIL DETAILS] O Custom: {o_custom_cpu[0, :5].tolist()}")
        print(f"                 O Golden: {o_golden_cpu[0, :5].tolist()}")
        print(f"                 L Custom: {l_custom_cpu[:5].tolist()}")
        print(f"                 L Golden: {l_golden_cpu[:5].tolist()}")
    return passed

def main():
    print("=" * 115)
    print("   Ascend C FlashAttention Forward Operator Test Suite (FP16 Cube-Vector)")
    print("=" * 115)
    print(f"[INFO] Using Device: {'Ascend NPU' if has_npu else 'CPU (Direct ACL Host Launch)'}")
    print(f"[INFO] Successfully loaded 'flash_attention_npu' module.\n")

    test_cases = [
        ("Base Block [N_q=64, N_k=64, d=64]", 64, 64, 64, None),
        ("Multi-Block K [N_q=64, N_k=128, d=64]", 64, 128, 64, None),
        ("Multi-Block Q [N_q=128, N_k=64, d=64]", 128, 64, 64, None),
        ("Square Multi-Block [N_q=128, N_k=128, d=64]", 128, 128, 64, None),
        ("Custom Scale 0.125 [N_q=64, N_k=64, d=64]", 64, 64, 64, 0.125),
        ("Deep Sequence [N_q=128, N_k=256, d=64]", 128, 256, 64, None),
    ]

    all_passed = True
    for name, N_q, N_k, d, scale in test_cases:
        passed = run_test_case(name, N_q, N_k, d, scale=scale)
        if not passed:
            all_passed = False

    print("=" * 115)
    if all_passed:
        print("🎉 All FlashAttention Test Cases Passed Successfully!")
    else:
        print("❌ Some test cases failed. Please inspect logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
