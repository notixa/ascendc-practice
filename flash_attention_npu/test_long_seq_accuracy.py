import os
os.environ["TE_PARALLEL_COMPILER"] = "1"
os.environ["TE_PARALLEL_NUM"] = "1"
os.environ["TBE_PARALLEL_NUM"] = "1"
os.environ["MAX_COMPILE_CORE_NUMBER"] = "1"
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
import torch
import torch_npu
import torch.nn.functional as F
import flash_attention_npu
import time
import math

torch.npu.set_device(0)
torch.manual_seed(42)

def evaluate_accuracy(name, b, h, nq, nk, d, check_bwd=True):
    print(f"\n{'='*70}")
    print(f"[{name}] Testing B={b}, H={h}, N_q={nq}, N_k={nk}, d={d}")
    print(f"{'='*70}")
    scale = 1.0 / math.sqrt(d)

    # 1. CPU Reference in FP32
    if b == 1 and h == 1:
        shape_q = (nq, d)
        shape_k = (nk, d)
        shape_l = (nq,)
    elif b == 1:
        shape_q = (h, nq, d)
        shape_k = (h, nk, d)
        shape_l = (h, nq)
    else:
        shape_q = (b, h, nq, d)
        shape_k = (b, h, nk, d)
        shape_l = (b, h, nq)

    q_cpu = torch.randn(shape_q, dtype=torch.float32)
    k_cpu = torch.randn(shape_k, dtype=torch.float32)
    v_cpu = torch.randn(shape_k, dtype=torch.float32)
    do_cpu = torch.randn(shape_q, dtype=torch.float32) if check_bwd else None

    # Forward Golden
    scores_ref = torch.matmul(q_cpu, k_cpu.transpose(-1, -2)) * scale
    p_ref = F.softmax(scores_ref, dim=-1)
    o_ref = torch.matmul(p_ref, v_cpu)
    l_ref = torch.logsumexp(scores_ref, dim=-1)

    # Backward Golden
    if check_bwd:
        D_ref = (do_cpu * o_ref).sum(dim=-1, keepdim=True)
        dP_ref = torch.matmul(do_cpu, v_cpu.transpose(-1, -2))
        dS_ref = p_ref * (dP_ref - D_ref) * scale
        dq_ref = torch.matmul(dS_ref, k_cpu)
        dk_ref = torch.matmul(dS_ref.transpose(-1, -2), q_cpu)
        dv_ref = torch.matmul(p_ref.transpose(-1, -2), do_cpu)

    # 2. Custom NPU Execution (FP16)
    q_npu = q_cpu.half().to("npu:0")
    k_npu = k_cpu.half().to("npu:0")
    v_npu = v_cpu.half().to("npu:0")
    if check_bwd:
        do_npu = do_cpu.half().to("npu:0")
    torch.npu.synchronize()

    t0 = time.time()
    o_c, l_c = flash_attention_npu.forward(q_npu, k_npu, v_npu, scale)
    torch.npu.synchronize()
    t_fwd = (time.time() - t0) * 1000

    o_c_cpu = o_c.cpu().float()
    l_c_cpu = l_c.cpu().float()

    diff_o = (o_c_cpu - o_ref).abs()
    diff_l = (l_c_cpu - l_ref).abs()

    rel_o = diff_o / (o_ref.abs() + 1e-5)
    rel_l = diff_l / (l_ref.abs() + 1e-5)

    has_nan_o = torch.isnan(o_c_cpu).any().item()
    has_inf_o = torch.isinf(o_c_cpu).any().item()
    has_nan_l = torch.isnan(l_c_cpu).any().item()
    has_inf_l = torch.isinf(l_c_cpu).any().item()

    print(f"Forward (Time: {t_fwd:.2f} ms):")
    print(f"  O -> Max Diff: {diff_o.max().item():.5f}, Mean Diff: {diff_o.mean().item():.5f}, Max Rel: {rel_o.max().item():.4f}, NaN/Inf: {has_nan_o}/{has_inf_o}")
    print(f"  L -> Max Diff: {diff_l.max().item():.5f}, Mean Diff: {diff_l.mean().item():.5f}, Max Rel: {rel_l.max().item():.4f}, NaN/Inf: {has_nan_l}/{has_inf_l}")

    bwd_res = {}
    if check_bwd:
        t0 = time.time()
        dq_c, dk_c, dv_c = flash_attention_npu.backward(
            do_npu, q_npu, k_npu, v_npu, o_c, l_c, scale
        )
        torch.npu.synchronize()
        t_bwd = (time.time() - t0) * 1000

        dq_c_cpu = dq_c.cpu().float()
        dk_c_cpu = dk_c.cpu().float()
        dv_c_cpu = dv_c.cpu().float()

        diff_dq = (dq_c_cpu - dq_ref).abs()
        diff_dk = (dk_c_cpu - dk_ref).abs()
        diff_dv = (dv_c_cpu - dv_ref).abs()

        print(f"Backward (Time: {t_bwd:.2f} ms):")
        print(f"  dQ -> Max Diff: {diff_dq.max().item():.5f}, Mean Diff: {diff_dq.mean().item():.5f}")
        print(f"  dK -> Max Diff: {diff_dk.max().item():.5f}, Mean Diff: {diff_dk.mean().item():.5f}")
        print(f"  dV -> Max Diff: {diff_dv.max().item():.5f}, Mean Diff: {diff_dv.mean().item():.5f}")

        bwd_pass = (diff_dq.max().item() < 0.03 and diff_dk.max().item() < 0.03 and diff_dv.max().item() < 0.03)
        bwd_res = {
            "dq_max": diff_dq.max().item(),
            "dk_max": diff_dk.max().item(),
            "dv_max": diff_dv.max().item(),
            "bwd_pass": bwd_pass,
            "t_bwd": t_bwd,
        }

    fwd_pass = (diff_o.max().item() < 0.02 and diff_l.max().item() < 0.02 and not has_nan_o and not has_inf_o)

    verdict = fwd_pass and (bwd_res.get("bwd_pass", True))
    print(f">>> Verdict: {'PASS' if verdict else 'FAIL'} (FWD: {fwd_pass}, BWD: {bwd_res.get('bwd_pass', True)})")

    return {
        "name": name,
        "shape": f"B={b}, H={h}, N={nq}, d={d}",
        "fwd_time_ms": t_fwd,
        "o_max_diff": diff_o.max().item(),
        "o_mean_diff": diff_o.mean().item(),
        "l_max_diff": diff_l.max().item(),
        "l_mean_diff": diff_l.mean().item(),
        "has_nan": has_nan_o or has_nan_l,
        "has_inf": has_inf_o or has_inf_l,
        "fwd_pass": fwd_pass,
        **bwd_res
    }

if __name__ == "__main__":
    test_cases = [
        # 1. 序列长度渐增曲线 (单头验证长序列累积误差)
        ("Seq-128", 1, 1, 128, 128, 64, True),
        ("Seq-256", 1, 1, 256, 256, 64, True),
        ("Seq-512", 1, 1, 512, 512, 64, True),
        ("Seq-1024", 1, 1, 1024, 1024, 64, True),
        ("Seq-2048", 1, 1, 2048, 2048, 64, True),

        # 2. 真实多头 / 大 Batch 场景 (大模型注意力配置)
        ("LLaMA-Style-H4-N512", 1, 4, 512, 512, 64, True),
        ("LLaMA-Style-H4-N1024", 1, 4, 1024, 1024, 64, True),
        ("Batch2-H4-N512", 2, 4, 512, 512, 64, True),
        ("Batch2-H8-N512", 2, 8, 512, 512, 64, True),
        ("Batch1-H8-N1024", 1, 8, 1024, 1024, 64, True),
        ("Batch1-H8-N2048", 1, 8, 2048, 2048, 64, True),
    ]

    results = []
    for name, b, h, nq, nk, d, check_bwd in test_cases:
        res = evaluate_accuracy(name, b, h, nq, nk, d, check_bwd)
        results.append(res)

    print("\n" + "="*85)
    print("                    ACCURACY SUMMARY TABLE")
    print("="*85)
    print(f"{'Case Name':<22} | {'Shape':<22} | {'O Max':<8} | {'O Mean':<8} | {'L Max':<8} | {'dQ/dK/dV Max':<15} | {'Verdict'}")
    print("-" * 85)
    all_ok = True
    for r in results:
        bwd_str = f"{r.get('dq_max',0):.4f}/{r.get('dk_max',0):.4f}/{r.get('dv_max',0):.4f}" if 'dq_max' in r else "N/A"
        pass_str = "PASS" if (r['fwd_pass'] and r.get('bwd_pass', True)) else "FAIL"
        if pass_str == "FAIL":
            all_ok = False
        print(f"{r['name']:<22} | {r['shape']:<22} | {r['o_max_diff']:<8.5f} | {r['o_mean_diff']:<8.5f} | {r['l_max_diff']:<8.5f} | {bwd_str:<15} | {pass_str}")
    print("="*85)
    print(f"Final Accuracy Assessment: {'100% ALL PASSED' if all_ok else 'SOME CASES FAILED'}")
