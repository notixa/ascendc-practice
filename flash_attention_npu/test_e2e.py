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

torch.npu.set_device(0)
torch.manual_seed(42)

def test_shape(N_q, N_k, d):
    print(f"\n========================================================")
    print(f"Testing Shape: N_q={N_q}, N_k={N_k}, d={d}")
    print(f"========================================================")
    scale = 1.0 / (d ** 0.5)

    # 1. CPU Reference (FP32)
    q_cpu = torch.randn((N_q, d), dtype=torch.float32)
    k_cpu = torch.randn((N_k, d), dtype=torch.float32)
    v_cpu = torch.randn((N_k, d), dtype=torch.float32)
    do_cpu = torch.randn((N_q, d), dtype=torch.float32)

    scores_ref = torch.matmul(q_cpu, k_cpu.T) * scale
    p_ref = F.softmax(scores_ref, dim=-1)
    o_ref = torch.matmul(p_ref, v_cpu)
    l_ref = torch.logsumexp(scores_ref, dim=-1)

    D_ref = (do_cpu * o_ref).sum(dim=-1, keepdim=True)
    dP_ref = torch.matmul(do_cpu, v_cpu.T)
    dS_ref = p_ref * (dP_ref - D_ref) * scale
    dq_ref = torch.matmul(dS_ref, k_cpu)
    dk_ref = torch.matmul(dS_ref.T, q_cpu)
    dv_ref = torch.matmul(p_ref.T, do_cpu)

    # 2. Prepare NPU inputs
    q_npu = q_cpu.half().to("npu:0")
    k_npu = k_cpu.half().to("npu:0")
    v_npu = v_cpu.half().to("npu:0")
    do_npu = do_cpu.half().to("npu:0")
    o_ref_npu = o_ref.half().to("npu:0")
    l_ref_npu = l_ref.half().to("npu:0")
    torch.npu.synchronize()

    # 3. Test Forward
    t0 = time.time()
    o_c, l_c = flash_attention_npu.forward(q_npu, k_npu, v_npu, scale)
    torch.npu.synchronize()
    t_fwd = (time.time() - t0) * 1000

    diff_o = (o_c.cpu().float() - o_ref).abs()
    diff_l = (l_c.cpu().float() - l_ref).abs()

    print(f"Forward ({t_fwd:.2f} ms):")
    print(f"  O max diff: {diff_o.max().item():.5f}, mean: {diff_o.mean().item():.5f}")
    print(f"  L max diff: {diff_l.max().item():.5f}, mean: {diff_l.mean().item():.5f}")

    # 4. Test Backward (Isolated with Golden O, L)
    t0 = time.time()
    dq_iso, dk_iso, dv_iso = flash_attention_npu.backward(
        do_npu, q_npu, k_npu, v_npu, o_ref_npu, l_ref_npu, scale
    )
    torch.npu.synchronize()
    t_bwd_iso = (time.time() - t0) * 1000

    diff_dq_iso = (dq_iso.cpu().float() - dq_ref).abs()
    diff_dk_iso = (dk_iso.cpu().float() - dk_ref).abs()
    diff_dv_iso = (dv_iso.cpu().float() - dv_ref).abs()

    print(f"\nBackward Isolated ({t_bwd_iso:.2f} ms):")
    print(f"  dQ max diff: {diff_dq_iso.max().item():.5f}, mean: {diff_dq_iso.mean().item():.5f}")
    print(f"  dK max diff: {diff_dk_iso.max().item():.5f}, mean: {diff_dk_iso.mean().item():.5f}")
    print(f"  dV max diff: {diff_dv_iso.max().item():.5f}, mean: {diff_dv_iso.mean().item():.5f}")

    # 5. Test Backward (Chained with Custom O, L)
    t0 = time.time()
    dq_c, dk_c, dv_c = flash_attention_npu.backward(
        do_npu, q_npu, k_npu, v_npu, o_c, l_c, scale
    )
    torch.npu.synchronize()
    t_bwd_c = (time.time() - t0) * 1000

    diff_dq_c = (dq_c.cpu().float() - dq_ref).abs()
    diff_dk_c = (dk_c.cpu().float() - dk_ref).abs()
    diff_dv_c = (dv_c.cpu().float() - dv_ref).abs()

    print(f"\nBackward Chained ({t_bwd_c:.2f} ms):")
    print(f"  dQ max diff: {diff_dq_c.max().item():.5f}, mean: {diff_dq_c.mean().item():.5f}")
    print(f"  dK max diff: {diff_dk_c.max().item():.5f}, mean: {diff_dk_c.mean().item():.5f}")
    print(f"  dV max diff: {diff_dv_c.max().item():.5f}, mean: {diff_dv_c.mean().item():.5f}")

    # 6. Detailed dK row analysis
    dk_row_max = diff_dk_iso.max(dim=-1).values
    bad_rows = [r for r in range(N_k) if dk_row_max[r].item() > 0.02]
    print(f"\n  dK Isolated bad rows (>0.02): {len(bad_rows)} / {N_k}")
    if bad_rows:
        print(f"    Bad row indices: {bad_rows[:10]}...")

    # Verdict
    fwd_pass = (diff_o.max().item() < 0.02 and diff_l.max().item() < 0.02)
    bwd_pass = (diff_dq_iso.max().item() < 0.02 and diff_dk_iso.max().item() < 0.02 and diff_dv_iso.max().item() < 0.02)
    print(f"\n>>> Verdict: FWD {'PASS' if fwd_pass else 'FAIL'}, BWD {'PASS' if bwd_pass else 'FAIL'}")
    return fwd_pass and bwd_pass

if __name__ == "__main__":
    shapes = [
        (64, 64, 64),
        (64, 128, 64),
        (128, 64, 64),
        (128, 128, 64),
    ]
    all_passed = True
    for nq, nk, d in shapes:
        ok = test_shape(nq, nk, d)
        if not ok:
            all_passed = False
    print(f"\n========================================================")
    print(f"OVERALL SUMMARY: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print(f"========================================================")
