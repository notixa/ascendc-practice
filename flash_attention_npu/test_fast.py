import os

os.environ["TE_PARALLEL_NUM"] = "1"
os.environ["TBE_PARALLEL_NUM"] = "1"
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
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
    print("[ERROR] Cannot import flash_attention_npu module.")
    sys.exit(1)


def run_fast_case(N_q, N_k, d, scale=None):
    torch.manual_seed(42)
    if scale is None:
        scale = 1.0 / (d**0.5)

    # 1. CPU Golden computation (Instantaneous, avoids NPU graph compiler!)
    q_cpu = torch.randn((N_q, d), dtype=torch.float32)
    k_cpu = torch.randn((N_k, d), dtype=torch.float32)
    v_cpu = torch.randn((N_k, d), dtype=torch.float32)
    do_cpu = torch.randn((N_q, d), dtype=torch.float32)

    scores = torch.matmul(q_cpu, k_cpu.T) * scale
    p = F.softmax(scores, dim=-1)
    o_golden = torch.matmul(p, v_cpu)
    l_golden = torch.logsumexp(scores, dim=-1)

    # Backward golden
    D = (do_cpu * o_golden).sum(dim=-1, keepdim=True)
    dP = torch.matmul(do_cpu, v_cpu.T)
    dS = p * (dP - D) * scale
    dq_golden = torch.matmul(dS, k_cpu)
    dk_golden = torch.matmul(dS.T, q_cpu)
    dv_golden = torch.matmul(p.T, do_cpu)

    # 2. Transfer to NPU via simple DMA (no GE graphs!)
    device = "npu:0" if has_npu else "cpu"
    q_npu = q_cpu.half().to(device)
    k_npu = k_cpu.half().to(device)
    v_npu = v_cpu.half().to(device)
    do_npu = do_cpu.half().to(device)
    o_ref_npu = o_golden.half().to(device)
    l_ref_npu = l_golden.half().to(device)

    # 3. Custom Forward
    o_custom, l_custom = flash_attention_npu.forward(q_npu, k_npu, v_npu, scale)
    if has_npu:
        torch.npu.synchronize()

    diff_o = (o_custom.cpu().float() - o_golden).abs().max().item()
    diff_l = (l_custom.cpu().float() - l_golden).abs().max().item()

    # 4. Custom Backward (isolated with golden O, L)
    dq_custom, dk_custom, dv_custom = flash_attention_npu.backward(
        do_npu, q_npu, k_npu, v_npu, o_ref_npu, l_ref_npu, scale
    )
    if has_npu:
        torch.npu.synchronize()

    diff_dq = (dq_custom.cpu().float() - dq_golden).abs().max().item()
    diff_dk = (dk_custom.cpu().float() - dk_golden).abs().max().item()
    diff_dv = (dv_custom.cpu().float() - dv_golden).abs().max().item()

    print(f"[{N_q}x{N_k}, d={d}] Forward:  max diff O = {diff_o:.5f}, L = {diff_l:.5f}")
    print(
        f"[{N_q}x{N_k}, d={d}] Backward: max diff dQ = {diff_dq:.5f}, dK = {diff_dk:.5f}, dV = {diff_dv:.5f}"
    )

    if diff_dk > 0.05:
        print(
            "  dK custom [:6]: ",
            [round(x, 4) for x in dk_custom.cpu().float()[0, :6].tolist()],
        )
        print("  dK golden [:6]: ", [round(x, 4) for x in dk_golden[0, :6].tolist()])
        # Check transpose
        diff_dkt = (dk_custom.cpu().float() - dk_golden.T).abs().max().item()
        print(f"  dK vs golden.T max diff: {diff_dkt:.5f}")


if __name__ == "__main__":
    if has_npu:
        torch.npu.set_device(0)
    print("Testing 64x64...")
    run_fast_case(64, 64, 64)
