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

torch.npu.set_device(0)
torch.manual_seed(42)

N_q, N_k, d = 64, 64, 64
scale = 1.0 / (d**0.5)

q = torch.randn((N_q, d), dtype=torch.half, device="npu:0")
k = torch.randn((N_k, d), dtype=torch.half, device="npu:0")
v = torch.randn((N_k, d), dtype=torch.half, device="npu:0")
do = torch.randn((N_q, d), dtype=torch.half, device="npu:0")

# 1. Custom Forward
o_custom, l_custom = flash_attention_npu.forward(q, k, v)
torch.npu.synchronize()

# 2. Golden Forward & Backward via PyTorch Autograd
q_ref = q.cpu().float().clone().detach().requires_grad_(True)
k_ref = k.cpu().float().clone().detach().requires_grad_(True)
v_ref = v.cpu().float().clone().detach().requires_grad_(True)
do_ref = do.cpu().float()

scores_ref = torch.matmul(q_ref, k_ref.transpose(-1, -2)) * scale
p_ref = F.softmax(scores_ref, dim=-1)
o_ref = torch.matmul(p_ref, v_ref)
l_ref = torch.logsumexp(scores_ref, dim=-1)

loss = (o_ref * do_ref).sum()
loss.backward()

dq_golden = q_ref.grad.half()
dk_golden = k_ref.grad.half()
dv_golden = v_ref.grad.half()

# 3. Custom Backward using custom o & l
dq_custom, dk_custom, dv_custom = flash_attention_npu.backward(
    do, q, k, v, o_custom, l_custom
)
torch.npu.synchronize()

print("--- Comparison with Custom O, L ---")
diff_dq = (dq_custom.cpu().float() - dq_golden.float()).abs()
diff_dk = (dk_custom.cpu().float() - dk_golden.float()).abs()
diff_dv = (dv_custom.cpu().float() - dv_golden.float()).abs()

print(f"Max diff dQ: {diff_dq.max().item():.5f}, mean: {diff_dq.mean().item():.5f}")
print(f"Max diff dK: {diff_dk.max().item():.5f}, mean: {diff_dk.mean().item():.5f}")
print(f"Max diff dV: {diff_dv.max().item():.5f}, mean: {diff_dv.mean().item():.5f}")

# 4. Custom Backward using Golden O, L (to isolate backward kernel from forward errors)
o_golden_npu = o_ref.half().to("npu:0")
l_golden_npu = l_ref.half().to("npu:0")

dq_c2, dk_c2, dv_c2 = flash_attention_npu.backward(
    do, q, k, v, o_golden_npu, l_golden_npu
)
torch.npu.synchronize()

print("\n--- Comparison with Golden O, L (Isolated Backward) ---")
diff_dq2 = (dq_c2.cpu().float() - dq_golden.float()).abs()
diff_dk2 = (dk_c2.cpu().float() - dk_golden.float()).abs()
diff_dv2 = (dv_c2.cpu().float() - dv_golden.float()).abs()

print(f"Max diff dQ: {diff_dq2.max().item():.5f}, mean: {diff_dq2.mean().item():.5f}")
print(f"Max diff dK: {diff_dk2.max().item():.5f}, mean: {diff_dk2.mean().item():.5f}")
print(f"Max diff dV: {diff_dv2.max().item():.5f}, mean: {diff_dv2.mean().item():.5f}")

print("\n--- Detailed dK inspection in Section 4 ---")
print("dk_c2 first row [:8]:")
print("  Custom dk_c2: ", [round(x, 5) for x in dk_c2.cpu().float()[0, :8].tolist()])
print("  Golden dk:    ", [round(x, 5) for x in dk_golden.float()[0, :8].tolist()])
print("dk_c2 second row [:8]:")
print("  Custom dk_c2: ", [round(x, 5) for x in dk_c2.cpu().float()[1, :8].tolist()])
print("  Golden dk:    ", [round(x, 5) for x in dk_golden.float()[1, :8].tolist()])
print("Ratio dk_c2 / dk_golden (first 8 non-zeros):")
ratio = dk_c2.cpu().float()[0, :8] / (dk_golden.float()[0, :8] + 1e-7)
print("  Ratio:        ", [round(x, 4) for x in ratio.tolist()])

# Check if dk_c2 matches dk_golden transpose or permuted
diff_transpose = (dk_c2.cpu().float() - dk_golden.float().T).abs().max().item()
print("Diff against transpose: ", diff_transpose)

# Check forward O and L diff
diff_o = (o_custom.cpu().float() - o_ref.float()).abs().max().item()
diff_l = (l_custom.cpu().float() - l_ref.float()).abs().max().item()
print(f"\nForward check: max diff O = {diff_o:.5f}, max diff L = {diff_l:.5f}")
