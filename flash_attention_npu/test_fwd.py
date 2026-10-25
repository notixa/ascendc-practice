import os

os.environ["TE_PARALLEL_NUM"] = "1"
os.environ["TBE_PARALLEL_NUM"] = "1"
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
import torch
import torch_npu
import torch.nn.functional as F
import flash_attention_npu
import time

torch.npu.set_device(0)
torch.manual_seed(42)

N_q, N_k, d = 64, 64, 64
scale = 1.0 / (d**0.5)

print(f"Creating tensors shape ({N_q}, {d})...")
q_cpu = torch.randn((N_q, d), dtype=torch.float32)
k_cpu = torch.randn((N_k, d), dtype=torch.float32)
v_cpu = torch.randn((N_k, d), dtype=torch.float32)

print("Computing CPU reference...")
scores = torch.matmul(q_cpu, k_cpu.T) * scale
p = F.softmax(scores, dim=-1)
o_ref = torch.matmul(p, v_cpu)
l_ref = torch.logsumexp(scores, dim=-1)

q_npu = q_cpu.half().to("npu:0")
k_npu = k_cpu.half().to("npu:0")
v_npu = v_cpu.half().to("npu:0")
torch.npu.synchronize()

print("Calling flash_attention_npu.forward...")
t0 = time.time()
o_c, l_c = flash_attention_npu.forward(q_npu, k_npu, v_npu, scale)
torch.npu.synchronize()
t1 = time.time()
print(f"Forward completed in {(t1 - t0) * 1000:.2f} ms")

diff_o = (o_c.cpu().float() - o_ref).abs()
diff_l = (l_c.cpu().float() - l_ref).abs()

print(f"Max diff O: {diff_o.max().item():.5f}, mean: {diff_o.mean().item():.5f}")
print(f"Max diff L: {diff_l.max().item():.5f}, mean: {diff_l.mean().item():.5f}")

row_max_o = diff_o.max(dim=-1).values
print("\n=== Rows of O with diff > 0.02 ===")
bad_o_rows = []
for r in range(N_q):
    if row_max_o[r].item() > 0.02:
        bad_o_rows.append(r)
        print(
            f"Row {r:2d}: max diff O = {row_max_o[r].item():.5f}, L_custom = {l_c.cpu().float()[r].item():.4f}, L_ref = {l_ref[r].item():.4f}"
        )

print(f"Total bad O rows: {len(bad_o_rows)} / {N_q}")
