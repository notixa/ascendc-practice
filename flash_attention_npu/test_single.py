import sys
import torch
import torch_npu

import flash_attention_npu

print("[1] Initializing tensors on NPU...")
N_q, N_k, d = 64, 64, 64
device = "npu:0"
q = torch.randn((N_q, d), dtype=torch.half, device=device)
k = torch.randn((N_k, d), dtype=torch.half, device=device)
v = torch.randn((N_k, d), dtype=torch.half, device=device)
do = torch.randn((N_q, d), dtype=torch.half, device=device)

print("[2] Calling flash_attention_npu.forward...")
o, l = flash_attention_npu.forward(q, k, v)
torch.npu.synchronize()
print("[3] Forward completed successfully! Output shapes:", o.shape, l.shape)
print("    O sample:", o.cpu().flatten()[:4].tolist())
print("    L sample:", l.cpu().flatten()[:4].tolist())

print("[4] Calling flash_attention_npu.backward...")
dq, dk, dv = flash_attention_npu.backward(do, q, k, v, o, l)
torch.npu.synchronize()
print("[5] Backward completed successfully! Output shapes:", dq.shape, dk.shape, dv.shape)
print("    dQ sample:", dq.cpu().flatten()[:4].tolist())
print("    dK sample:", dk.cpu().flatten()[:4].tolist())
print("    dV sample:", dv.cpu().flatten()[:4].tolist())

print("ALL PASSED!")
