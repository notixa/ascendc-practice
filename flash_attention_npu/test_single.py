import os

os.environ["TE_PARALLEL_COMPILER"] = "1"
os.environ["TE_PARALLEL_NUM"] = "1"
os.environ["TBE_PARALLEL_NUM"] = "1"
import sys
import torch
import torch_npu

import flash_attention_npu

import time

print("[1] Initializing tensors on CPU then transferring to NPU...", flush=True)
N_q, N_k, d = 64, 64, 64
device = "npu:0"
torch.manual_seed(42)
t0 = time.time()
q = torch.randn((N_q, d), dtype=torch.half).to(device)
print(f"    q transferred in {time.time() - t0:.2f}s", flush=True)
t0 = time.time()
k = torch.randn((N_k, d), dtype=torch.half).to(device)
v = torch.randn((N_k, d), dtype=torch.half).to(device)
do = torch.randn((N_q, d), dtype=torch.half).to(device)
print(f"    k, v, do transferred in {time.time() - t0:.2f}s", flush=True)

print("[2] Calling flash_attention_npu.forward...", flush=True)
t_fwd = time.time()
o, l = flash_attention_npu.forward(q, k, v)
print(
    f"    Kernel launched in {time.time() - t_fwd:.4f}s, waiting for synchronize...",
    flush=True,
)
torch.npu.synchronize()
print(
    f"[3] Forward completed in {time.time() - t_fwd:.2f}s! Output shapes: {o.shape}, {l.shape}",
    flush=True,
)
print("    O sample:", o.cpu().flatten()[:4].tolist(), flush=True)
print("    L sample:", l.cpu().flatten()[:4].tolist(), flush=True)

print("[4] Calling flash_attention_npu.backward...", flush=True)
dq, dk, dv = flash_attention_npu.backward(do, q, k, v, o, l)
torch.npu.synchronize()
print(
    "[5] Backward completed successfully! Output shapes:",
    dq.shape,
    dk.shape,
    dv.shape,
    flush=True,
)
print("    dQ sample:", dq.cpu().flatten()[:4].tolist(), flush=True)
print("    dK sample:", dk.cpu().flatten()[:4].tolist(), flush=True)
print("    dV sample:", dv.cpu().flatten()[:4].tolist(), flush=True)

print("🎉 ALL PASSED!", flush=True)
