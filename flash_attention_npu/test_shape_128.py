import os

os.environ["TE_PARALLEL_COMPILER"] = "1"
os.environ["TE_PARALLEL_NUM"] = "1"
os.environ["TBE_PARALLEL_NUM"] = "1"
os.environ["MAX_COMPILE_CORE_NUMBER"] = "1"
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
import sys
import time
import torch
import torch_npu
import flash_attention_npu

device = "npu:0"
torch.manual_seed(42)

N_q, N_k, d = 64, 128, 64
scale = 1.0 / (d**0.5)

print(f"[1] Allocating tensors for shape ({N_q}, {N_k}, {d})...", flush=True)
q = torch.randn((N_q, d), dtype=torch.half).to(device)
k = torch.randn((N_k, d), dtype=torch.half).to(device)
v = torch.randn((N_k, d), dtype=torch.half).to(device)
do = torch.randn((N_q, d), dtype=torch.half).to(device)
torch.npu.synchronize()
print("[1] Tensors allocated successfully!", flush=True)

print("[2] Running Forward...", flush=True)
t0 = time.time()
o, l = flash_attention_npu.forward(q, k, v, scale)
torch.npu.synchronize()
print(
    f"[2] Forward succeeded in {time.time() - t0:.4f}s! Shapes: O={o.shape}, L={l.shape}",
    flush=True,
)

print("[3] Running Backward...", flush=True)
t0 = time.time()
dq, dk, dv = flash_attention_npu.backward(do, q, k, v, o, l, scale)
torch.npu.synchronize()
print(
    f"[3] Backward succeeded in {time.time() - t0:.4f}s! Shapes: dQ={dq.shape}, dK={dk.shape}, dV={dv.shape}",
    flush=True,
)

print("Sample dQ:", dq.cpu().flatten()[:4].tolist(), flush=True)
print("Sample dK:", dk.cpu().flatten()[:4].tolist(), flush=True)
print("Sample dV:", dv.cpu().flatten()[:4].tolist(), flush=True)
print("🎉 ALL PASSED FOR SHAPE (64, 128, 64)!", flush=True)
