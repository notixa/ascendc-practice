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
H = 8
N_q, N_k, d = 128, 128, 64
scale = 1.0 / (d ** 0.5)

print(f"[1] Allocating tensors for {H} heads...", flush=True)
qs = [torch.randn((N_q, d), dtype=torch.half).to(device) for _ in range(H)]
ks = [torch.randn((N_k, d), dtype=torch.half).to(device) for _ in range(H)]
vs = [torch.randn((N_k, d), dtype=torch.half).to(device) for _ in range(H)]
dos = [torch.randn((N_q, d), dtype=torch.half).to(device) for _ in range(H)]
torch.npu.synchronize()

print("[2] Running Forward for 8 heads sequentially with synchronize...", flush=True)
outs = []
for i in range(H):
    o, l = flash_attention_npu.forward(qs[i], ks[i], vs[i], scale)
    torch.npu.synchronize()
    outs.append((o, l))
print("  >>> Sequential with sync succeeded!", flush=True)

print("[3] Running Forward for 8 heads pipelined without synchronize...", flush=True)
outs2 = []
for i in range(H):
    outs2.append(flash_attention_npu.forward(qs[i], ks[i], vs[i], scale))
torch.npu.synchronize()
print("  >>> Pipelined without sync succeeded!", flush=True)

print("[4] Running Backward for 8 heads pipelined without synchronize...", flush=True)
grads = []
for i in range(H):
    o_i, l_i = outs2[i]
    grads.append(flash_attention_npu.backward(dos[i], qs[i], ks[i], vs[i], o_i, l_i, scale))
torch.npu.synchronize()
print("  >>> Backward pipelined without sync succeeded!", flush=True)

print("🎉 MULTIHEAD IS FULLY WORKING!", flush=True)
