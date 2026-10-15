import torch
import torch_npu
import flash_attention_npu

torch.manual_seed(42)
q = torch.randn((64, 64), dtype=torch.half, device="npu:0")
k = torch.randn((128, 64), dtype=torch.half, device="npu:0")
v = torch.randn((128, 64), dtype=torch.half, device="npu:0")

print("Launching forward Nk=128...")
o, l = flash_attention_npu.forward(q, k, v)
torch.npu.synchronize()
print("Success! o shape:", o.shape, "l shape:", l.shape)
print("O[:2, :4]:", o[:2, :4])
