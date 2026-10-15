import torch
import torch_npu
import flash_attention_npu

torch.npu.set_device(0)
torch.manual_seed(42)

print("[1] Test forward 64x64...")
q = torch.randn((64, 64), dtype=torch.half, device="npu:0")
k = torch.randn((64, 64), dtype=torch.half, device="npu:0")
v = torch.randn((64, 64), dtype=torch.half, device="npu:0")
o, l = flash_attention_npu.forward(q, k, v)
torch.npu.synchronize()
print("   Passed!")

print("[2] Test forward 64x128...")
q2 = torch.randn((64, 64), dtype=torch.half, device="npu:0")
k2 = torch.randn((128, 64), dtype=torch.half, device="npu:0")
v2 = torch.randn((128, 64), dtype=torch.half, device="npu:0")
o2, l2 = flash_attention_npu.forward(q2, k2, v2)
torch.npu.synchronize()
print("   Passed!")

print("[3] Test backward 64x64...")
do = torch.randn((64, 64), dtype=torch.half, device="npu:0")
dq, dk, dv = flash_attention_npu.backward(do, q, k, v, o, l)
torch.npu.synchronize()
print("   Passed!")

print("[4] Test backward 64x128...")
do2 = torch.randn((64, 64), dtype=torch.half, device="npu:0")
dq2, dk2, dv2 = flash_attention_npu.backward(do2, q2, k2, v2, o2, l2)
torch.npu.synchronize()
print("   Passed!")

print("🎉 ALL 4 STEPS PASSED!")
