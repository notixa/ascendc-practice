import torch
import torch_npu
import torch.nn.functional as F
import flash_attention_npu

torch.npu.set_device(0)
torch.manual_seed(42)
q = torch.randn((64, 64), dtype=torch.half, device="npu:0")
k = torch.randn((64, 64), dtype=torch.half, device="npu:0")
v = torch.randn((64, 64), dtype=torch.half, device="npu:0")
o_custom, l_custom = flash_attention_npu.forward(q, k, v)
torch.npu.synchronize()

q_f = q.cpu().float()
k_f = k.cpu().float()
v_f = v.cpu().float()
scale = 1.0 / 8.0
scores = torch.matmul(q_f, k_f.transpose(-1, -2)) * scale
p = F.softmax(scores, dim=-1)
o_golden = torch.matmul(p, v_f).half()
l_golden = torch.logsumexp(scores, dim=-1).half()

diff_o = (o_custom.cpu().float() - o_golden.float()).abs()
diff_l = (l_custom.cpu().float() - l_golden.float()).abs()

row_diff_o = diff_o.max(dim=-1).values
print("Per-row max diff in O:")
for r in range(64):
    if row_diff_o[r] > 0.02:
        print(f"Row {r:2d}: max diff = {row_diff_o[r].item():.4f}")
