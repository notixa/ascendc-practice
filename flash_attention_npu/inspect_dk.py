import os
os.environ["TE_PARALLEL_NUM"] = "1"
os.environ["TBE_PARALLEL_NUM"] = "1"
import torch
import torch_npu
import torch.nn.functional as F
import flash_attention_npu

torch.npu.set_device(0)
torch.manual_seed(42)

N_q, N_k, d = 64, 64, 64
scale = 1.0 / (d ** 0.5)

q = torch.randn((N_q, d), dtype=torch.half, device="npu:0")
k = torch.randn((N_k, d), dtype=torch.half, device="npu:0")
v = torch.randn((N_k, d), dtype=torch.half, device="npu:0")
do = torch.randn((N_q, d), dtype=torch.half, device="npu:0")

# Golden Forward & Backward on CPU
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

o_golden_npu = o_ref.half().to("npu:0")
l_golden_npu = l_ref.half().to("npu:0")

dq_c, dk_c, dv_c = flash_attention_npu.backward(
    do, q, k, v, o_golden_npu, l_golden_npu
)
torch.npu.synchronize()

diff_dk = (dk_c.cpu().float() - dk_golden.float()).abs()
row_max = diff_dk.max(dim=-1).values

print("=== Per-Row Max Diff in dK (Threshold > 0.02) ===")
err_rows = []
for r in range(N_k):
    val = row_max[r].item()
    if val > 0.02:
        err_rows.append(r)
        print(f"Row {r:2d}: max diff = {val:.5f}")

print(f"\nTotal rows with diff > 0.02: {len(err_rows)} / {N_k}")
if len(err_rows) > 0:
    first_err = err_rows[0]
    print(f"\nDetails for first error row (row {first_err}):")
    print("  Custom: ", [round(x, 5) for x in dk_c.cpu().float()[first_err, :8].tolist()])
    print("  Golden: ", [round(x, 5) for x in dk_golden.float()[first_err, :8].tolist()])
    ratio = dk_c.cpu().float()[first_err, :8] / (dk_golden.float()[first_err, :8] + 1e-7)
    print("  Ratio:  ", [round(x, 4) for x in ratio.tolist()])
