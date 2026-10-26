"""
test_fat.py —— torch.ops.catlass.fat 的最小验证

用法（在 910B 机器上）：
  cd catlass_op && mkdir -p build && cd build && cmake .. && make -j
  cd .. && python3 test_fat.py
"""
import math
import os

import torch
import torch_npu  # noqa: F401  （必须 import 才有 PrivateUse1 后端）

script_dir = os.path.dirname(os.path.abspath(__file__))
torch.ops.load_library(os.path.join(script_dir, "build", "libcatlass_fat.so"))


def ref_attn(q, k, v, batch, N_q, N_k, heads, head_dim, scale, causal=False):
    """CPU fp32 golden，输入 q/k/v 是 3-D [B*N, H, D]

    ⚠ 当前 kernel 没有 mask → full attention。等接上 mask 后把 causal 改成 True。
    """
    qf = q.float().reshape(batch, N_q, heads, head_dim)
    kf = k.float().reshape(batch, N_k, heads, head_dim)
    vf = v.float().reshape(batch, N_k, heads, head_dim)

    s = torch.einsum("bqhd,bkhd->bhqk", qf, kf) * scale          # [B,H,N_q,N_k]
    if causal:
        mask = torch.triu(torch.ones(N_q, N_k, dtype=torch.bool), diagonal=1)
        s = s.masked_fill(mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    o = torch.einsum("bhqk,bkhd->bqhd", p, vf)
    return o.reshape(batch * N_q, heads, head_dim)


def run_case(batch, N_q, N_k, heads, head_dim, block_dim=0, use_4d=False):
    scale = 1.0 / math.sqrt(head_dim)
    dev = "npu"

    torch.manual_seed(0)
    if use_4d:
        q = torch.randn(batch, N_q, heads, head_dim, dtype=torch.float16).to(dev)
        k = torch.randn(batch, N_k, heads, head_dim, dtype=torch.float16).to(dev)
        v = torch.randn(batch, N_k, heads, head_dim, dtype=torch.float16).to(dev)
        batch_arg = 0          # 0 = 从 q.size(0) 推
    else:
        q = torch.randn(batch * N_q, heads, head_dim, dtype=torch.float16).to(dev)
        k = torch.randn(batch * N_k, heads, head_dim, dtype=torch.float16).to(dev)
        v = torch.randn(batch * N_k, heads, head_dim, dtype=torch.float16).to(dev)
        batch_arg = batch

    o = torch.ops.catlass.fat(q, k, v, scale, batch_arg, block_dim)
    torch.npu.synchronize()

    assert o.shape == q.shape, f"shape mismatch: got {o.shape}, expect {q.shape}"

    ref = ref_attn(q.cpu(), k.cpu(), v.cpu(), batch, N_q, N_k, heads, head_dim, scale)
    ref = ref.reshape(o.shape)

    diff = (o.cpu().float() - ref).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    layout = "4-D" if use_4d else "3-D"
    print(f"[{layout}] B={batch} N_q={N_q} N_k={N_k} H={heads} D={head_dim} "
          f"blockDim={block_dim or 'auto'}  max_err={max_err:.5f} mean_err={mean_err:.6f}")
    return max_err


if __name__ == "__main__":
    print(f"device: {torch.npu.get_device_name(0)}")

    cases = [
        # (batch, N_q, N_k, heads, head_dim, block_dim, use_4d)
        (1, 128, 128, 1, 128, 0, False),   # 单任务、单 head、一个 q block
        (1, 128, 128, 1, 128, 0, True),    # 同上，但 4-D 输入
        (1, 128, 128, 2, 128, 0, False),   # 多 head
        (1, 256, 256, 1, 128, 0, False),   # 两个 q block（前缀块 + 对角块）
        (1, 256, 256, 2, 128, 0, True),    # 组合 + 4-D
        (2, 256, 256, 2, 128, 0, False),   # 多 batch（3-D 合并维）
        (2, 256, 256, 2, 128, 0, True),    # 多 batch + 4-D
    ]
    results = [run_case(*c) for c in cases]

    worst = max(results)
    print("=" * 60)
    print(f"worst max_err = {worst:.5f}")
    assert worst < 2e-2, f"precision failed: {worst}"
    print("PASS")
