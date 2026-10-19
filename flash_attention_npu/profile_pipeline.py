import os
import torch
import torch_npu
import flash_attention_npu

def main():
    device = "npu:0"
    N = 1024
    d = 64
    scale = 0.125

    q = torch.randn(N, d, dtype=torch.half, device=device)
    k = torch.randn(N, d, dtype=torch.half, device=device)
    v = torch.randn(N, d, dtype=torch.half, device=device)
    do = torch.randn(N, d, dtype=torch.half, device=device)
    torch.npu.synchronize()

    # Warmup 3 次规避冷启动与初始状态
    for _ in range(3):
        o, l = flash_attention_npu.forward(q, k, v, scale)
        dq, dk, dv = flash_attention_npu.backward(do, q, k, v, o, l, scale)
    torch.npu.synchronize()

    # 正式采集 5 次
    for _ in range(5):
        o, l = flash_attention_npu.forward(q, k, v, scale)
        dq, dk, dv = flash_attention_npu.backward(do, q, k, v, o, l, scale)
    torch.npu.synchronize()
    print("Profiling iterations completed successfully.")

if __name__ == "__main__":
    main()
