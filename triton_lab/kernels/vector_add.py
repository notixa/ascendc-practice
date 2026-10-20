"""最小可跑 kernel：验证 Triton-Ascend 工具链真的通了。

用法：
    uv run python kernels/vector_add.py           # 自动选设备（npu 优先）
    uv run python kernels/vector_add.py --cpu     # 只在 CPU tensor 上跑（仅验证语言层）

注意：
    - 这里**没有** num_warps / num_stages —— 那是 GPU 专有参数，NPU 上不生效（见 README 第 3 节）。
    - grid 由 program 数决定；核数不是 kernel 参数。
"""

import sys

import triton
import triton.language as tl


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def main():
    try:
        import torch
    except ModuleNotFoundError:
        print("没装 torch。本工程只装了 triton-ascend（它不强制带 torch）。")
        print("在有卡的机器上：")
        print("  uv pip install torch --index-url https://download.pytorch.org/whl/cpu   # 先装 CPU 版 torch")
        print("  uv pip install torch_npu==<与 CANN 配套的版本>                          # 再装 NPU 插件")
        return 2

    use_npu = hasattr(torch, "npu") and torch.npu.is_available() and "--cpu" not in sys.argv
    device = "npu" if use_npu else "cpu"
    print(f"device = {device}")

    n = 2 ** 20
    x = torch.rand(n, dtype=torch.float32, device=device)
    y = torch.rand(n, dtype=torch.float32, device=device)
    out = torch.empty_like(x)

    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK)

    if device == "npu":
        torch.npu.synchronize()

    ok = torch.allclose(out, x + y, atol=1e-5, rtol=1e-5)
    print("max abs err =", (out - (x + y)).abs().max().item())
    print("vector add:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
