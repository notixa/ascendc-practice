import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    z = torch.empty_like(x)
    n_elements = z.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    # BLOCK_SIZE: tl.constexpr 是给 Triton 编译器看的标记，运行时传的就是 int，
    # pyright 会把 constexpr 当成一个独立的类型，故只忽略这一条参数类型告警。
    add_kernel[grid](x, y, z, n_elements, BLOCK_SIZE=1024)  # pyright: ignore[reportArgumentType]
    return z
