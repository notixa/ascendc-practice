# pyright: reportUnreachable=false
# tl.max/tl.sum 等 @triton.jit 函数在 Python 层的 __call__ 永远 raise，
# pyright 会把调用后的代码判成 unreachable；实际执行发生在 Triton 编译器里。

import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def layer_norm_kernel(
    x_ptr,
    y_ptr,
    gamma_ptr,
    beta_ptr,
    stride_bq,
    stride_qq,
    batch_num,
    row_num,
    col_num,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    total_row_num = row_num * batch_num
    taskIdx = tl.program_id(0)
    coreNum = tl.num_programs(0)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < col_num

    gamma_ptrs = gamma_ptr + col_offsets
    beta_ptrs = beta_ptr + col_offsets
    gamma = tl.load(gamma_ptrs, mask=mask, other=0.0)
    beta = tl.load(beta_ptrs, mask=mask, other=0.0)

    for i in range(taskIdx, total_row_num, coreNum):
        b = i // row_num
        r = i % row_num
        offset = b * stride_bq + r * stride_qq
        x_start_ptr = x_ptr + offset

        x_ptrs = x_start_ptr + col_offsets
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        avg = tl.sum(x, axis=-1) / col_num
        diff = tl.where(mask, x - avg, 0.0)
        delta_square = tl.sum(diff * diff, axis=-1) / col_num
        x = diff / tl.sqrt(delta_square + eps)
        x = gamma * x + beta

        y_start_ptr = y_ptr + offset
        y_ptrs = y_start_ptr + col_offsets
        tl.store(y_ptrs, x, mask=mask)
