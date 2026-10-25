# pyright: reportUnreachable=false
# tl.max/tl.sum 等 @triton.jit 函数在 Python 层的 __call__ 永远 raise，
# pyright 会把调用后的代码判成 unreachable；实际执行发生在 Triton 编译器里。

import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in range(row_start, n_rows, row_step):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float("inf"))
        row_minus_max = row - tl.max(row, axis=-1)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=-1)
        softmax_output = numerator / denominator
        output_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)
