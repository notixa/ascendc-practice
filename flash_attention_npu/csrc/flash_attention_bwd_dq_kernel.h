#pragma once
#include "flash_attention_bwd_tiling.h"
#include <cstdint>

// CANN ascendc_library 在 host_stub.cpp 中自动生成的底层内核发射符号
extern "C" {
uint32_t aclrtlaunch_flash_attention_bwd_dq_kernel(
    uint32_t blockDim, void *stream,
    void *q, void *o, void *do_grad, void *k, void *v,
    void *l, void *d, void *dq, void *tiling);
}

// 供 Host / PyTorch 调用的内联封装函数
inline void FlashAttentionBwd_dQ_Kernel_do(
    uint32_t blockDim, void *stream,
    void *q, void *o, void *do_grad, void *k, void *v,
    void *l, void *d, void *dq,
    const FlashAttentionBwd_dQ_TilingData &tiling) {
    aclrtlaunch_flash_attention_bwd_dq_kernel(
        blockDim, stream,
        q, o, do_grad, k, v, l, d, dq,
        const_cast<FlashAttentionBwd_dQ_TilingData *>(&tiling));
}
