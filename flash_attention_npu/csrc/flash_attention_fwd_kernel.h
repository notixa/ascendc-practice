#pragma once
#include "flash_attention_fwd_tiling.h"
#include <cstdint>

// CANN ascendc_library 在 host_stub.cpp 中自动生成的底层内核发射符号
extern "C" {
uint32_t aclrtlaunch_flash_attention_fwd_kernel(
    uint32_t blockDim, void *stream,
    void *q, void *k, void *v, void *o, void *l, void *tiling);
}

// 供 Host / PyTorch 调用的内联封装函数
inline void FlashAttentionFwdKernel_do(
    uint32_t blockDim, void *stream,
    void *q, void *k, void *v, void *o, void *l,
    const FlashAttentionFwdTilingData &tiling) {
    aclrtlaunch_flash_attention_fwd_kernel(
        blockDim, stream, q, k, v, o, l,
        const_cast<FlashAttentionFwdTilingData *>(&tiling));
}
