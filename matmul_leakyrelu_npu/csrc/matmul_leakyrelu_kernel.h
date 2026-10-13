#pragma once
#include "matmul_leakyrelu_tiling.h"
#include <cstdint>

// CANN ascendc_library 在 host_stub.cpp 中自动生成的底层内核发射符号
extern "C" {
uint32_t aclrtlaunch_matmul_leakyrelu_custom_fp16(
    uint32_t blockDim, void *stream,
    void *a, void *b, void *bias, void *c, void *tiling);
}

// 供 Host / PyTorch 调用的内联封装函数
inline void Matmul_leakyrelu_custom_fp16_do(
    uint32_t blockDim, void *stream,
    void *a, void *b, void *bias, void *c,
    const MatmulLeakyReluTilingData &tiling) {
    aclrtlaunch_matmul_leakyrelu_custom_fp16(
        blockDim, stream, a, b, bias, c,
        const_cast<MatmulLeakyReluTilingData *>(&tiling));
}
