#pragma once
#include "softmax_tiling.h"
#include <cstdint>
#include <type_traits>

// CANN ascendc_library 自动在 host_stub.cpp 中生成的真实启动符号
extern "C" {
uint32_t aclrtlaunch_softmax_fp32_b8(uint32_t blockDim, void *stream, void *x,
                                     void *y, SoftmaxTiling *tiling);
uint32_t aclrtlaunch_softmax_fp16_b8(uint32_t blockDim, void *stream, void *x,
                                     void *y, SoftmaxTiling *tiling);
}

// 供 Host / PyTorch 调用的便捷内联包装函数
inline void SoftMax_fp32_b8_do(uint32_t blockDim, void *stream, void *x, void *y,
                               const SoftmaxTiling &tiling) {
  aclrtlaunch_softmax_fp32_b8(blockDim, stream, x, y,
                              const_cast<SoftmaxTiling *>(&tiling));
}

inline void SoftMax_fp16_b8_do(uint32_t blockDim, void *stream, void *x, void *y,
                               const SoftmaxTiling &tiling) {
  aclrtlaunch_softmax_fp16_b8(blockDim, stream, x, y,
                              const_cast<SoftmaxTiling *>(&tiling));
}

// 统一模板化接口
template <class dtype>
inline void SoftMax_b8_do(uint32_t blockDim, void *stream, void *x, void *y,
                          const SoftmaxTiling &tiling) {
  if constexpr (std::is_same_v<dtype, float>) {
    SoftMax_fp32_b8_do(blockDim, stream, x, y, tiling);
  } else {
    SoftMax_fp16_b8_do(blockDim, stream, x, y, tiling);
  }
}
