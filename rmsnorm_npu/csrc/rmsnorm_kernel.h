#pragma once
#include "rmsnorm_tiling.h"
#include <cstdint>
#include <type_traits>

// Non-template launch functions implemented in rmsnorm_kernel.cpp (compiled by ccec)
void RMSNorm_fp32_do(uint32_t blockDim, void *stream, void *x, void *y,
                     void *z, const RMSNormTiling &tiling);
void RMSNorm_fp16_do(uint32_t blockDim, void *stream, void *x, void *y,
                     void *z, const RMSNormTiling &tiling);
void RMSNorm_bf16_do(uint32_t blockDim, void *stream, void *x, void *y,
                     void *z, const RMSNormTiling &tiling);

// Unified template interface for PyTorch AT_DISPATCH
template <class dtype>
inline void RMSNorm_do(uint32_t blockDim, void *stream, void *x, void *y,
                       void *z, const RMSNormTiling &tiling) {
  if constexpr (std::is_same_v<dtype, float>) {
    RMSNorm_fp32_do(blockDim, stream, x, y, z, tiling);
  } else {
    // For c10::Half / half or other 16-bit floating types
    RMSNorm_fp16_do(blockDim, stream, x, y, z, tiling);
  }
}
