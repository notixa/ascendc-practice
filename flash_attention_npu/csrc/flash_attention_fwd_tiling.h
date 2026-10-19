#ifndef FLASH_ATTENTION_FWD_TILING_H
#define FLASH_ATTENTION_FWD_TILING_H

#include <cstdint>

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
#include "kernel_tiling/kernel_tiling.h"
#else
#include <math.h>

#ifndef __TIKCFW_KERNEL_TILING_H_
#define __TIKCFW_KERNEL_TILING_H_
#pragma pack(push, 8)
struct TCubeTiling {
  int32_t usedCoreNum = 1;
  int32_t M = 0;
  int32_t N = 0;
  int32_t Ka = 0;
  int32_t Kb = 0;
  int32_t singleCoreM = 0;
  int32_t singleCoreN = 0;
  int32_t singleCoreK = 0;
  int32_t baseM = 64;
  int32_t baseN = 64;
  int32_t baseK = 64;
  int32_t depthA1 = 1;
  int32_t depthB1 = 1;
  int32_t stepM = 1;
  int32_t stepN = 1;
  int32_t isBias = 0;
  int32_t transLength = 0;
  int32_t iterateOrder = 0;
  int32_t shareMode = 0;
  int32_t shareL1Size = 0;
  int32_t shareL0CSize = 0;
  int32_t shareUbSize = 0;
  int32_t batchM = 0;
  int32_t batchN = 0;
  int32_t singleBatchM = 0;
  int32_t singleBatchN = 0;
  int32_t stepKa = 1;
  int32_t stepKb = 1;
  int32_t depthAL1CacheUB = 0;
  int32_t depthBL1CacheUB = 0;
  int32_t dbL0A = 0;
  int32_t dbL0B = 0;
  int32_t dbL0C = 0;
  int32_t ALayoutInfoB = 0;
  int32_t ALayoutInfoS = 0;
  int32_t ALayoutInfoN = 0;
  int32_t ALayoutInfoG = 0;
  int32_t ALayoutInfoD = 0;
  int32_t BLayoutInfoB = 0;
  int32_t BLayoutInfoS = 0;
  int32_t BLayoutInfoN = 0;
  int32_t BLayoutInfoG = 0;
  int32_t BLayoutInfoD = 0;
  int32_t CLayoutInfoB = 0;
  int32_t CLayoutInfoS1 = 0;
  int32_t CLayoutInfoN = 0;
  int32_t CLayoutInfoG = 0;
  int32_t CLayoutInfoS2 = 0;
  int32_t BatchNum = 1;
  int32_t reserved = 0;
};
#pragma pack(pop)
#endif // __TIKCFW_KERNEL_TILING_H_
#endif // defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)

static_assert(sizeof(TCubeTiling) == 200, "TCubeTiling must be exactly 200 bytes!");

#pragma pack(push, 8)
struct FlashAttentionFwdTilingData {
  uint32_t coreNum;
  uint32_t largeCoreNum;

  TCubeTiling cubeTilingQK;
  TCubeTiling cubeTilingPV;
  uint32_t N_q;
  uint32_t N_k;
  uint32_t d;
  uint32_t B_q;
  uint32_t B_k;
  float scale;
  uint32_t batchSize;
};
#pragma pack(pop)

#if !defined(__CCE_AICORE__)
inline uint32_t AlignUp(uint32_t val, uint32_t alignment) {
  return ((val + alignment - 1) / alignment) * alignment;
}

/**
 * Host 侧计算 FlashAttention 前向分块与 TCubeTiling 参数
 */
inline void ComputeFlashAttentionFwdTiling(
    uint32_t N_q, uint32_t N_k, uint32_t d, uint32_t targetCoreNum,
    FlashAttentionFwdTilingData &tilingData, float customScale = 0.0f,
    uint32_t batchSize = 1) {
  tilingData.N_q = N_q;
  tilingData.N_k = N_k;
  tilingData.d = d;
  tilingData.batchSize = batchSize;

  // 默认 Tile 分块大小 B_q = 64, B_k = 64
  uint32_t B_q = 64;
  uint32_t B_k = 64;
  if (N_q < 64) {
    B_q = AlignUp(N_q, 16);
  }
  if (N_k < 64) {
    B_k = AlignUp(N_k, 16);
  }
  tilingData.B_q = B_q;
  tilingData.B_k = B_k;

  if (customScale > 0.0f) {
    tilingData.scale = customScale;
  } else {
    tilingData.scale = 1.0f / sqrtf(static_cast<float>(d));
  }

  uint32_t T_q = (N_q + B_q - 1) / B_q;
  uint32_t actualCoreNum = (targetCoreNum < T_q) ? targetCoreNum : T_q;
  if (actualCoreNum == 0)
    actualCoreNum = 1;

  tilingData.coreNum = actualCoreNum;
  tilingData.largeCoreNum = T_q % actualCoreNum;

  // 1. 设置 Q @ K^T 的 Matmul TCubeTiling (A: B_q x d, B: B_k x d (trans=true),
  // C: B_q x B_k)
  TCubeTiling &qk = tilingData.cubeTilingQK;
  qk.M = B_q;
  qk.N = B_k;
  qk.Ka = d;
  qk.Kb = d;
  qk.singleCoreM = B_q;
  qk.singleCoreN = B_k;
  qk.singleCoreK = d;
  qk.baseM = B_q;
  qk.baseN = B_k;
  qk.baseK = (d < 64) ? AlignUp(d, 16) : 64;
  qk.stepM = 1;
  qk.stepN = 1;
  qk.stepKa = 1;
  qk.stepKb = 1;
  qk.depthA1 = 1;
  qk.depthB1 = 1;
  qk.isBias = 0;
  qk.iterateOrder = 0;
  qk.usedCoreNum = 1;

  // 2. 设置 P @ V 的 Matmul TCubeTiling (A: B_q x B_k, B: B_k x d
  // (trans=false), C: B_q x d)
  TCubeTiling &pv = tilingData.cubeTilingPV;
  pv.M = B_q;
  pv.N = d;
  pv.Ka = B_k;
  pv.Kb = B_k;
  pv.singleCoreM = B_q;
  pv.singleCoreN = (d < 64) ? AlignUp(d, 16) : 64;
  pv.singleCoreK = B_k;
  pv.baseM = B_q;
  pv.baseN = (d < 64) ? AlignUp(d, 16) : 64;
  pv.baseK = (B_k < 64) ? AlignUp(B_k, 16) : 64;
  pv.stepM = 1;
  pv.stepN = 1;
  pv.stepKa = 1;
  pv.stepKb = 1;
  pv.depthA1 = 1;
  pv.depthB1 = 1;
  pv.isBias = 0;
  pv.iterateOrder = 0;
  pv.usedCoreNum = 1;
}
#endif // !defined(__CCE_AICORE__)

#endif // FLASH_ATTENTION_TILING_H
