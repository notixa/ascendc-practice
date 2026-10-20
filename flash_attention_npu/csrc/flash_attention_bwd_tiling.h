#ifndef FLASH_ATTENTION_BWD_TILING_H
#define FLASH_ATTENTION_BWD_TILING_H

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
struct FlashAttentionBwd_dKdV_TilingData {
  uint32_t coreNum;
  uint32_t largeCoreNum;

  TCubeTiling cubeTilingQK;
  TCubeTiling cubeTilingPdO;
  TCubeTiling cubeTilingdOV;
  TCubeTiling cubeTilingdSQ;
  uint32_t N_q;
  uint32_t N_k;
  uint32_t d;
  uint32_t B_q;
  uint32_t B_k;
  float scale;
  uint32_t batchSize;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct FlashAttentionBwd_dQ_TilingData {
  uint32_t coreNum;
  uint32_t largeCoreNum;

  TCubeTiling cubeTilingQK;
  TCubeTiling cubeTilingdOV;
  TCubeTiling cubeTilingdSK;
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
inline uint32_t AlignUpBwd(uint32_t val, uint32_t alignment) {
  return ((val + alignment - 1) / alignment) * alignment;
}

inline void InitMatmulTCubeTilingBwd(TCubeTiling &t, int32_t M, int32_t N, int32_t Ka, int32_t Kb) {
  t.M = M;
  t.N = N;
  t.Ka = Ka;
  t.Kb = Kb;
  t.singleCoreM = M;
  t.singleCoreN = N;
  t.singleCoreK = Ka;
  t.baseM = M;
  t.baseN = N;
  t.baseK = Ka;
  t.stepM = 1;
  t.stepN = 1;
  t.stepKa = 1;
  t.stepKb = 1;
  t.depthA1 = 1;
  t.depthB1 = 2; // B 矩阵在 L1 开启双缓冲
  t.dbL0A = 0;
  t.dbL0B = 0;
  t.isBias = 0;
  t.iterateOrder = 0;
  t.usedCoreNum = 1;
}

inline void ComputeFlashAttentionBwd_dKdV_Tiling(
    uint32_t N_q, uint32_t N_k, uint32_t d, uint32_t targetCoreNum,
    FlashAttentionBwd_dKdV_TilingData &tilingData, float customScale = 0.0f,
    uint32_t batchSize = 1) {
  tilingData.N_q = N_q;
  tilingData.N_k = N_k;
  tilingData.d = d;
  tilingData.batchSize = batchSize;

  // 扩展 Tile 分块大小: B_q = 128 (若 N_q >= 128), B_k = 64
  uint32_t B_q = 128;
  uint32_t B_k = 64;
  if (N_q < 128) {
    B_q = AlignUpBwd(N_q, 16);
  }
  if (N_k < 64) {
    B_k = AlignUpBwd(N_k, 16);
  }
  tilingData.B_q = B_q;
  tilingData.B_k = B_k;

  if (customScale > 0.0f) {
    tilingData.scale = customScale;
  } else {
    tilingData.scale = 1.0f / sqrtf(static_cast<float>(d));
  }

  uint32_t T_k = (N_k + B_k - 1) / B_k;
  uint32_t actualCoreNum = (targetCoreNum < T_k) ? targetCoreNum : T_k;
  if (actualCoreNum == 0) actualCoreNum = 1;

  tilingData.coreNum = actualCoreNum;
  tilingData.largeCoreNum = T_k % actualCoreNum;

  InitMatmulTCubeTilingBwd(tilingData.cubeTilingQK, B_q, B_k, d, d);
  InitMatmulTCubeTilingBwd(tilingData.cubeTilingPdO, B_k, d, B_q, B_q);
  InitMatmulTCubeTilingBwd(tilingData.cubeTilingdOV, B_q, B_k, d, d);
  InitMatmulTCubeTilingBwd(tilingData.cubeTilingdSQ, B_k, d, B_q, B_q);
}

inline void ComputeFlashAttentionBwd_dQ_Tiling(
    uint32_t N_q, uint32_t N_k, uint32_t d, uint32_t targetCoreNum,
    FlashAttentionBwd_dQ_TilingData &tilingData, float customScale = 0.0f,
    uint32_t batchSize = 1) {
  tilingData.N_q = N_q;
  tilingData.N_k = N_k;
  tilingData.d = d;
  tilingData.batchSize = batchSize;

  // 扩展 Tile 分块大小: B_q = 128 (若 N_q >= 128), B_k = 64
  uint32_t B_q = 128;
  uint32_t B_k = 64;
  if (N_q < 128) {
    B_q = AlignUpBwd(N_q, 16);
  }
  if (N_k < 64) {
    B_k = AlignUpBwd(N_k, 16);
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
  if (actualCoreNum == 0) actualCoreNum = 1;

  tilingData.coreNum = actualCoreNum;
  tilingData.largeCoreNum = T_q % actualCoreNum;

  InitMatmulTCubeTilingBwd(tilingData.cubeTilingQK, B_q, B_k, d, d);
  InitMatmulTCubeTilingBwd(tilingData.cubeTilingdOV, B_q, B_k, d, d);
  InitMatmulTCubeTilingBwd(tilingData.cubeTilingdSK, B_q, d, B_k, B_k);
}
#endif

#endif

