#ifndef FLASH_ATTENTION_BWD_TILING_H
#define FLASH_ATTENTION_BWD_TILING_H

#include <cstdint>

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
#include "kernel_tiling/kernel_tiling.h"
#else
#include <algorithm>
#include <cmath>

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
};
#pragma pack(pop)

#endif
