#pragma once
#include <cstdint>

struct RMSNormTiling {
  uint32_t totalRows;
  uint32_t hiddenSize;

  uint32_t reduceTBufBytes;

  uint32_t coreNum;
  uint32_t fullCoreNum;
  uint32_t rowsPerFullCore;
  uint32_t rowsPerTailCore;
};
