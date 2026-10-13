#pragma once
#include <cstdint>

struct SoftmaxTiling {
  uint32_t largeCoreNum;

  uint32_t rowsPerLargeCore;
  uint32_t rowsPerTailCore;

  uint32_t hiddenSize;
};
