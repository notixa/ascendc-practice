#include <cstdint>
#include "utils/kernel_utils_macros.h"

struct FATParams {
    GM_ADDR q, k, v, mask, o, s, p, oTemp, tiling;
};

struct FATTiling {
    uint32_t N_q;
    uint32_t N_k;
    uint32_t B_q;
    uint32_t B_k;
    uint32_t h;
    uint32_t d;
    uint32_t numTask;
    float scale;
};
