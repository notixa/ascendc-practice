#include <cstdint>
#include "utils/kernel_utils_macros.h"

struct FATParams {
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR mask;
    GM_ADDR o;
    GM_ADDR s;
    GM_ADDR p;
    GM_ADDR oTemp;
    GM_ADDR tiling;
};

struct FATTiling {
    uint32_t N_q;
    uint32_t N_k;
    uint32_t B_q;
    uint32_t B_k;
    uint32_t head_num;
    uint32_t head_dim;
    uint32_t batch_num;
    uint32_t numTask;
    float scale;
};
