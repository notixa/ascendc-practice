/**
 * fat_params.h —— **设备侧专用**的 kernel 入参（含 GM_ADDR）
 *
 * ⚠️ 只能被 bisheng（.asc / ASC 语言）编译：
 *    GM_ADDR 展开是 `__gm__ uint8_t*`，gcc 不认识 __gm__。
 *    => host 侧（catlass_op/src/catlass_torch.cpp）**不要** include 这个头，
 *       它只需要 fat_tiling.h 里的 FATTiling。
 */

#pragma once

#include <cstdint>

#include "utils/kernel_utils_macros.h"   // GM_ADDR
#include "fat_tiling.h"                  // FATTiling

struct FATParams {
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR mask;
    GM_ADDR o;
    GM_ADDR s;
    GM_ADDR p;
    GM_ADDR oTemp;
    GM_ADDR tiling;      // 指向一块 FATTiling
};
