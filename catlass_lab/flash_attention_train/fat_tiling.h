/**
 * fat_tiling.h —— host/device **共用**的运行时数据结构（纯 POD，无 CANN 类型）
 *
 * ⚠️ 这个头会被 **gcc**（catlass_op/src/catlass_torch.cpp）和 **bisheng**（.asc）同时编译，
 *    所以里面**绝对不能**出现 GM_ADDR / __gm__ / GlobalTensor / half 等设备侧类型。
 *    设备侧的 FATParams 在 fat_params.h（只被 .asc 看到）。
 */

#pragma once

#include <cstdint>

struct FATTiling {
    uint32_t N_q;        // 每个 batch 的 q 序列长度（局部行数）
    uint32_t N_k;        // 每个 batch 的 kv 序列长度
    uint32_t B_q;        // q 分块大小（须与 fat_types.h 的 L1TileShapeQK::M 一致）
    uint32_t B_k;        // kv 分块大小（须与 L1TileShapeQK::N 一致）
    uint32_t head_num;
    uint32_t head_dim;   // 本 kernel 要求 128
    uint32_t batch_num;
    uint32_t numTask;    // batch_num * ceil(N_q/B_q) * head_num
    float    scale;      // 1/sqrt(head_dim)
};
