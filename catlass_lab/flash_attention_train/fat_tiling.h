/**
 * fat_tiling.h —— host/device **共用**的运行时数据结构（纯 POD，无 CANN 类型）
 *
 * ⚠️ 这个头会被 **gcc**（catlass_op/src/catlass_torch.cpp）和 **bisheng**（.asc）同时编译，
 *    所以里面**绝对不能**出现 GM_ADDR / __gm__ / GlobalTensor / half 等设备侧类型。
 *    设备侧的 FATParams 在 fat_params.h（只被 .asc 看到）。
 */

#pragma once

#include <cstdint>
#include <cstddef>

struct FATTiling {
    uint32_t N_q; // 每个 batch 的 q 序列长度（局部行数）
    uint32_t N_k; // 每个 batch 的 kv 序列长度
    uint32_t B_q; // q 分块大小（须与 fat_types.h 的 L1TileShapeQK::M 一致）
    uint32_t B_k; // kv 分块大小（须与 L1TileShapeQK::N 一致）
    uint32_t head_num;
    uint32_t head_dim; // 本 kernel 要求 128
    uint32_t batch_num;
    uint32_t numTask; // batch_num * ceil(N_q/B_q) * head_num
    float scale;      // 1/sqrt(head_dim)
    uint32_t causal;
};

// ---------------------------------------------------------------------------
// workspace 的几何常量 —— host 和 device 必须用【同一份】公式，否则就是踩内存
//
// 形状（每 core、每流水槽）：
//   S    : [B_q, B_k * STACK_NUM]  fp32   ← QK 写 / softmax 读
//   P    : [B_q, B_k * STACK_NUM]  fp16   ← softmax 写 / PV 读
//   oTmp : [B_q, head_dim]         fp32   ← PV 写 / rescale 读
//
// ⚠️ STACK_NUM 必须与 FATKernel 里的 STACK_NUM 一致；
//    NUM_SLOTS 是流水槽数（preLaunch + 1），M1 阶段不做流水 = 1。
// ---------------------------------------------------------------------------
namespace fat_ws {

constexpr uint32_t STACK_NUM = 4;
constexpr uint32_t NUM_SLOTS = 1;

inline size_t ElemsS(uint32_t coreNum, uint32_t B_q, uint32_t B_k)
{
    return static_cast<size_t>(NUM_SLOTS) * coreNum * B_q * B_k * STACK_NUM;
}
inline size_t ElemsP(uint32_t coreNum, uint32_t B_q, uint32_t B_k)
{
    return ElemsS(coreNum, B_q, B_k);
}
inline size_t ElemsOt(uint32_t coreNum, uint32_t B_q, uint32_t headDim)
{
    return static_cast<size_t>(NUM_SLOTS) * coreNum * B_q * headDim;
}

inline size_t BytesS(uint32_t coreNum, uint32_t B_q, uint32_t B_k)
{
    return ElemsS(coreNum, B_q, B_k) * sizeof(float);
}
inline size_t BytesP(uint32_t coreNum, uint32_t B_q, uint32_t B_k)
{
    return ElemsP(coreNum, B_q, B_k) * 2; // fp16：不写 sizeof(half)，避免引入设备类型
}
inline size_t BytesOt(uint32_t coreNum, uint32_t B_q, uint32_t headDim)
{
    return ElemsOt(coreNum, B_q, headDim) * sizeof(float);
}

} // namespace fat_ws
