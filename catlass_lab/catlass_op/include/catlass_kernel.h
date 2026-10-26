/**
 * catlass_kernel.h —— device 侧 wrapper 的声明（给 src/catlass_torch.cpp 用）
 *
 * 注意：这个头必须能被 gcc 编译，所以只能出现 C++ 基础类型，
 *       不能出现 __gm__ / GlobalTensor / Catlass 的任何类型。
 */
#ifndef CATLASS_FAT_KERNEL_H_
#define CATLASS_FAT_KERNEL_H_

#include <cstddef>
#include <cstdint>

namespace catlass_kernel {

// 启动一次 FAT（FlashAttention Training/prefill）forward。
//
//   coreNum : 使用多少个 AIC 核（= blockDim）
//   stream  : 当前 NPU stream（void* 而不是 aclrtStream，避免 ASC 类型解析问题）
//   其余    : device 指针，语义见 fat_tiling.h 的 FATParams
//
// workspace 三块的大小必须与 kernel 里 layoutS / layoutP / layoutOTmp 的 stride 一致：
//   s      : coreNum * B_q * B_k     float
//   p      : coreNum * B_q * B_k     half
//   oTemp  : coreNum * B_q * headDim  float
void fat(uint32_t coreNum, void* stream,
         uint8_t* q, uint8_t* k, uint8_t* v, uint8_t* mask, uint8_t* o,
         uint8_t* s, uint8_t* p, uint8_t* oTemp, uint8_t* tiling);

// 查 workspace 需求（单位：字节）。
// FA 的 kernel 没有 GetWorkspaceSize，所以这里用和 kernel 相同的公式自己算，
// 让 host 侧只需要调这一个函数就能拿到总大小。
size_t get_fat_workspace_size(uint32_t coreNum, uint32_t B_q, uint32_t B_k, uint32_t headDim);

} // namespace catlass_kernel

#endif // CATLASS_FAT_KERNEL_H_
