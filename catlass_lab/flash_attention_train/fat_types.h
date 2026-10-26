/**
 * fat_types.h —— FATkernel 的**编译期契约**（唯一真相）
 *
 * 这个头回答四个问题，改任何一条都要从这里改：
 *   1. 用什么 dtype / layout（Q/K/V/S/P/O/oTemp/mask）
 *   2. tile 尺寸是多少（决定 L0/L1 占用、决定 head_dim 上限）
 *   3. 哪四个 Block 组件 + 它们的 DispatchPolicy
 *   4. FatKernelType 的最终实例化
 *
 * 消费方：
 *   - catlass_op/kernel/catlass_kernel.asc   （生产 kernel，bisheng 编）
 *   - flash_attention_train/standalone/*.cpp （可选：不依赖 torch 的验证程序）
 *   - 未来的 bwd kernel
 *
 * ⚠ 边界：
 *   - 本头只能被 **ASC/bisheng** 编，不能被 gcc 编（含 Catlass/AscendC 类型）
 *     → host 侧（catlass_op/src/catlass_torch.cpp）只 #include "fat_tiling.h"
 *   - fat_tiling.h 没有 include guard，本头经 fat_kernel.h 间接包含它，
 *     同一 TU 内不要再去 include 一次
 */

#pragma once

#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"

// fat_kernel.h 内部包含 fat_tiling.h（FATParams / FATTiling）与 FATKernel 类模板
#include "fat_kernel.h"

namespace fat {

// ---------------------------------------------------------------------------
// 1. dtype / layout
// ---------------------------------------------------------------------------
//   注意：S 与 oTemp 必须是 float（epilogue 里直接 DataCopy 进 LocalTensor<float>）
//         P 必须与输入同精度（要喂给 Cube 的 P·V）
using ElementQ     = half;
using LayoutQ      = layout::RowMajor;

using ElementK     = half;
using LayoutK      = layout::ColumnMajor;   // 借列主序表达 K^T，零拷贝转置

using ElementV     = half;
using LayoutV      = layout::RowMajor;

using ElementS     = float;
using LayoutS      = layout::RowMajor;

using ElementP     = half;
using LayoutP      = layout::RowMajor;

using ElementO     = half;
using LayoutO      = layout::RowMajor;

using ElementOTmp  = float;
using LayoutOTmp   = layout::RowMajor;

using ElementMask  = half;                  // 必须 2 字节：epilogue 里字节数写死 *2
using LayoutMask   = layout::RowMajor;      // 必须 RowMajor：只取 stride(0)

using ElementOUpdate = float;               // 死参数（对应死掉的 oUpdate GM 缓冲），填合法值即可
using LayoutOUpdate  = layout::RowMajor;

using ArchTag = Catlass::Arch::AtlasA2;

// ---------------------------------------------------------------------------
// 2. GemmType 包装
// ---------------------------------------------------------------------------
using QType       = Catlass::Gemm::GemmType<ElementQ, LayoutQ>;
using KType       = Catlass::Gemm::GemmType<ElementK, LayoutK>;
using VType       = Catlass::Gemm::GemmType<ElementV, LayoutV>;
using SType       = Catlass::Gemm::GemmType<ElementS, LayoutS>;
using PType       = Catlass::Gemm::GemmType<ElementP, LayoutP>;
using OType       = Catlass::Gemm::GemmType<ElementO, LayoutO>;
using OTempType   = Catlass::Gemm::GemmType<ElementOTmp, LayoutOTmp>;
using MaskType    = Catlass::Gemm::GemmType<ElementMask, LayoutMask>;
using OUpdateType = Catlass::Gemm::GemmType<ElementOUpdate, LayoutOUpdate>;

// ---------------------------------------------------------------------------
// 3. tile 尺寸（编译期常量，决定硬件占用）
// ---------------------------------------------------------------------------
//   M = B_q, N = B_k, K = head_dim
//   全 128 时：
//     L1A_SIZE = 128*128*2 = 32KB，L1B_SIZE = 32KB
//     QK 占用 (32+32)*2 = 128KB，PV 占用 (32+32)*2 = 128KB  → 合计 256KB < 512KB
//   改大（如 256）会：(a) L1 超预算 (b) L1 的 nz 布局按 256 建、实际只填 128 → 错位
using L1TileShapeQK = Catlass::GemmShape<128, 128, 128>;
using L0TileShapeQK = Catlass::GemmShape<128, 128, 64>;
using L1TileShapePV = Catlass::GemmShape<128, 128, 128>;
using L0TileShapePV = Catlass::GemmShape<128, 128, 64>;

// ---------------------------------------------------------------------------
// 4. Block 组件（DispatchPolicy + Mmad + Epilogue，必须成对）
// ---------------------------------------------------------------------------
//   FAQK / FAPV 是"一次一个 tile、KV 循环由调用者写"的朴素变体；
//   不带 I 的版本没有 blockTable、没有 nLoop/kvSeqlen 的隐式 clamp。
using DispatchPolicyQK = Catlass::Gemm::MmadAtlasA2FAQK;
using DispatchPolicyPV = Catlass::Gemm::MmadAtlasA2FAPV;

using BlockMmadQK = Catlass::Gemm::Block::BlockMmad<
    DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, QType, KType, SType>;
using BlockMmadPV = Catlass::Gemm::Block::BlockMmad<
    DispatchPolicyPV, L1TileShapePV, L0TileShapePV, PType, VType, OTempType>;

// 配对：OnlineSoftmax（S→P，产出 dm/gl 到 UB） + RescaleO（oTemp→O，读 dm/gl）
//   换成 _FD 变体时必须同时换成 CombineScale，不能单换一个
using DispatchPolicyOnlineSoftmax = Catlass::Epilogue::EpilogueAtlasA2OnlineSoftmax;
using DispatchPolicyRescaleO      = Catlass::Epilogue::EpilogueAtlasA2RescaleO;

using BlockEpilogueOnlineSoftmax = Catlass::Epilogue::Block::BlockEpilogue<
    DispatchPolicyOnlineSoftmax, PType, SType, MaskType>;
using BlockEpilogueRescaleO = Catlass::Epilogue::Block::BlockEpilogue<
    DispatchPolicyRescaleO, OType, OTempType, OUpdateType>;

// ---------------------------------------------------------------------------
// 5. 最终实例化
// ---------------------------------------------------------------------------
using FatKernelType = FATKernel<
    BlockMmadQK, BlockMmadPV, BlockEpilogueOnlineSoftmax, BlockEpilogueRescaleO>;

} // namespace fat
