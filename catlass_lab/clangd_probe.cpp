// @file  clangd_probe.cpp
// @brief 用来验证 clangd 能否正确解析 CATLASS 头文件（不参与真实编译）。
//         只要能跳转 / hover / 补全，这个文件就完成任务了。
#include "catlass/catlass.hpp"

#include "catlass/arch/arch.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/basic_matmul.hpp"
#include "catlass/layout/layout.hpp"

using namespace Catlass;

using ArchTag = Arch::AtlasA2;
using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<true>;
using L1TileShape = GemmShape<128, 256, 256>;
using L0TileShape = GemmShape<128, 256, 64>;

using ElementA = half;
using ElementB = half;
using ElementC = half;
using LayoutA = layout::RowMajor;
using LayoutB = layout::RowMajor;
using LayoutC = layout::RowMajor;

using AType = Gemm::GemmType<ElementA, LayoutA>;
using BType = Gemm::GemmType<ElementB, LayoutB>;
using CType = Gemm::GemmType<ElementC, LayoutC>;

using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
using BlockEpilogue = void;
using BlockScheduler = Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
using MatmulKernel = Gemm::Kernel::BasicMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;

// 触摸几个成员，确认模板实例化可被 clangd 解析
static_assert(ArchTag::L1_SIZE == 512 * 1024, "L1 size of AtlasA2");
using M = typename MatmulKernel::BlockMmad;
using N = typename MatmulKernel::ElementAccumulator;

int main()
{
    return 0;
}
