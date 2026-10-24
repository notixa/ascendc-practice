#include <cstdint>
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm/kernel/basic_matmul.hpp"

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/layout/matrix.hpp"
#include "catlass/status.hpp"
#include "fat_kernel.h"

#include "golden.hpp"
#include "helper.hpp"
#include "options.hpp"

using namespace Catlass;

struct Options {
    GemmOptions optionsQK;
};

static void Run(const Options& tiling)
{
    auto optionsQK = tiling.optionsQK;
    aclrtStream stream{nullptr};

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(optionsQK.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    auto mQK = optionsQK.problemShape.m();
    auto nQK = optionsQK.problemShape.n();
    auto kQK = optionsQK.problemShape.k();

    using LayoutQ = layout::RowMajor;
    using LayoutK = layout::ColumnMajor;
    using LayoutS = layout::RowMajor;
    using LayoutMask = layout::RowMajor;
    using LayoutP = layout::RowMajor;
    using LayoutV = layout::RowMajor;
    using LayoutOTemp = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    using LayoutOUpdate = layout::RowMajor;
    LayoutQ layoutQ{mQK, kQK};
    LayoutK layoutK{nQK, kQK};
    LayoutS layoutS{mQK, nQK};
    LayoutMask layoutMask{128, 128};
    LayoutP layoutP{mQK, nQK};
    LayoutV layoutV{nQK, kQK};
    LayoutOTemp layoutOTemp{mQK, kQK};
    LayoutO layoutO{mQK, kQK};
    LayoutOUpdate layoutOUpdate{mQK, 1};

    using ElementQ = half;
    using ElementK = half;
    using ElementS = float;
    using ElementMask = half;
    using ElementP = half;
    using ElementV = half;
    using ElementOTemp = float;
    using ElementO = half;
    using ElementOUpdate = float;

    using ArchTag = Arch::AtlasA2;

    using QType = Gemm::GemmType<ElementQ, LayoutQ>;
    using KType = Gemm::GemmType<ElementK, LayoutK>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using MaskType = Gemm::GemmType<ElementMask, LayoutMask>;
    using PType = Gemm::GemmType<ElementP, LayoutP>;
    using VType = Gemm::GemmType<ElementV, LayoutV>;
    using OTempType = Gemm::GemmType<ElementOTemp, LayoutOTemp>;
    using OType = Gemm::GemmType<ElementO, LayoutO>;
    using OUpdateType = Gemm::GemmType<ElementOUpdate, LayoutOUpdate>;

    using DispatchPolicyQK = Gemm::MmadAtlasA2FAQK;
    using L1TileShapeQK = GemmShape<128, 128, 256>;
    using L0TileShapeQK = GemmShape<128, 128, 64>;

    using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, QType, KType, SType>;

    using EpilogueDispatchPolicyOnlineSoftmax = Epilogue::EpilogueAtlasA2OnlineSoftmax;
    constexpr uint32_t computeLength = 16384;
    using BlockEpilogueOnlineSoftmax =
        Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicyOnlineSoftmax, PType, SType, MaskType>;

    using DispatchPolicyPV = Gemm::MmadAtlasA2FAPV;
    using L1TileShapePV = GemmShape<128, 256, 256>;
    using L0TileShapePV = GemmShape<128, 256, 64>;

    using BlockMmadPV = Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShapePV, L0TileShapePV, PType, VType, OTempType>;

    using EpilogueDispatchPolicyRescaleO = Epilogue::EpilogueAtlasA2RescaleO;
    using BlockEpilogueRescaleO =
        Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicyRescaleO, OType, OTempType, OUpdateType>;

    using FATKernel = FATKernel<BlockMmadQK, BlockMmadPV, BlockEpilogueOnlineSoftmax, BlockEpilogueRescaleO>;
}
