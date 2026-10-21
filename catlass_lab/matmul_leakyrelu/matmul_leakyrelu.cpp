#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "tile_elemwise_leakyrelu.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/gemm/kernel/matmul_activation.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"

#include "golden.hpp"
#include "helper.hpp"

using namespace Catlass;

using Options = GemmOptions;

static void Run(const Options& options)
{
    aclrtStream stream{nullptr};

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();

    using ElementA = half;
    using ElementB = half;
    using ElementC = float;

    size_t lenA = static_cast<size_t>(m) * k;
    size_t lenB = static_cast<size_t>(k) * n;
    size_t lenC = static_cast<size_t>(m) * n;

    size_t sizeA = lenA * sizeof(ElementA);
    size_t sizeB = lenB * sizeof(ElementB);
    size_t sizeD = lenC * sizeof(half);

    using LayoutA = layout::RowMajor;
    using LayoutB = layout::RowMajor;
    using LayoutC = layout::RowMajor;
    LayoutA layoutA = LayoutA::template MakeLayout<ElementA>(m, k);
    LayoutB layoutB = LayoutB::template MakeLayout<ElementB>(k, n);
    LayoutC layoutC{m, n};

    std::vector<fp16_t> hostA(lenA);
    std::vector<fp16_t> hostB(lenB);
    golden::FillRandomData<fp16_t>(hostA, -5.0f, 5.0f);
    golden::FillRandomData<fp16_t>(hostB, -5.0f, 5.0f);

    uint8_t* deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceC), sizeD, ACL_MEM_MALLOC_HUGE_FIRST));

    // Prepare hardware sync address
    uint64_t hardwareSyncAddr{0};
    ACL_CHECK(aclrtGetHardwareSyncAddr(reinterpret_cast<void**>(&hardwareSyncAddr)));

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    // Mmad
    using ArchTag = Arch::AtlasA2;
    using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<true>;
    using L1TileShape = GemmShape<128, 256, 256>;
    using L0TileShape = GemmShape<128, 256, 64>;
    using AType = Gemm::GemmType<ElementA, LayoutA>;
    using BType = Gemm::GemmType<ElementB, LayoutB>;
    using CType = Gemm::GemmType<ElementC, LayoutC>;
    using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;

    // Epilogue
    using EpilogueDispatchPolicy = Epilogue::EpilogueAtlasA2ElemWiseNoSource;
    using DType = Gemm::GemmType<half, LayoutC>;
    using ComputeType = CType;
    constexpr uint32_t computeLength = 64 * 256;
    using TileElemWiseEpilogue = Epilogue::Tile::TileElemWiseLeakyRelu<ArchTag, ComputeType, computeLength, 1>;
    using EpilogueTileCopy = Epilogue::Tile::TileCopy<ArchTag, CType, DType>;
    using BlockEpilogue =
        Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy, CType, DType, TileElemWiseEpilogue, EpilogueTileCopy>;
    if (m > n) {
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
        using MatmulKernel = Gemm::Kernel::MatmulActivation<BlockMmad, BlockEpilogue, BlockScheduler>;
        typename MatmulKernel::Arguments arguments{options.problemShape, sizeof(float), deviceA, deviceB, deviceC};
        using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
        MatmulAdapter matmulOp;
        RunAdapter(matmulOp, arguments, stream, aicCoreNum, hardwareSyncAddr);
    } else {
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
        using MatmulKernel = Gemm::Kernel::MatmulActivation<BlockMmad, BlockEpilogue, BlockScheduler>;
        typename MatmulKernel::Arguments arguments{options.problemShape, sizeof(float), deviceA, deviceB, deviceC};
        using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
        MatmulAdapter matmulOp;
        RunAdapter(matmulOp, arguments, stream, aicCoreNum, hardwareSyncAddr);
    }

    std::vector<fp16_t> hostC(lenC);
    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeD, deviceC, sizeD, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> hostGolden(lenC);
    golden::ComputeMatmulElemWiseLeakyRelu(
        options.problemShape, hostA, layoutA, hostB, layoutB, hostGolden, layoutC, 0.01f);

    std::vector<uint64_t> errorIndices = golden::CompareData(hostC, hostGolden, k);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceC));

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

int main(int argc, const char** argv)
{
    Options options;
    if (options.Parse(argc, argv) != 0) {
        return -1;
    }
    Run(options);
    return 0;
}
