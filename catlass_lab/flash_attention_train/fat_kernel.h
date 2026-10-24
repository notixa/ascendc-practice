#include <algorithm>
#include <cstdint>
#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/detail/macros.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"

#include "dav_c220/core_mng/roc/kernel_operator_group_barrier_impl.h"
#include "fat_tiling.h"
#include "kernel_event.h"
#include "kernel_macros.h"
#include "kernel_operator.h"
#include "kernel_operator_block_sync_intf.h"
#include "kernel_operator_common_intf_impl.h"
#include "kernel_tensor.h"
#include "stub_def.h"
#include "stub_fun.h"
using namespace Catlass;

template <class BlockMmadQK_, class BlockMmadPV_, class BlockEpilogueOnlineSoftmax_, class BlockEpilogueRescaleO_>
class FATKernel {
public:
    using ArchTag = typename BlockMmadQK_::ArchTag;
    using L1TileShape = typename BlockMmadQK_::L1TileShape;

    using ElementQ = typename BlockMmadQK_::ElementA;
    using LayoutQ = typename BlockMmadQK_::LayoutA;
    using ElementK = typename BlockMmadQK_::ElementB;
    using LayoutK = typename BlockMmadQK_::LayoutB;
    using ElementS = typename BlockMmadQK_::ElementC;
    using LayoutS = typename BlockMmadQK_::LayoutC;

    using ElementP = typename BlockMmadPV_::ElementA;
    using LayoutP = typename BlockMmadPV_::LayoutA;
    using ElementV = typename BlockMmadPV_::ElementB;
    using LayoutV = typename BlockMmadPV_::LayoutB;
    using ElementOTmp = typename BlockMmadPV_::ElementC;
    using LayoutOTmp = typename BlockMmadPV_::LayoutC;

    using ElementMask = typename BlockEpilogueOnlineSoftmax_::ElementMask;
    using LayoutMask = typename BlockEpilogueOnlineSoftmax_::LayoutMask;

    using ElementO = typename BlockEpilogueRescaleO_::ElementOutput;
    using LayoutO = typename BlockEpilogueRescaleO_::LayoutOutput;

    static constexpr uint32_t QK_READY_ID = 1;
    static constexpr uint32_t SM_READY_ID = 2;
    static constexpr uint32_t PV_READY_ID = 3;

    CATLASS_DEVICE
    FATKernel()
    {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(const FATParams& params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(const FATParams& params)
    {
        static constexpr uint32_t L1_QK_SIZE =
            BlockMmadQK_::L1TileShape::M * BlockMmadQK_::L1TileShape::K * 2 * sizeof(ElementQ) +
            BlockMmadQK_::L1TileShape::K * BlockMmadQK_::L1TileShape::N * 2 * sizeof(ElementK);
        BlockMmadQK_ mmadQK(resource);
        BlockMmadPV_ mmadPV(resource, L1_QK_SIZE);

        __gm__ FATTiling* t = reinterpret_cast<__gm__ FATTiling*>(params.tiling);
        uint32_t headNum = t->h;
        uint32_t dPerHead = t->d;
        uint32_t B_q = t->B_q;
        uint32_t N_q = t->N_q;
        uint32_t B_k = t->B_k;
        uint32_t N_k = t->N_k;
        uint32_t numTask = t->numTask;

        GlobalTensor<ElementQ> qGm;
        GlobalTensor<ElementK> kGm;
        GlobalTensor<ElementS> sGm;
        GlobalTensor<ElementP> pGm;
        GlobalTensor<ElementV> vGm;
        GlobalTensor<ElementOTmp> oTmpGm;
        qGm.SetGlobalBuffer((__gm__ ElementQ*)params.q);
        kGm.SetGlobalBuffer((__gm__ ElementK*)params.k);
        sGm.SetGlobalBuffer((__gm__ ElementS*)params.s);
        pGm.SetGlobalBuffer((__gm__ ElementP*)params.p);
        vGm.SetGlobalBuffer((__gm__ ElementV*)params.v);
        oTmpGm.SetGlobalBuffer((__gm__ ElementOTmp*)params.oTemp);
        const uint32_t strideQ_OTmp = headNum * dPerHead;
        const uint32_t strideK_V = headNum * dPerHead;
        const uint32_t strideS_P = B_k;

        LayoutQ layoutQ{B_q, dPerHead};
        LayoutK layoutK{B_k, dPerHead};
        LayoutS layoutS{B_q, B_k};
        LayoutP layoutP{B_q, B_k};
        LayoutV layoutV{B_k, dPerHead};
        LayoutOTmp layoutOTmp{B_q, dPerHead};

        for (uint32_t taskIdx = AscendC::GetBlockIdx(); taskIdx < numTask; taskIdx += AscendC::GetBlockNum()) {
            TaskInfo taskInfo = DecodeTask();

            uint32_t pingpongFlag = 0;
            for (uint32_t j = 0; j <= taskInfo.nDiag; ++j) {
                uint32_t nIdx = j;
                uint32_t kRowNum = std::min(B_k, N_k - j * B_k);
                uint32_t kvOffset =
                    taskInfo.b * N_k * headNum * dPerHead + nIdx * B_k * headNum * dPerHead + taskInfo.h * dPerHead;

                GemmCoord actualShapeQK{taskInfo.qRowNum, kRowNum, dPerHead};
                mmadQK(
                    qGm[taskInfo.qOffset], kGm[kvOffset], sGm, layoutQ, layoutK, layoutS, actualShapeQK, pingpongFlag,
                    nIdx == 0);
                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);

                GemmCoord actualShapePV{taskInfo.qRowNum, dPerHead, kRowNum};
                mmadPV(pGm, vGm[kvOffset], oTmpGm, layoutP, layoutV, layoutOTmp, actualShapePV, pingpongFlag, smReady);
                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);
            }
        }
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(const FATParams& params)
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);

        __gm__ FATTiling* t = reinterpret_cast<__gm__ FATTiling*>(params.tiling);
        float scaleValue = t->scale;
        uint32_t N_q = t->N_q;
        uint32_t N_k = t->N_k;
        uint32_t B_q = t->B_q;
        uint32_t B_k = t->B_k;
        uint32_t headNum = t->h;
        uint32_t dPerHead = t->d;

        BlockEpilogueOnlineSoftmax_ onlineSoftmax(resource, scaleValue);
        BlockEpilogueRescaleO_ rescaleO(resource);

        GlobalTensor<ElementS> sGm;
        GlobalTensor<ElementP> pGm;
        GlobalTensor<ElementOTmp> oTmpGm;
        GlobalTensor<ElementO> oGm;
        sGm.SetGlobalBuffer((__gm__ ElementS*)params.s);
        pGm.SetGlobalBuffer((__gm__ ElementP*)params.p);
        oTmpGm.SetGlobalBuffer((__gm__ ElementOTmp*)params.oTemp);
        oGm.SetGlobalBuffer((__gm__ ElementO*)params.o);

        LayoutS layoutS{B_q, B_k};
        LayoutP layoutP{B_q, B_k};
        LayoutO layoutO{B_q, dPerHead * headNum};
        LayoutOTmp layoutOTmp{B_q, dPerHead};

        for (uint32_t taskIdx = AscendC::GetBlockIdx(); taskIdx < t->numTask; taskIdx += AscendC::GetBlockNum()) {
            TaskInfo taskInfo = DecodeTask();

            for (uint32_t j = 0; j <= taskInfo.nDiag; ++j) {
                uint32_t kRowNum = std::min(B_k, N_k - j * B_k);
                uint32_t oOffset = taskInfo.b * N_q * headNum * dPerHead + taskInfo.i * B_q * headNum * dPerHead +
                                   taskInfo.h * dPerHead;
                Arch::CrossCoreWaitFlag(qkReady);
                GemmCoord onlineSoftmaxShape{taskInfo.qRowNum, kRowNum, dPerHead};
                onlineSoftmax(
                    pGm, sGm, layoutP, layoutS, onlineSoftmaxShape, (j == 0) ? 1u : 0u, taskInfo.qRowNum, 1u, 0u);
                Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(smReady);
                Arch::CrossCoreWaitFlag(pvReady);
                GemmCoord rescaleOShape{taskInfo.qRowNum, dPerHead, kRowNum};
                rescaleO(
                    oGm[oOffset], oTmpGm, layoutO, layoutOTmp, rescaleOShape, taskInfo.qRowNum, 1u, (j == 0) ? 1u : 0u,
                    j == taskInfo.nDiag, 0u);
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
    }

private:
    struct TaskInfo {
        uint32_t b;
        uint32_t i;
        uint32_t h;
        uint32_t nDiag; // 全通前缀
        uint32_t qRowNum;
        uint32_t qOffset;
    };
    TaskInfo DecodeTask();
    Arch::Resource<ArchTag> resource;
    Arch::CrossCoreFlag qkReady{QK_READY_ID};
    Arch::CrossCoreFlag smReady{SM_READY_ID};
    Arch::CrossCoreFlag pvReady{PV_READY_ID};
};
