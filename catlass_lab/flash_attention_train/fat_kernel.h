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

#include "fat_tiling.h"
#include "fat_params.h" // FATParams（设备侧：含 GM_ADDR），host 侧不要引
#include "kernel_operator.h"
// ── 临时诊断：-DFAT_TRACE 才生效，定完位就删 ──
#if defined(FAT_TRACE)
// 极简：单字符 + 不换行，避免把设备端 print buffer 打满
#define FATT(msg) do { if (AscendC::GetBlockIdx() == 0) { AscendC::PRINTF(msg); } } while (0)
#define FATT1(msg, a) do { if (AscendC::GetBlockIdx() == 0) { AscendC::PRINTF(msg); } } while (0)
#else
#define FATT(msg) do { } while (0)
#define FATT1(msg, a) do { } while (0)
#endif

#include "kernel_operator_common_intf_impl.h"
using namespace Catlass;

template <
    class BlockMmadQK_, class BlockMmadPV_, class BlockMmadQKTail_, class BlockMmadPVTail_,
    class BlockEpilogueOnlineSoftmax_, class BlockEpilogueRescaleO_>
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

    static constexpr uint32_t STACK_NUM = 4;

    CATLASS_DEVICE
    FATKernel()
    {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(const FATParams& params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(const FATParams& params)
    {
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);

        static constexpr uint32_t L1_QK_SIZE =
            BlockMmadQK_::L1TileShape::M * BlockMmadQK_::L1TileShape::K * sizeof(ElementQ) +
            BlockMmadQK_::L1TileShape::K * BlockMmadQK_::L1TileShape::N * 2 * sizeof(ElementK);
        BlockMmadQK_ mmadQK(resource);
        BlockMmadPV_ mmadPV(resource, L1_QK_SIZE);
        BlockMmadQKTail_ mmadQKTail(resource);
        BlockMmadPVTail_ mmadPVTail(resource, L1_QK_SIZE);

        __gm__ FATTiling* t = reinterpret_cast<__gm__ FATTiling*>(params.tiling);
        uint32_t headNum = t->head_num;
        uint32_t headDim = t->head_dim;
        uint32_t B_q = t->B_q;
        uint32_t N_q = t->N_q;
        uint32_t B_k = t->B_k;
        uint32_t N_k = t->N_k;
        uint32_t numTask = t->numTask;
        uint32_t batchNum = t->batch_num;

        uint32_t delta_qk = N_k - N_q;

        GlobalTensor<ElementQ> qGm;
        GlobalTensor<ElementK> kGm;
        GlobalTensor<ElementS> sGm;
        GlobalTensor<ElementP> pGm;
        GlobalTensor<ElementV> vGm;
        GlobalTensor<ElementOTmp> oTmpGm;
        GlobalTensor<int32_t> dummyBlockTable;
        qGm.SetGlobalBuffer((__gm__ ElementQ*)params.q);
        kGm.SetGlobalBuffer((__gm__ ElementK*)params.k);
        sGm.SetGlobalBuffer((__gm__ ElementS*)params.s);
        pGm.SetGlobalBuffer((__gm__ ElementP*)params.p);
        vGm.SetGlobalBuffer((__gm__ ElementV*)params.v);
        oTmpGm.SetGlobalBuffer((__gm__ ElementOTmp*)params.oTemp);
        dummyBlockTable.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.tiling), 1);

        LayoutQ layoutQ{B_q, headDim};
        LayoutK layoutK{headNum * headDim, B_k * STACK_NUM};
        LayoutS layoutS{B_q, B_k * STACK_NUM};
        LayoutP layoutP{B_q, B_k * STACK_NUM};
        LayoutV layoutV{B_k * STACK_NUM, headDim, headDim * headNum};
        LayoutOTmp layoutOTmp{B_q, headDim};

        uint32_t spOffset = GetBlockIdx() * B_q * B_k * STACK_NUM;
        uint32_t oTmpOffset = GetBlockIdx() * B_q * headDim;

        for (uint32_t taskIdx = AscendC::GetBlockIdx(); taskIdx < numTask; taskIdx += AscendC::GetBlockNum()) {
            TaskInfo taskInfo = DecodeTask(taskIdx, N_q, N_k, B_q, B_k, headNum, headDim, delta_qk);

            uint32_t singleGroupHeads = 1;
            mmadQK.loadQGM(qGm[taskInfo.qOffset], layoutQ, taskInfo.qRowNum, singleGroupHeads, headNum);
            for (uint32_t nIdx = 0; nIdx < taskInfo.nLoop; nIdx += STACK_NUM) {
                uint32_t kRowNum = (B_k * STACK_NUM < taskInfo.visibleEnd - nIdx * B_k) ?
                                       B_k * STACK_NUM :
                                       taskInfo.visibleEnd - nIdx * B_k;
                uint32_t kvOffset =
                    taskInfo.b * N_k * headNum * headDim + nIdx * B_k * headNum * headDim + taskInfo.h * headDim;

                GemmCoord actualShapeQK{taskInfo.qRowNum, kRowNum, headDim};
                mmadQK(
                    qGm[taskInfo.qOffset], kGm[kvOffset], sGm[spOffset], dummyBlockTable, layoutQ, layoutK,
                    actualShapeQK, nIdx, taskInfo.nLoop, B_q, taskInfo.noMaskEnd, headNum * headDim);
                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);

                GemmCoord actualShapePV{taskInfo.qRowNum, headDim, kRowNum};
                mmadPV(
                    pGm[spOffset], vGm[kvOffset], oTmpGm[oTmpOffset], dummyBlockTable, layoutP, layoutV, actualShapePV,
                    nIdx, taskInfo.nLoop, B_q, taskInfo.noMaskEnd, headNum * headDim, smReady);
                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);
            }
            for (uint32_t maskedIdx = taskInfo.nLoop; maskedIdx < taskInfo.visibleKBlockNum; maskedIdx += STACK_NUM) {
                uint32_t kRowNum = (B_k * STACK_NUM < taskInfo.visibleEnd - maskedIdx * B_k) ?
                                       B_k * STACK_NUM :
                                       taskInfo.visibleEnd - maskedIdx * B_k;
                uint32_t kvOffset =
                    taskInfo.b * N_k * headNum * headDim + maskedIdx * B_k * headNum * headDim + taskInfo.h * headDim;

                GemmCoord actualShapeQK{taskInfo.qRowNum, kRowNum, headDim};
                mmadQKTail(
                    qGm[taskInfo.qOffset], kGm[kvOffset], sGm[spOffset], dummyBlockTable, layoutQ, layoutK,
                    actualShapeQK, maskedIdx, taskInfo.visibleKBlockNum, B_q, taskInfo.visibleEnd, headNum * headDim,
                    taskInfo.maskTailS, true);   // ★ preloadFlag：首块必须 true，否则 K 不预搬 → 死等
                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
                FATT("Q");

                GemmCoord actualShapePV{taskInfo.qRowNum, headDim, kRowNum};
                mmadPVTail(
                    pGm[spOffset], vGm[kvOffset], oTmpGm[oTmpOffset], dummyBlockTable, layoutP, layoutV, actualShapePV,
                    maskedIdx, taskInfo.visibleKBlockNum, B_q, taskInfo.visibleEnd, headNum * headDim, smReady,
                    taskInfo.maskTailS, true);   // ★ preloadFlag：首块必须 true，否则 V 不预搬 → 死等
                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);
                FATT("P");
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);

        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(const FATParams& params)
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);

        __gm__ FATTiling* t = reinterpret_cast<__gm__ FATTiling*>(params.tiling);
        float scaleValue = t->scale;
        uint32_t N_q = t->N_q;
        uint32_t N_k = t->N_k;
        uint32_t B_q = t->B_q;
        uint32_t B_k = t->B_k;
        uint32_t headNum = t->head_num;
        uint32_t headDim = t->head_dim;
        uint32_t batchNum = t->batch_num;

        uint32_t delta_qk = N_k - N_q;

        BlockEpilogueOnlineSoftmax_ onlineSoftmax(resource, scaleValue);
        BlockEpilogueRescaleO_ rescaleO(resource);

        GlobalTensor<ElementS> sGm;
        GlobalTensor<ElementP> pGm;
        GlobalTensor<ElementOTmp> oTmpGm;
        GlobalTensor<ElementO> oGm;
        GlobalTensor<ElementMask> maskGm;
        sGm.SetGlobalBuffer((__gm__ ElementS*)params.s);
        pGm.SetGlobalBuffer((__gm__ ElementP*)params.p);
        oTmpGm.SetGlobalBuffer((__gm__ ElementOTmp*)params.oTemp);
        oGm.SetGlobalBuffer((__gm__ ElementO*)params.o);
        maskGm.SetGlobalBuffer((__gm__ ElementMask*)params.mask);

        LayoutS layoutS{B_q, B_k * STACK_NUM};
        LayoutP layoutP{B_q, B_k * STACK_NUM};
        LayoutO layoutO{B_q, headDim * headNum, headDim * headNum};
        LayoutOTmp layoutOTmp{B_q, headDim};
        LayoutMask layoutMask{B_q, B_k, B_k};

        uint32_t spOffset = GetBlockIdx() * B_q * B_k * STACK_NUM;
        uint32_t oTmpOffset = GetBlockIdx() * B_q * headDim;

        for (uint32_t taskIdx = AscendC::GetBlockIdx(); taskIdx < t->numTask; taskIdx += AscendC::GetBlockNum()) {
            TaskInfo taskInfo = DecodeTask(taskIdx, N_q, N_k, B_q, B_k, headNum, headDim, delta_qk);
            uint32_t oOffset =
                taskInfo.b * N_q * headNum * headDim + taskInfo.i * B_q * headNum * headDim + taskInfo.h * headDim;
            for (uint32_t nIdx = 0; nIdx < taskInfo.nLoop; nIdx += STACK_NUM) {
                uint32_t kRowNum = (STACK_NUM * B_k < taskInfo.noMaskEnd - nIdx * B_k) ?
                                       B_k * STACK_NUM :
                                       taskInfo.noMaskEnd - nIdx * B_k;

                GemmCoord onlineSoftmaxShape{taskInfo.qRowNum, kRowNum, headDim};

                Arch::CrossCoreWaitFlag(qkReady);
                onlineSoftmax(
                    pGm[spOffset], sGm[spOffset], layoutP, layoutS, onlineSoftmaxShape, (nIdx == 0) ? 1u : 0u,
                    taskInfo.qRowNum, 1u, 0u);

                Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(smReady);
                Arch::CrossCoreWaitFlag(pvReady);
                GemmCoord rescaleOShape{taskInfo.qRowNum, headDim, kRowNum};
                rescaleO(
                    oGm[oOffset], oTmpGm[oTmpOffset], layoutO, layoutOTmp, rescaleOShape, taskInfo.qRowNum, 1u,
                    (nIdx == 0) ? 1u : 0u, false, 0u);
            }
            for (uint32_t maskedIdx = taskInfo.nLoop; maskedIdx < taskInfo.visibleKBlockNum; maskedIdx += STACK_NUM) {
                uint32_t kRowNum = (STACK_NUM * B_k < taskInfo.visibleEnd - maskedIdx * B_k) ?
                                       B_k * STACK_NUM :
                                       taskInfo.visibleEnd - maskedIdx * B_k;

                GemmCoord onlineSoftmaxShape{taskInfo.qRowNum, kRowNum, headDim};

                onlineSoftmax(
                    pGm[spOffset], sGm[spOffset], maskGm, layoutP, layoutS, layoutMask, onlineSoftmaxShape,
                    (maskedIdx == 0) ? 1u : 0u, taskInfo.qRowNum, 1u, 0u, qkReady);
                FATT("s");                                   // softmax 返回

                Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(smReady);
                FATT("m");                                   // smReady 已置
                Arch::CrossCoreWaitFlag(pvReady);
                FATT("v");                                   // 等到 pvReady
                GemmCoord rescaleOShape{taskInfo.qRowNum, headDim, kRowNum};
                rescaleO(
                    oGm[oOffset], oTmpGm[oTmpOffset], layoutO, layoutOTmp, rescaleOShape, taskInfo.qRowNum, 1u,
                    (maskedIdx == 0) ? 1u : 0u, maskedIdx + STACK_NUM >= taskInfo.visibleKBlockNum, 0u);
                FATT("r");                                   // rescale 完成
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
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
        uint32_t qRowNum;
        uint32_t qOffset;
        uint32_t nLoop;
        uint32_t noMaskEnd;
        uint32_t visibleEnd;
        uint32_t visibleKBlockNum;
        uint32_t maskTailS;
    };
    Arch::Resource<ArchTag> resource;
    Arch::CrossCoreFlag qkReady{QK_READY_ID};
    Arch::CrossCoreFlag smReady{SM_READY_ID};
    Arch::CrossCoreFlag pvReady{PV_READY_ID};

private:
    TaskInfo DecodeTask(
        uint32_t taskIdx, uint32_t N_q, uint32_t N_k, uint32_t B_q, uint32_t B_k, uint32_t headNum, uint32_t headDim,
        uint32_t delta_qk)
    {
        TaskInfo info;
        uint32_t blocksPerBatch = (N_q + B_q - 1) / B_q;
        info.b = taskIdx / (blocksPerBatch * headNum);
        info.i = (taskIdx % (blocksPerBatch * headNum)) / headNum;
        info.h = (taskIdx % (blocksPerBatch * headNum)) % headNum;
        info.qRowNum = (B_q < N_q - info.i * B_q) ? B_q : N_q - info.i * B_q;
        info.qOffset = info.b * N_q * headNum * headDim + info.i * B_q * headDim * headNum + info.h * headDim;
        info.visibleEnd = info.i * B_q + info.qRowNum + delta_qk;
        info.noMaskEnd = info.visibleEnd - B_q;
        info.maskTailS = info.noMaskEnd % B_k;
        info.nLoop = (info.noMaskEnd + B_k - 1) / B_k;
        info.visibleKBlockNum = (info.visibleEnd + B_k - 1) / B_k;
        return info;
    }
};
