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

#include "fat_params.h"   // FATParams（含 GM_ADDR）+ FATTiling
#include "kernel_operator.h"
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

    // 软件流水：AIV 领先 AIC 的 KV 块数（照 examples/23 的 preLaunch）
    static constexpr uint32_t PIPE_DEPTH = 2;
    static constexpr uint32_t NUM_SLOTS = PIPE_DEPTH + 1;   // S/P/oTmp 的 pingpong 槽数

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
        // [TRACE] 调完记得删
        if (AscendC::GetBlockIdx() == 0) { AscendC::PRINTF("[AIC] enter\n"); }

        __gm__ FATTiling* t = reinterpret_cast<__gm__ FATTiling*>(params.tiling);
        if (t->dbg_sm_id != 0) { smReady.id = static_cast<Catlass::Arch::FlagID>(t->dbg_sm_id); }
        uint32_t headNum = t->head_num;
        uint32_t headDim = t->head_dim;
        uint32_t B_q = t->B_q;
        uint32_t N_q = t->N_q;
        uint32_t B_k = t->B_k;
        uint32_t N_k = t->N_k;
        uint32_t numTask = t->numTask;
        uint32_t batchNum = t->batch_num;

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

        LayoutQ layoutQ{B_q, headDim, headNum * headDim};
        LayoutK layoutK{B_k, headDim, headNum * headDim};
        LayoutS layoutS{B_q, B_k};
        LayoutP layoutP{B_q, B_k};
        LayoutV layoutV{B_k, headDim, headNum * headDim};
        LayoutOTmp layoutOTmp{B_q, headDim};

        for (uint32_t taskIdx = AscendC::GetBlockIdx(); taskIdx < numTask; taskIdx += AscendC::GetBlockNum()) {
            TaskInfo taskInfo = DecodeTask(taskIdx, N_q, B_q, B_k, headNum, headDim);

            uint32_t pingpongFlagQK = 0;
            uint32_t pingpongFlagPV = 0;
            // ── 深度 PIPE_DEPTH 的软件流水（照 CATLASS examples/23 的 preLaunch=2）──
            // AIC: 每轮先做 QK[jj]，再做 PV[jj-PIPE_DEPTH]
            // ⇒ AIC 去等 smReady 时，AIV 已经有两整轮的时间在做那块 softmax，
            //   即「先置位、后等待」，而不是「先等待、后置位」。
            const uint64_t sSlotElems = (uint64_t)B_q * B_k;
            const uint64_t pSlotElems = (uint64_t)B_q * B_k;
            const uint64_t otSlotElems = (uint64_t)B_q * headDim;
            const uint32_t coreIdx = AscendC::GetBlockIdx();
            for (uint32_t jj = 0; jj <= taskInfo.nDiag + PIPE_DEPTH; ++jj) {
                if (jj <= taskInfo.nDiag) {
                    const uint32_t j = jj;
                    uint32_t visibleEnd = taskInfo.i * B_q + taskInfo.qRowNum;
                    uint32_t kRowNum = (B_k < visibleEnd - j * B_k) ? B_k : visibleEnd - j * B_k;
                    uint32_t kvOffset = taskInfo.b * N_k * headNum * headDim +
                                        j * B_k * headNum * headDim + taskInfo.h * headDim;

                    GemmCoord actualShapeQK{taskInfo.qRowNum, kRowNum, headDim};
                    const uint64_t sOff = ((uint64_t)coreIdx * NUM_SLOTS + (j % NUM_SLOTS)) * sSlotElems;
                    mmadQK(
                        qGm[taskInfo.qOffset], kGm[kvOffset], sGm[sOff], layoutQ, layoutK, layoutS, actualShapeQK,
                        pingpongFlagQK, j == 0);
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
                }
                if (jj >= PIPE_DEPTH) {
                    const uint32_t j = jj - PIPE_DEPTH;
                    uint32_t visibleEnd = taskInfo.i * B_q + taskInfo.qRowNum;
                    uint32_t kRowNum = (B_k < visibleEnd - j * B_k) ? B_k : visibleEnd - j * B_k;
                    uint32_t kvOffset = taskInfo.b * N_k * headNum * headDim +
                                        j * B_k * headNum * headDim + taskInfo.h * headDim;
                    const uint64_t slot = (uint64_t)coreIdx * NUM_SLOTS + (j % NUM_SLOTS);

                    GemmCoord actualShapePV{taskInfo.qRowNum, headDim, kRowNum};
#if !defined(FAT_QK_ONLY)
                    mmadPV(
                        pGm[slot * pSlotElems], vGm[kvOffset], oTmpGm[slot * otSlotElems], layoutP, layoutV,
                        layoutOTmp, actualShapePV, pingpongFlagPV,
                        smReady);   // ← 现在这里是「先置后等」
#endif
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);
                }
            }
        }
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(const FATParams& params)
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
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
        uint32_t headNum = t->head_num;
        uint32_t headDim = t->head_dim;

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

        LayoutS layoutS{B_q, B_k};
        LayoutP layoutP{B_q, B_k};
        LayoutO layoutO{B_q, headDim, headDim * headNum};
        LayoutOTmp layoutOTmp{B_q, headDim};
        LayoutMask layoutMask{B_q, B_k};

        const uint64_t sSlotElems = (uint64_t)B_q * B_k;
        const uint64_t pSlotElems = (uint64_t)B_q * B_k;
        const uint64_t otSlotElems = (uint64_t)B_q * headDim;
        const uint32_t coreIdx = AscendC::GetBlockIdx();

        for (uint32_t taskIdx = AscendC::GetBlockIdx(); taskIdx < t->numTask; taskIdx += AscendC::GetBlockNum()) {
            TaskInfo taskInfo = DecodeTask(taskIdx, N_q, B_q, B_k, headNum, headDim);

            // ── 与 AIC 对称的深度 PIPE_DEPTH 软件流水 ──
            // AIV: 每轮先做 softmax[jj]，再做 rescale[jj-PIPE_DEPTH]
            // ⇒ AIV 领先 AIC，AIC 去等 smReady 时它早已置好（先置后等）
            for (uint32_t jj = 0; jj <= taskInfo.nDiag + PIPE_DEPTH; ++jj) {
                if (jj <= taskInfo.nDiag) {
                    const uint32_t j = jj;
                    uint32_t visibleEnd = taskInfo.i * B_q + taskInfo.qRowNum;
                    uint32_t kRowNum = (B_k < visibleEnd - j * B_k) ? B_k : visibleEnd - j * B_k;
                    const uint64_t slot = (uint64_t)coreIdx * NUM_SLOTS + (j % NUM_SLOTS);
                    GemmCoord onlineSoftmaxShape{taskInfo.qRowNum, kRowNum, headDim};

                    bool isDiag = (j == taskInfo.nDiag);
                    if (isDiag) {
                        onlineSoftmax(
                            pGm[slot * pSlotElems], sGm[slot * sSlotElems], maskGm, layoutP, layoutS, layoutMask,
                            onlineSoftmaxShape, (j == 0) ? 1u : 0u, taskInfo.qRowNum, 1u, 0u, qkReady);
                    } else {
                        Arch::CrossCoreWaitFlag(qkReady);
                        onlineSoftmax(
                            pGm[slot * pSlotElems], sGm[slot * sSlotElems], layoutP, layoutS, onlineSoftmaxShape,
                            (j == 0) ? 1u : 0u, taskInfo.qRowNum, 1u, 0u);
                    }
                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(smReady);
                }
                if (jj >= PIPE_DEPTH) {
                    const uint32_t j = jj - PIPE_DEPTH;
                    uint32_t visibleEnd = taskInfo.i * B_q + taskInfo.qRowNum;
                    uint32_t kRowNum = (B_k < visibleEnd - j * B_k) ? B_k : visibleEnd - j * B_k;
                    const uint64_t slot = (uint64_t)coreIdx * NUM_SLOTS + (j % NUM_SLOTS);
                    uint32_t oOffset = taskInfo.b * N_q * headNum * headDim +
                                       taskInfo.i * B_q * headNum * headDim + taskInfo.h * headDim;

                    Arch::CrossCoreWaitFlag(pvReady);
                    GemmCoord rescaleOShape{taskInfo.qRowNum, headDim, kRowNum};
                    rescaleO(
                        oGm[oOffset], oTmpGm[slot * otSlotElems], layoutO, layoutOTmp, rescaleOShape,
                        taskInfo.qRowNum, 1u, (j == 0) ? 1u : 0u, j == taskInfo.nDiag, 0u);
                }
            }
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
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
    Arch::Resource<ArchTag> resource;
    Arch::CrossCoreFlag qkReady{QK_READY_ID};
    Arch::CrossCoreFlag smReady{SM_READY_ID};
    Arch::CrossCoreFlag pvReady{PV_READY_ID};

private:
    TaskInfo DecodeTask(uint32_t taskIdx, uint32_t N_q, uint32_t B_q, uint32_t B_k, uint32_t headNum, uint32_t headDim)
    {
        TaskInfo info;
        uint32_t blocksPerBatch = (N_q + B_q - 1) / B_q;
        info.b = taskIdx / (blocksPerBatch * headNum);
        info.i = (taskIdx % (blocksPerBatch * headNum)) / headNum;
        info.h = (taskIdx % (blocksPerBatch * headNum)) % headNum;
        info.nDiag = ((info.i + 1) * B_q - 1) / B_k;
        info.qRowNum = (B_q < N_q - info.i * B_q) ? B_q : N_q - info.i * B_q;
        info.qOffset = info.b * N_q * headNum * headDim + info.i * B_q * headDim * headNum + info.h * headDim;
        return info;
    }
};
