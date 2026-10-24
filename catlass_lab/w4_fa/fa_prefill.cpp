/**
 * fa_prefill.cpp —— 教学版 FlashAttention 前向（prefill / 训练）
 *
 * 相对 examples/23_flash_attention_infer 删掉的东西：
 *   1. 【PagedAttention】K/V 连续存放，不再有 blockTable、不再查表求物理块号
 *   2. 【GQA/MQA】        numHeads == kvHeads，qHeads/kvHeads 两个 stride 合并成一个
 *   3. 【变长序列】       所有 batch 共用 qSeqlen == kvSeqlen == S
 *   4. 【流水深度】       preLaunch = 0，QK → softmax → PV 锁步，先求正确再谈性能
 *   5. 【Tail 变体】      要求 S 按 128 对齐分支处理，mask 只作用在对角块
 *
 * 保留的东西（这些是 FA 的骨架，不能删）：
 *   AIC/AIV 分离、跨核 flag、online softmax、O 在 UB 里累加、
 *   causal mask（128x128 上三角，只作用于对角块）、
 *   "全通前缀 + 对角块" 的 KV 分段。
 *
 * 之所以仍然用 *Tail* 那两个 BlockMmad：
 *   只有它们做到了"一个 paged block = 一条 Mmad = 一次 Fixpipe"（gCOffset = stackTile），
 *   普通变体是"两个 block 合一次写出"，S 的列布局对不上。传 maskTailS=0 即退化成普通逻辑。
 *
 * 约束：D = 128（模板写死）、half 精度。
 */

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "helper.hpp" // ACL_CHECK / acl / fp16_t

using namespace Catlass;

// ======================= 1. 常量 =======================

constexpr uint32_t HEAD_DIM = 128;                    // 头维，模板限制
constexpr uint32_t Q_BLK = 128;                       // 一个 q block 的行数
constexpr uint32_t KV_BLK = 128;                      // 一个 kv 分块（原来叫 pagedBlockSize）
constexpr uint32_t BLOCK_STACK_NUM = 4;               // 一次叠 4 个 KV 分块（原版 blockStackNum）
                                                      // ⚠ 与 GQA 的 groupSize 无关
constexpr uint32_t TILE_N = BLOCK_STACK_NUM * KV_BLK; // = 512，S/P 的行跨度

constexpr uint32_t S_ELEMS = Q_BLK * TILE_N;    // S workspace：65536 float
constexpr uint32_t P_ELEMS = Q_BLK * TILE_N;    // P workspace：65536 half
constexpr uint32_t OT_ELEMS = Q_BLK * HEAD_DIM; // oTemp：16384 float

constexpr uint32_t QK_READY_ID = 1;
constexpr uint32_t SM_READY_ID = 2;
constexpr uint32_t PV_READY_ID = 3;

// ======================= 2. tiling / params =======================

struct FaTiling {
    uint32_t batch;
    uint32_t numHeads;
    uint32_t qSeqlen;  // == kvSeqlen
    uint32_t numTasks; // batch * ceil(S/128) * numHeads
    float scaleValue;  // 1/sqrt(D)
};

struct FaParams {
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR mask; // 128x128 上三角，half
    GM_ADDR o;
    GM_ADDR s;     // workspace: 每核 S_ELEMS 个 float
    GM_ADDR p;     // workspace: 每核 P_ELEMS 个 half
    GM_ADDR oTemp; // workspace: 每核 OT_ELEMS 个 float
    GM_ADDR tiling;

    CATLASS_DEVICE
    FaParams()
    {}
    CATLASS_DEVICE
    FaParams(
        GM_ADDR q_, GM_ADDR k_, GM_ADDR v_, GM_ADDR mask_, GM_ADDR o_, GM_ADDR s_, GM_ADDR p_, GM_ADDR oTemp_,
        GM_ADDR tiling_)
        : q(q_), k(k_), v(v_), mask(mask_), o(o_), s(s_), p(p_), oTemp(oTemp_), tiling(tiling_)
    {}
};

// ======================= 3. kernel =======================

template <class BlockMmadQK, class BlockMmadPV, class EpilogueSM, class EpilogueRO>
class FaPrefillKernel {
public:
    using ArchTag = typename BlockMmadQK::ArchTag;

    using ElementQ = typename BlockMmadQK::ElementA;
    using LayoutQ = typename BlockMmadQK::LayoutA;
    using ElementK = typename BlockMmadQK::ElementB;
    using LayoutK = typename BlockMmadQK::LayoutB;
    using ElementS = typename BlockMmadQK::ElementC;
    using LayoutS = typename BlockMmadQK::LayoutC;

    using ElementP = typename BlockMmadPV::ElementA;
    using LayoutP = typename BlockMmadPV::LayoutA;
    using ElementV = typename BlockMmadPV::ElementB;
    using LayoutV = typename BlockMmadPV::LayoutB;

    using ElementMask = typename EpilogueSM::ElementMask;
    using LayoutMask = typename EpilogueSM::LayoutMask;

    using ElementO = typename EpilogueRO::ElementOutput;
    using LayoutO = typename EpilogueRO::LayoutOutput;
    using ElementOTmp = typename EpilogueRO::ElementInput;
    using LayoutOTmp = typename EpilogueRO::LayoutInput;

    CATLASS_DEVICE
    FaPrefillKernel()
    {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(FaParams const& params);

    // ---------------------- AIC：算 S = Q*K^T 和 O_tile = P*V ----------------------
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(FaParams const& params)
    {
        // 和原版一样：给 4 个 BlockMmad 补上 RAII 预置（它们构造函数是空的）
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

        // L1 分区：QK 占 96KB，PV 从 96KB 开始
        static constexpr uint32_t L1_QK_SIZE =
            BlockMmadQK::L1TileShape::M * BlockMmadQK::L1TileShape::K * sizeof(ElementQ) +
            BlockMmadQK::L1TileShape::N * BlockMmadQK::L1TileShape::K * sizeof(ElementK) * 2;

        BlockMmadQK mmadQK(resource);
        BlockMmadPV mmadPV(resource, L1_QK_SIZE);

        __gm__ FaTiling* t = reinterpret_cast<__gm__ FaTiling*>(params.tiling);
        const uint32_t batch = t->batch;
        const uint32_t heads = t->numHeads;
        const uint32_t seqLen = t->qSeqlen;
        const uint32_t numTask = t->numTasks;

        const uint32_t blocksPerSeq = (seqLen + Q_BLK - 1) / Q_BLK;
        const uint64_t strideQO = (uint64_t)heads * HEAD_DIM; // 无 GQA → Q/O 与 K/V 同一 stride
        const uint32_t strideKV = (uint32_t)strideQO;

        AscendC::GlobalTensor<ElementQ> gQ;
        gQ.SetGlobalBuffer((__gm__ ElementQ*)params.q);
        AscendC::GlobalTensor<ElementK> gK;
        gK.SetGlobalBuffer((__gm__ ElementK*)params.k);
        AscendC::GlobalTensor<ElementK> gV;
        gV.SetGlobalBuffer((__gm__ ElementK*)params.v);
        AscendC::GlobalTensor<ElementS> gS;
        gS.SetGlobalBuffer((__gm__ ElementS*)params.s);
        AscendC::GlobalTensor<ElementP> gP;
        gP.SetGlobalBuffer((__gm__ ElementP*)params.p);
        AscendC::GlobalTensor<ElementOTmp> gOT;
        gOT.SetGlobalBuffer((__gm__ ElementOTmp*)params.oTemp);
        // 无 PagedAttention：这个 tensor 只是为了让 mmad 的函数签名成立，永远不被读
        AscendC::GlobalTensor<int32_t> gDummyTable;
        gDummyTable.SetGlobalBuffer((__gm__ int32_t*)params.tiling);

        const uint32_t coreIdx = AscendC::GetBlockIdx();
        const uint32_t coreNum = AscendC::GetBlockNum();
        const uint32_t perBatch = blocksPerSeq * heads;

        // 每个核独占一段 workspace，否则多核会互相踩（GM 是全 device 共享的）
        const uint32_t wsS = coreIdx * S_ELEMS;
        const uint32_t wsP = coreIdx * P_ELEMS;
        const uint32_t wsOT = coreIdx * OT_ELEMS;

        for (uint32_t taskIdx = coreIdx; taskIdx < numTask; taskIdx += coreNum) {
            const uint32_t b = taskIdx / perBatch;
            const uint32_t rem = taskIdx - b * perBatch;
            const uint32_t qBlk = rem / heads;
            const uint32_t h = rem - qBlk * heads;

            const uint32_t q0 = qBlk * Q_BLK;
            const uint32_t rowNum = (seqLen - q0 < Q_BLK) ? (seqLen - q0) : Q_BLK;
            const uint32_t nDiag = q0 / KV_BLK; // 对角块编号 == 全通前缀的分块数

            // Q / K / V / O 的 GM 起点（K/V 只有 head 偏移，没有序列偏移）
            const uint64_t qOffset = ((uint64_t)b * seqLen + q0) * strideQO + (uint64_t)h * HEAD_DIM;
            const uint64_t kOffset = (uint64_t)b * seqLen * strideKV + (uint64_t)h * HEAD_DIM;
            const uint32_t embed = HEAD_DIM;

            LayoutQ layoutQ(rowNum, embed);
            // ⚠ 原版写的是 LayoutK(strideKV, TILE_N)：那是拿 shape(0) 当 stride 用（2 参构造里 stride[1]=rows）。
            //   形状上 K 的 embed 只有 128，不是 256，所以这里用 3 参形式把 shape 和 stride 分开写：
            //     ColumnMajor(rows, cols, ldm) → shape=(128, 512), stride=(1, strideKV)
            //     RowMajor   (rows, cols, ldm) → shape=(512, 128), stride=(strideKV, 1)
            //   两者 GetOffset 完全一样，但 shape() 现在是诚实的。
            LayoutK layoutK(HEAD_DIM, TILE_N, strideKV); // K 的 K^T 视图：[dim, token]
            LayoutV layoutV(TILE_N, HEAD_DIM, strideKV); // V 的原始视图：[token, dim]
            LayoutP layoutP(rowNum, TILE_N, TILE_N);     // P stride 固定 512

            // Q 整个 K 循环只搬一次
            uint32_t singleHead = 1;
            uint32_t qHeadsL = heads;
            mmadQK.loadQGM(gQ[qOffset], layoutQ, rowNum, singleHead, qHeadsL);

            uint32_t blkSize = KV_BLK; // 无 paged，mmad 内部不会用到它

            // ---------- ① 全通前缀：KV 块 [0, nDiag)，不需要 mask ----------
            bool firstQK = true;
            bool firstPV = true;
            for (uint32_t j = 0; j < nDiag; j += BLOCK_STACK_NUM) {
                const uint32_t nb = (nDiag - j < BLOCK_STACK_NUM) ? (nDiag - j) : BLOCK_STACK_NUM;
                const uint32_t tileN = nb * KV_BLK;
                uint32_t nIdx = j; // 非 const：mmad 的形参是 uint32_t&
                uint32_t nLoop = nDiag;
                const uint32_t kvLen = nDiag * KV_BLK; // 前缀是 128 的整数倍，无需 clamp

                GemmCoord shapeQK{rowNum, tileN, embed};
                mmadQK(
                    gQ[qOffset], gK[kOffset], gS[wsS], gDummyTable, layoutQ, layoutK, shapeQK, nIdx, nLoop, blkSize,
                    kvLen, strideKV, 0, firstQK ? 1 : 0);
                firstQK = false;
                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady); // S 已落到 GM

                GemmCoord shapePV{rowNum, embed, tileN};
                mmadPV(
                    gP[wsP], gV[kOffset], gOT[wsOT], gDummyTable, layoutP, layoutV, shapePV, nIdx, nLoop, blkSize,
                    kvLen, strideKV, smReady, 0, firstPV ? 1 : 0);
                firstPV = false;
                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady); // oTemp 已落到 GM
            }

            // ---------- ② 对角块：需要 causal mask ----------
            {
                const uint32_t tileN = rowNum;
                uint32_t nIdx = nDiag; // 非 const：mmad 的形参是 uint32_t&
                uint32_t nLoop = nDiag + 1;
                const uint32_t kvLen = q0 + rowNum; // 本 q block 能看到的最远 KV

                GemmCoord shapeQK{rowNum, tileN, embed};
                mmadQK(
                    gQ[qOffset], gK[kOffset], gS[wsS], gDummyTable, layoutQ, layoutK, shapeQK, nIdx, nLoop, blkSize,
                    kvLen, strideKV, 0, firstQK ? 1 : 0);
                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);

                GemmCoord shapePV{rowNum, embed, tileN};
                mmadPV(
                    gP[wsP], gV[kOffset], gOT[wsOT], gDummyTable, layoutP, layoutV, shapePV, nIdx, nLoop, blkSize,
                    kvLen, strideKV, smReady, 0, firstPV ? 1 : 0);
                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);
            }
        }

        // 收尾 drain（与入口预置一一对应）
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

    // ---------------------- AIV：online softmax + rescale O ----------------------
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(FaParams const& params)
    {
        // AIV 侧预置（和原版一致）
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);

        __gm__ FaTiling* t = reinterpret_cast<__gm__ FaTiling*>(params.tiling);
        const uint32_t heads = t->numHeads;
        const uint32_t seqLen = t->qSeqlen;
        const uint32_t numTask = t->numTasks;
        const float scale = t->scaleValue;

        const uint32_t blocksPerSeq = (seqLen + Q_BLK - 1) / Q_BLK;
        const uint64_t strideQO = (uint64_t)heads * HEAD_DIM;

        EpilogueSM epilogueSM(resource, scale);
        EpilogueRO epilogueRO(resource);

        AscendC::GlobalTensor<ElementS> gS;
        gS.SetGlobalBuffer((__gm__ ElementS*)params.s);
        AscendC::GlobalTensor<ElementP> gP;
        gP.SetGlobalBuffer((__gm__ ElementP*)params.p);
        AscendC::GlobalTensor<ElementOTmp> gOT;
        gOT.SetGlobalBuffer((__gm__ ElementOTmp*)params.oTemp);
        AscendC::GlobalTensor<ElementO> gO;
        gO.SetGlobalBuffer((__gm__ ElementO*)params.o);
        AscendC::GlobalTensor<ElementMask> gMask;
        gMask.SetGlobalBuffer((__gm__ ElementMask*)params.mask);

        const uint32_t coreIdx = AscendC::GetBlockIdx();
        const uint32_t coreNum = AscendC::GetBlockNum();
        const uint32_t perBatch = blocksPerSeq * heads;

        // 与 AIC 完全相同的分段公式：两边各算一遍，是跨核契约
        const uint32_t wsS = coreIdx * S_ELEMS;
        const uint32_t wsP = coreIdx * P_ELEMS;
        const uint32_t wsOT = coreIdx * OT_ELEMS;

        for (uint32_t taskIdx = coreIdx; taskIdx < numTask; taskIdx += coreNum) {
            const uint32_t b = taskIdx / perBatch;
            const uint32_t rem = taskIdx - b * perBatch;
            const uint32_t qBlk = rem / heads;
            const uint32_t h = rem - qBlk * heads;

            const uint32_t q0 = qBlk * Q_BLK;
            const uint32_t rowNum = (seqLen - q0 < Q_BLK) ? (seqLen - q0) : Q_BLK;
            const uint32_t nDiag = q0 / KV_BLK;
            const uint32_t embed = HEAD_DIM;

            const uint64_t oOffset = ((uint64_t)b * seqLen + q0) * strideQO + (uint64_t)h * HEAD_DIM;

            LayoutS layoutS(rowNum, TILE_N, TILE_N);
            LayoutP layoutP(rowNum, TILE_N, TILE_N);
            LayoutO layoutO(rowNum, (uint32_t)strideQO, (uint32_t)strideQO);
            LayoutOTmp layoutOTmp(rowNum, embed, embed);
            LayoutMask layoutMask(KV_BLK, KV_BLK, KV_BLK); // 128x128 上三角

            uint32_t tileIdx = 0;

            // ---------- ① 全通前缀 ----------
            for (uint32_t j = 0; j < nDiag; j += BLOCK_STACK_NUM) {
                const uint32_t nb = (nDiag - j < BLOCK_STACK_NUM) ? (nDiag - j) : BLOCK_STACK_NUM;
                const uint32_t tileN = nb * KV_BLK;

                GemmCoord shapeQK{rowNum, tileN, embed};
                Arch::CrossCoreWaitFlag(qkReady); // 等 AIC 把 S 写完
                // 参数末三个：isFirstStackTile, qSBlockSize(=rowNum), qNBlockSize(恒 1), curStackTileMod
                // qNBlockSize 是"这个 tile 里包含几个 Q head"。无 GQA 时一个 tile 只有 1 个 head，恒为 1
                epilogueSM(gP[wsP], gS[wsS], layoutP, layoutS, shapeQK, (tileIdx == 0), rowNum, 1, 0);
                Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(smReady); // P 已落到 GM

                GemmCoord shapePV{rowNum, embed, tileN};
                Arch::CrossCoreWaitFlag(pvReady); // 等 AIC 把 oTemp 写完
                // 末三个：isFirstStackTile, isLastStackTile, curStackTileMod
                epilogueRO(gO[oOffset], gOT[wsOT], layoutO, layoutOTmp, shapePV, rowNum, 1, (tileIdx == 0), 0, 0);
                tileIdx++;
            }

            // ---------- ② 对角块（带 mask） ----------
            {
                const uint32_t tileN = rowNum;
                GemmCoord shapeQK{rowNum, tileN, embed};
                // 带 mask 的重载内部自己 WaitFlag(qkReady)，所以这里不用显式 wait
                epilogueSM(
                    gP[wsP], gS[wsS], gMask, layoutP, layoutS, layoutMask, shapeQK, (tileIdx == 0), rowNum, 1, 0,
                    qkReady);
                Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(smReady);

                GemmCoord shapePV{rowNum, embed, tileN};
                Arch::CrossCoreWaitFlag(pvReady);
                epilogueRO(gO[oOffset], gOT[wsOT], layoutO, layoutOTmp, shapePV, rowNum, 1, (tileIdx == 0), 1, 0);
            }
        }

        // 收尾
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
    Arch::Resource<ArchTag> resource;
    Arch::CrossCoreFlag qkReady{QK_READY_ID};
    Arch::CrossCoreFlag smReady{SM_READY_ID};
    Arch::CrossCoreFlag pvReady{PV_READY_ID};
};

// ======================= 4. 入口 =======================

extern "C" CATLASS_GLOBAL void FaPrefillHalf(
    uint64_t hardwareSyncAddr, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask, GM_ADDR o, GM_ADDR s, GM_ADDR p,
    GM_ADDR oTemp, GM_ADDR tiling)
{
    AscendC::SetSyncBaseAddr(hardwareSyncAddr);

    using ArchTag = Arch::AtlasA2;
    using ElementQ = half;
    using LayoutQ = layout::RowMajor;
    using ElementK = half;
    using LayoutK = layout::ColumnMajor;
    using ElementV = half;
    using LayoutV = layout::RowMajor;
    using ElementS = float;
    using LayoutS = layout::RowMajor;
    using ElementP = half;
    using LayoutP = layout::RowMajor;
    using ElementO = half;
    using LayoutO = layout::RowMajor;
    using ElementMask = half;
    using LayoutMask = layout::RowMajor;
    using ElementOTmp = float;
    using LayoutOTmp = layout::RowMajor;
    using ElementUpdate = float;
    using LayoutUpdate = layout::RowMajor;

    using L1TileShape = GemmShape<128, 128, 128>;
    using L0TileShape = L1TileShape;

    using QType = Gemm::GemmType<ElementQ, LayoutQ>;
    using KType = Gemm::GemmType<ElementK, LayoutK>;
    using VType = Gemm::GemmType<ElementV, LayoutV>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using PType = Gemm::GemmType<ElementP, LayoutP>;
    using OType = Gemm::GemmType<ElementO, LayoutO>;
    using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
    using MaskType = Gemm::GemmType<ElementMask, LayoutMask>;
    using OUpdateType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;

    // 无 paged：第一个模板参数 false
    using DispatchQK = Gemm::MmadAtlasA2FAITailQK<false, false>;
    using DispatchPV = Gemm::MmadAtlasA2FAITailPV<false, false>;
    using BlockMmadQK = Gemm::Block::BlockMmad<DispatchQK, L1TileShape, L0TileShape, QType, KType, SType>;
    using BlockMmadPV = Gemm::Block::BlockMmad<DispatchPV, L1TileShape, L0TileShape, PType, VType, OTmpType>;

    using EpilogueSM = Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2OnlineSoftmax, PType, SType, MaskType>;
    using EpilogueRO = Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2RescaleO, OType, OTmpType, OUpdateType>;

    FaParams params{q, k, v, mask, o, s, p, oTemp, tiling};
    FaPrefillKernel<BlockMmadQK, BlockMmadPV, EpilogueSM, EpilogueRO> kernel;
    kernel(params);
}

// ======================= 5. host =======================

template <typename T>
static void FillRandom(std::vector<T>& buf, float lo, float hi, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (auto& x : buf) {
        x = static_cast<T>(dist(rng));
    }
}

// CPU golden：causal 单头 attention，O(S^2 * D)，测正确性够用
static void GoldenCausal(
    const std::vector<fp16_t>& q, const std::vector<fp16_t>& k, const std::vector<fp16_t>& v, std::vector<float>& o,
    uint32_t batch, uint32_t seq, uint32_t heads, float scale)
{
    o.assign((size_t)batch * seq * heads * HEAD_DIM, 0.0f);
    std::vector<float> s(seq);
    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t h = 0; h < heads; ++h) {
            for (uint32_t i = 0; i < seq; ++i) {
                float m = -1e30f;
                for (uint32_t j = 0; j <= i; ++j) {
                    float acc = 0.0f;
                    for (uint32_t d = 0; d < HEAD_DIM; ++d) {
                        float qv = static_cast<float>(q[(((size_t)b * seq + i) * heads + h) * HEAD_DIM + d]);
                        float kv = static_cast<float>(k[(((size_t)b * seq + j) * heads + h) * HEAD_DIM + d]);
                        acc += qv * kv;
                    }
                    s[j] = acc * scale;
                    m = (s[j] > m) ? s[j] : m;
                }
                float sum = 0.0f;
                for (uint32_t j = 0; j <= i; ++j) {
                    s[j] = std::exp(s[j] - m);
                    sum += s[j];
                }
                for (uint32_t d = 0; d < HEAD_DIM; ++d) {
                    float acc = 0.0f;
                    for (uint32_t j = 0; j <= i; ++j) {
                        float vv = static_cast<float>(v[(((size_t)b * seq + j) * heads + h) * HEAD_DIM + d]);
                        acc += s[j] * vv;
                    }
                    o[(((size_t)b * seq + i) * heads + h) * HEAD_DIM + d] = acc / sum;
                }
            }
        }
    }
}

int main(int argc, const char** argv)
{
    uint32_t batch = (argc > 1) ? (uint32_t)atoi(argv[1]) : 1;
    uint32_t seq = (argc > 2) ? (uint32_t)atoi(argv[2]) : 256;
    uint32_t heads = (argc > 3) ? (uint32_t)atoi(argv[3]) : 2;
    uint32_t deviceId = 0;

    const float scale = 1.0f / std::sqrt((float)HEAD_DIM);
    const uint32_t blocksPerSeq = (seq + Q_BLK - 1) / Q_BLK;
    const uint32_t numTasks = batch * blocksPerSeq * heads;
    const uint64_t strideQO = (uint64_t)heads * HEAD_DIM;

    size_t qoLen = (size_t)batch * seq * strideQO;
    size_t kvLen = qoLen;
    size_t maskLen = (size_t)KV_BLK * KV_BLK;

    std::vector<fp16_t> hostQ(qoLen), hostK(kvLen), hostV(kvLen), hostO(qoLen), hostMask(maskLen);
    FillRandom(hostQ, -1.0f, 1.0f, 1);
    FillRandom(hostK, -1.0f, 1.0f, 2);
    FillRandom(hostV, -1.0f, 1.0f, 3);
    for (uint32_t r = 0; r < KV_BLK; ++r) {
        for (uint32_t c = 0; c < KV_BLK; ++c) {
            hostMask[r * KV_BLK + c] = (c > r) ? (fp16_t)1.0f : (fp16_t)0.0f; // 上三角=屏蔽
        }
    }

    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    uint32_t blockDim = (numTasks < (uint32_t)aicCoreNum) ? numTasks : (uint32_t)aicCoreNum;

    FaTiling hostTiling{batch, heads, seq, numTasks, scale};

    uint8_t *qDev, *kDev, *vDev, *maskDev, *oDev, *sDev, *pDev, *otDev, *tilingDev;
    size_t qoBytes = qoLen * sizeof(fp16_t), maskBytes = maskLen * sizeof(fp16_t);
    size_t sBytes = (size_t)blockDim * S_ELEMS * sizeof(float);
    size_t pBytes = (size_t)blockDim * P_ELEMS * sizeof(fp16_t);
    size_t otBytes = (size_t)blockDim * OT_ELEMS * sizeof(float);

    ACL_CHECK(aclrtMalloc((void**)&qDev, qoBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void**)&kDev, qoBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void**)&vDev, qoBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void**)&oDev, qoBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void**)&maskDev, maskBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void**)&sDev, sBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void**)&pDev, pBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void**)&otDev, otBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void**)&tilingDev, sizeof(FaTiling), ACL_MEM_MALLOC_HUGE_FIRST));

    ACL_CHECK(aclrtMemcpy(qDev, qoBytes, hostQ.data(), qoBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(kDev, qoBytes, hostK.data(), qoBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(vDev, qoBytes, hostV.data(), qoBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(maskDev, maskBytes, hostMask.data(), maskBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(tilingDev, sizeof(FaTiling), &hostTiling, sizeof(FaTiling), ACL_MEMCPY_HOST_TO_DEVICE));

    uint64_t hwSyncAddr{0};
    ACL_CHECK(aclrtGetHardwareSyncAddr(reinterpret_cast<void**>(&hwSyncAddr)));

    FaPrefillHalf<<<blockDim, nullptr, stream>>>(
        hwSyncAddr, qDev, kDev, vDev, maskDev, oDev, sDev, pDev, otDev, tilingDev);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtMemcpy(hostO.data(), qoBytes, oDev, qoBytes, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> golden;
    GoldenCausal(hostQ, hostK, hostV, golden, batch, seq, heads, scale);

    double maxErr = 0.0, sumErr = 0.0;
    size_t bad = 0;
    for (size_t i = 0; i < (size_t)batch * seq * strideQO; ++i) {
        double diff = std::fabs((double)static_cast<float>(hostO[i]) - (double)golden[i]);
        maxErr = (diff > maxErr) ? diff : maxErr;
        sumErr += diff;
        if (diff > 2e-2) {
            if (bad < 5) {
                printf("[BAD] idx=%zu got=%.5f ref=%.5f\n", i, (float)hostO[i], golden[i]);
            }
            bad++;
        }
    }
    printf(
        "task=%u blockDim=%u maxErr=%.5f avgErr=%.6f bad=%zu/%zu\n", numTasks, blockDim, maxErr,
        sumErr / (double)((size_t)batch * seq * strideQO), bad, (size_t)batch * seq * strideQO);
    printf("%s\n", (bad == 0) ? "Compare success." : "Compare FAILED.");

    aclrtFree(qDev);
    aclrtFree(kDev);
    aclrtFree(vDev);
    aclrtFree(oDev);
    aclrtFree(maskDev);
    aclrtFree(sDev);
    aclrtFree(pDev);
    aclrtFree(otDev);
    aclrtFree(tilingDev);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return (bad == 0) ? 0 : 1;
}
