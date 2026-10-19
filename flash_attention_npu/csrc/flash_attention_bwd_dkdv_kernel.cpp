// clang-format off
#include "kernel_operator.h"
#include "flash_attention_bwd_tiling.h"
#include "lib/matmul_intf.h"
#include <cstdint>
// clang-format on

using namespace AscendC;
using namespace matmul;

constexpr uint32_t FLASH_QUEUE_DEPTH = 2;
constexpr uint32_t FLASH_BUFFER_SIZE = 2;

template <typename HeaderT>
__aicore__ inline void LoadTilingFromGM(GM_ADDR tilingGM, HeaderT &dstStack) {
  constexpr uint32_t HEADER_INTS =
      (sizeof(HeaderT) + sizeof(int32_t) - 1U) / sizeof(int32_t);
  const auto *gmInts = reinterpret_cast<const __gm__ int32_t *>(tilingGM);
  auto *dstInts = reinterpret_cast<int32_t *>(&dstStack);
  for (uint32_t i = 0; i < HEADER_INTS; ++i) {
    dstInts[i] = gmInts[i];
  }
}

class FlashAttentionBwd_dKdV_Kernel {
  using A_TYPE_QK = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
  using B_TYPE_QK = MatmulType<TPosition::GM, CubeFormat::ND, half, true>;
  using C_TYPE_QK = MatmulType<TPosition::VECCALC, CubeFormat::ND, half>;
  using A_TYPE_PdO = MatmulType<TPosition::VECCALC, CubeFormat::ND, half, true>;
  using B_TYPE_PdO = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
  using C_TYPE_PdO = MatmulType<TPosition::VECCALC, CubeFormat::ND, half>;
  using A_TYPE_dOV = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
  using B_TYPE_dOV = MatmulType<TPosition::GM, CubeFormat::ND, half, true>;
  using C_TYPE_dOV = MatmulType<TPosition::VECCALC, CubeFormat::ND, half>;
  using A_TYPE_dSQ = MatmulType<TPosition::VECCALC, CubeFormat::ND, half, true>;
  using B_TYPE_dSQ = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
  using C_TYPE_dSQ = MatmulType<TPosition::VECCALC, CubeFormat::ND, half>;

public:
  __aicore__ inline FlashAttentionBwd_dKdV_Kernel() {}
  __aicore__ inline void Init(GM_ADDR Q, GM_ADDR O, GM_ADDR dO, GM_ADDR K,
                              GM_ADDR V, GM_ADDR L, GM_ADDR D, GM_ADDR dK,
                              GM_ADDR dV, GM_ADDR tiling);
  __aicore__ inline void Process();
  __aicore__ inline void End() {
    mmQK_.End();
    mmPdO_.End();
    mmdOV_.End();
    mmdSQ_.End();
  }

private:
  __aicore__ inline void CopyInAndInit(uint32_t batch);
  __aicore__ inline void InnerCycle(uint32_t i, uint64_t offsetKV, uint64_t headOffsetQ, uint64_t headOffset1D);
  __aicore__ inline void InnerCycle_CopyIn(uint32_t i, uint64_t headOffset1D);
  __aicore__ inline void InnerCycle_Compute(uint32_t i, uint64_t offsetKV, uint64_t headOffsetQ);

private:
  TPipe pipe_;
  GlobalTensor<half> qGm_;
  GlobalTensor<half> oGm_;
  GlobalTensor<half> doGm_;
  GlobalTensor<half> kGm_;
  GlobalTensor<half> vGm_;
  GlobalTensor<half> lGm_;
  GlobalTensor<half> dGm_;
  GlobalTensor<half> dkGm_;
  GlobalTensor<half> dvGm_;
  TBuf<TPosition::VECCALC> dkBuf_;
  TBuf<TPosition::VECCALC> dvBuf_;
  TQue<TPosition::VECIN, 1> tmpMmResQue_;
  TQue<TPosition::VECIN, 1> dpQue_;
  TQue<TPosition::VECIN, 1> sQue_;
  TQue<TPosition::VECIN, 1> pQue_;
  TQue<TPosition::VECIN, 1> dsQue_;
  TBuf<TPosition::VECCALC> mmWs1Buf_;
  TBuf<TPosition::VECCALC> mmWs2Buf_;
  TBuf<TPosition::VECCALC> mmWs3Buf_;
  TBuf<TPosition::VECCALC> mmWs4Buf_;
  TBuf<TPosition::VECCALC> lBuf_;
  TBuf<TPosition::VECCALC> dBuf_;
  Matmul<A_TYPE_QK, B_TYPE_QK, C_TYPE_QK> mmQK_;
  Matmul<A_TYPE_PdO, B_TYPE_PdO, C_TYPE_PdO> mmPdO_;
  Matmul<A_TYPE_dOV, B_TYPE_dOV, C_TYPE_dOV> mmdOV_;
  Matmul<A_TYPE_dSQ, B_TYPE_dSQ, C_TYPE_dSQ> mmdSQ_;
  FlashAttentionBwd_dKdV_TilingData tilingData_;
  uint32_t batchNum_;
  uint32_t batchOffset_;

  static constexpr event_t EVENT_V_MTE3_ID = EVENT_ID0;
};

__aicore__ inline void
FlashAttentionBwd_dKdV_Kernel::Init(GM_ADDR Q, GM_ADDR O, GM_ADDR dO, GM_ADDR K,
                                    GM_ADDR V, GM_ADDR L, GM_ADDR D, GM_ADDR dK,
                                    GM_ADDR dV, GM_ADDR tiling) {
  LoadTilingFromGM(tiling, tilingData_);

  uint32_t blockIdx = GetBlockIdx();
  uint32_t blockNum = GetBlockNum();
  uint32_t T_k = (tilingData_.N_k + tilingData_.B_k - 1) / tilingData_.B_k;
  batchNum_ = T_k / blockNum;
  if (blockIdx < tilingData_.largeCoreNum) {
    batchNum_ += 1;
    batchOffset_ = blockIdx * batchNum_;
  } else {
    batchOffset_ = tilingData_.largeCoreNum * (batchNum_ + 1) +
                   (blockIdx - tilingData_.largeCoreNum) * batchNum_;
  }

  uint32_t totalBatch = (tilingData_.batchSize > 0) ? tilingData_.batchSize : 1;

  qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(Q),
                       totalBatch * tilingData_.N_q * tilingData_.d);
  oGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(O),
                       totalBatch * tilingData_.N_q * tilingData_.d);
  doGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(dO),
                        totalBatch * tilingData_.N_q * tilingData_.d);
  kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(K),
                       totalBatch * tilingData_.N_k * tilingData_.d);
  vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(V),
                       totalBatch * tilingData_.N_k * tilingData_.d);
  lGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(L), totalBatch * tilingData_.N_q);
  dGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(D), totalBatch * tilingData_.N_q);
  dkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(dK),
                        totalBatch * tilingData_.N_k * tilingData_.d);
  dvGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(dV),
                        totalBatch * tilingData_.N_k * tilingData_.d);

  pipe_.InitBuffer(lBuf_, tilingData_.B_q * sizeof(half));
  pipe_.InitBuffer(dBuf_, tilingData_.B_q * sizeof(half));

  pipe_.InitBuffer(dkBuf_, tilingData_.B_k * tilingData_.d * sizeof(half));
  pipe_.InitBuffer(dvBuf_, tilingData_.B_k * tilingData_.d * sizeof(half));
  pipe_.InitBuffer(sQue_, 1, tilingData_.B_q * tilingData_.B_k * sizeof(half));
  pipe_.InitBuffer(pQue_, 1, tilingData_.B_q * tilingData_.B_k * sizeof(half));
  pipe_.InitBuffer(dsQue_, 1, tilingData_.B_q * tilingData_.B_k * sizeof(half));
  pipe_.InitBuffer(tmpMmResQue_, 1,
                   tilingData_.B_k * tilingData_.d * sizeof(half));
  pipe_.InitBuffer(dpQue_, 1, tilingData_.B_q * tilingData_.B_k * sizeof(half));
  pipe_.InitBuffer(mmWs1Buf_, 16384);
  pipe_.InitBuffer(mmWs2Buf_, 16384);
  pipe_.InitBuffer(mmWs3Buf_, 16384);
  pipe_.InitBuffer(mmWs4Buf_, 16384);

  mmQK_.Init(&tilingData_.cubeTilingQK, &pipe_);
  mmQK_.SetOrgShape(tilingData_.cubeTilingQK.M, tilingData_.cubeTilingQK.N,
                    tilingData_.cubeTilingQK.Ka, tilingData_.cubeTilingQK.Kb);

  mmPdO_.Init(&tilingData_.cubeTilingPdO, &pipe_);
  mmPdO_.SetOrgShape(tilingData_.cubeTilingPdO.M, tilingData_.cubeTilingPdO.N,
                     tilingData_.cubeTilingPdO.Ka,
                     tilingData_.cubeTilingPdO.Kb);

  mmdOV_.Init(&tilingData_.cubeTilingdOV, &pipe_);
  mmdOV_.SetOrgShape(tilingData_.cubeTilingdOV.M, tilingData_.cubeTilingdOV.N,
                     tilingData_.cubeTilingdOV.Ka,
                     tilingData_.cubeTilingdOV.Kb);

  mmdSQ_.Init(&tilingData_.cubeTilingdSQ, &pipe_);
  mmdSQ_.SetOrgShape(tilingData_.cubeTilingdSQ.M, tilingData_.cubeTilingdSQ.N,
                     tilingData_.cubeTilingdSQ.Ka,
                     tilingData_.cubeTilingdSQ.Kb);

  LocalTensor<uint8_t> mmWs1 = mmWs1Buf_.Get<uint8_t>();
  LocalTensor<uint8_t> mmWs2 = mmWs2Buf_.Get<uint8_t>();
  LocalTensor<uint8_t> mmWs3 = mmWs3Buf_.Get<uint8_t>();
  LocalTensor<uint8_t> mmWs4 = mmWs4Buf_.Get<uint8_t>();
  mmQK_.SetLocalWorkspace(mmWs1);
  mmdOV_.SetLocalWorkspace(mmWs2);
  mmPdO_.SetLocalWorkspace(mmWs3);
  mmdSQ_.SetLocalWorkspace(mmWs4);
}

__aicore__ inline void FlashAttentionBwd_dKdV_Kernel::Process() {
  uint32_t totalBatch = (tilingData_.batchSize > 0) ? tilingData_.batchSize : 1;
  uint32_t T_q = (tilingData_.N_q + tilingData_.B_q - 1) / tilingData_.B_q;

  for (uint32_t h = 0; h < totalBatch; ++h) {
    uint64_t headOffsetKV = static_cast<uint64_t>(h) * tilingData_.N_k * tilingData_.d;
    uint64_t headOffsetQ = static_cast<uint64_t>(h) * tilingData_.N_q * tilingData_.d;
    uint64_t headOffset1D = static_cast<uint64_t>(h) * tilingData_.N_q;

    for (uint32_t batch = 0; batch < batchNum_; ++batch) {
      CopyInAndInit(batch);
      uint64_t offsetKV =
          headOffsetKV + (batchOffset_ + batch) * tilingData_.d * tilingData_.B_k;
      for (uint32_t i = 0; i < T_q; ++i) {
        InnerCycle(i, offsetKV, headOffsetQ, headOffset1D);
      }
      SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
      WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
      LocalTensor<half> dk = dkBuf_.Get<half>();
      LocalTensor<half> dv = dvBuf_.Get<half>();
      DataCopy(dkGm_[offsetKV], dk, tilingData_.B_k * tilingData_.d);
      DataCopy(dvGm_[offsetKV], dv, tilingData_.B_k * tilingData_.d);
      SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
      WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
  }
}

__aicore__ inline void
FlashAttentionBwd_dKdV_Kernel::CopyInAndInit(uint32_t batch) {
  LocalTensor<half> dkLocal = dkBuf_.Get<half>();
  LocalTensor<half> dvLocal = dvBuf_.Get<half>();
  Duplicate(dkLocal, static_cast<half>(0.0f), tilingData_.B_k * tilingData_.d);
  Duplicate(dvLocal, static_cast<half>(0.0f), tilingData_.B_k * tilingData_.d);
}

__aicore__ inline void
FlashAttentionBwd_dKdV_Kernel::InnerCycle(uint32_t i, uint64_t offsetKV,
                                          uint64_t headOffsetQ,
                                          uint64_t headOffset1D) {
  InnerCycle_CopyIn(i, headOffset1D);
  InnerCycle_Compute(i, offsetKV, headOffsetQ);
}

__aicore__ inline void
FlashAttentionBwd_dKdV_Kernel::InnerCycle_CopyIn(uint32_t i, uint64_t headOffset1D) {
  LocalTensor<half> lLocal = lBuf_.Get<half>();
  LocalTensor<half> dLocal = dBuf_.Get<half>();
  uint64_t offset1D = headOffset1D + i * tilingData_.B_q;
  DataCopy(lLocal, lGm_[offset1D], tilingData_.B_q);
  DataCopy(dLocal, dGm_[offset1D], tilingData_.B_q);
  SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
  WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
}

__aicore__ inline void
FlashAttentionBwd_dKdV_Kernel::InnerCycle_Compute(uint32_t i,
                                                  uint64_t offsetKV,
                                                  uint64_t headOffsetQ) {
  LocalTensor<half> dkLocal = dkBuf_.Get<half>();
  LocalTensor<half> dvLocal = dvBuf_.Get<half>();
  LocalTensor<uint8_t> mmWs1 = mmWs1Buf_.Get<uint8_t>();
  LocalTensor<uint8_t> mmWs2 = mmWs2Buf_.Get<uint8_t>();
  LocalTensor<uint8_t> mmWs3 = mmWs3Buf_.Get<uint8_t>();
  LocalTensor<uint8_t> mmWs4 = mmWs4Buf_.Get<uint8_t>();

  uint64_t offset2D = headOffsetQ + i * tilingData_.B_q * tilingData_.d;

  // 1. Q @ K^T -> S
  mmQK_.SetLocalWorkspace(mmWs1);
  mmQK_.SetTensorA(qGm_[offset2D], false);
  mmQK_.SetTensorB(kGm_[offsetKV], true);
  mmQK_.Iterate();
  LocalTensor<half> sRes = sQue_.AllocTensor<half>();
  mmQK_.GetTensorC(sRes);
  sQue_.EnQue(sRes);

  // 2. 阶段一异步重叠：Cube 异步计算 dO @ V^T -> dP，Vector 并发计算 Softmax P
  mmdOV_.SetLocalWorkspace(mmWs2);
  mmdOV_.SetTensorA(doGm_[offset2D], false);
  mmdOV_.SetTensorB(vGm_[offsetKV], true);
  mmdOV_.Iterate();

  LocalTensor<half> sLocal = sQue_.DeQue<half>();
  Muls(sLocal, sLocal, static_cast<half>(tilingData_.scale),
       tilingData_.B_q * tilingData_.B_k);
  LocalTensor<half> lLocal = lBuf_.Get<half>();
  for (uint32_t r = 0; r < tilingData_.B_q; r++) {
    half l = lLocal.GetValue(r);
    uint64_t rowOffset = r * tilingData_.B_k;
    Adds(sLocal[rowOffset], sLocal[rowOffset],
         static_cast<half>(-static_cast<float>(l)), tilingData_.B_k);
  }
  LocalTensor<half> pLocal = pQue_.AllocTensor<half>();
  Exp(pLocal, sLocal, tilingData_.B_q * tilingData_.B_k);
  sQue_.FreeTensor(sLocal);
  pQue_.EnQue(pLocal);

  LocalTensor<half> dpRes = dpQue_.AllocTensor<half>();
  mmdOV_.GetTensorC(dpRes);
  dpQue_.EnQue(dpRes);

  // 3. Vector 先读取 pCube 并计算 dS，彻底避免与 mmPdO_ (MTE1) 产生片上 UB 总线读写冲突
  LocalTensor<half> pCube = pQue_.DeQue<half>();
  LocalTensor<half> dLocal = dBuf_.Get<half>();
  LocalTensor<half> dpLocal = dpQue_.DeQue<half>();
  for (uint32_t r = 0; r < tilingData_.B_q; r++) {
    half d = dLocal.GetValue(r);
    uint64_t rowOffset = r * tilingData_.B_k;
    Adds(dpLocal[rowOffset], dpLocal[rowOffset],
         static_cast<half>(-static_cast<float>(d)), tilingData_.B_k);
  }

  LocalTensor<half> dsLocal = dsQue_.AllocTensor<half>();
  Mul(dpLocal, pCube, dpLocal, tilingData_.B_q * tilingData_.B_k);
  PipeBarrier<PIPE_V>();
  Muls(dsLocal, dpLocal, static_cast<half>(tilingData_.scale),
       tilingData_.B_q * tilingData_.B_k);
  dpQue_.FreeTensor(dpLocal);
  dsQue_.EnQue(dsLocal);

  SetFlag<HardEvent::V_S>(EVENT_ID0);
  WaitFlag<HardEvent::V_S>(EVENT_ID0);

  // 4. Vector 计算完毕后，安全下发 P^T @ dO -> dV (Cube)
  mmPdO_.SetLocalWorkspace(mmWs3);
  mmPdO_.SetTensorA(pCube, true);
  mmPdO_.SetTensorB(doGm_[offset2D], false);
  mmPdO_.Iterate();

  LocalTensor<half> tmpMmRes = tmpMmResQue_.AllocTensor<half>();
  mmPdO_.GetTensorC(tmpMmRes);
  pQue_.FreeTensor(pCube);
  tmpMmResQue_.EnQue(tmpMmRes);
  LocalTensor<half> tmpMmLocal = tmpMmResQue_.DeQue<half>();

  Add(dvLocal, dvLocal, tmpMmLocal, tilingData_.d * tilingData_.B_k);
  tmpMmResQue_.FreeTensor(tmpMmLocal);

  // 4. 计算 dS^T @ Q -> dK
  LocalTensor<half> dsCube = dsQue_.DeQue<half>();
  mmdSQ_.SetLocalWorkspace(mmWs4);
  mmdSQ_.SetTensorA(dsCube, true);
  mmdSQ_.SetTensorB(qGm_[offset2D], false);
  mmdSQ_.Iterate();
  LocalTensor<half> tmpMmRes2 = tmpMmResQue_.AllocTensor<half>();
  mmdSQ_.GetTensorC(tmpMmRes2);
  dsQue_.FreeTensor(dsCube);
  tmpMmResQue_.EnQue(tmpMmRes2);
  LocalTensor<half> tmpMmLocal2 = tmpMmResQue_.DeQue<half>();
  Add(dkLocal, dkLocal, tmpMmLocal2, tilingData_.d * tilingData_.B_k);
  tmpMmResQue_.FreeTensor(tmpMmLocal2);
  PipeBarrier<PIPE_V>();
}

extern "C" __global__ __aicore__ void
flash_attention_bwd_dkdv_kernel(GM_ADDR q, GM_ADDR o, GM_ADDR do_grad,
                                GM_ADDR k, GM_ADDR v, GM_ADDR l, GM_ADDR d,
                                GM_ADDR dk, GM_ADDR dv, GM_ADDR tiling) {
  if ASCEND_IS_AIV {
    return;
  }
  FlashAttentionBwd_dKdV_Kernel op;
  op.Init(q, o, do_grad, k, v, l, d, dk, dv, tiling);
  op.Process();
  op.End();
}
