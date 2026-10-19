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

class FlashAttentionBwd_dQ_Kernel {
  using A_TYPE_QK = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
  using B_TYPE_QK = MatmulType<TPosition::GM, CubeFormat::ND, half, true>;
  using C_TYPE_QK = MatmulType<TPosition::VECCALC, CubeFormat::ND, half>;
  using A_TYPE_dOV = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
  using B_TYPE_dOV = MatmulType<TPosition::GM, CubeFormat::ND, half, true>;
  using C_TYPE_dOV = MatmulType<TPosition::VECCALC, CubeFormat::ND, half>;
  using A_TYPE_dSK =
      MatmulType<TPosition::VECCALC, CubeFormat::ND, half, false>;
  using B_TYPE_dSK = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
  using C_TYPE_dSK = MatmulType<TPosition::VECCALC, CubeFormat::ND, half>;

public:
  __aicore__ inline FlashAttentionBwd_dQ_Kernel() {}
  __aicore__ inline void Init(GM_ADDR Q, GM_ADDR O, GM_ADDR dO, GM_ADDR K,
                              GM_ADDR V, GM_ADDR L, GM_ADDR D, GM_ADDR dQ,
                              GM_ADDR tiling);
  __aicore__ inline void Process();
  __aicore__ inline void End() {
    mmQK_.End();
    mmdOV_.End();
    mmdSK_.End();
  }

private:
  __aicore__ inline void CopyInAndInit(uint32_t batch, uint64_t headOffset1D);
  __aicore__ inline void InnerCycle(uint32_t i, uint64_t offsetQ, uint64_t headOffsetKV);
  __aicore__ inline void InnerCycle_Compute(uint32_t i, uint64_t offsetQ, uint64_t headOffsetKV);

private:
  TPipe pipe_;
  GlobalTensor<half> qGm_;
  GlobalTensor<half> oGm_;
  GlobalTensor<half> doGm_;
  GlobalTensor<half> kGm_;
  GlobalTensor<half> vGm_;
  GlobalTensor<half> lGm_;
  GlobalTensor<half> dGm_;
  GlobalTensor<half> dqGm_;
  TBuf<TPosition::VECCALC> lBuf_;
  TBuf<TPosition::VECCALC> dBuf_;
  TBuf<TPosition::VECCALC> dqBuf_;
  TQue<TPosition::VECIN, 1> tmpMmResQue_;
  TQue<TPosition::VECIN, 1> dpQue_;
  TQue<TPosition::VECIN, 1> sQue_;
  TQue<TPosition::VECIN, 1> dsQue_;
  TBuf<TPosition::VECCALC> mmWs1Buf_;
  TBuf<TPosition::VECCALC> mmWs2Buf_;
  TBuf<TPosition::VECCALC> mmWs3Buf_;
  Matmul<A_TYPE_QK, B_TYPE_QK, C_TYPE_QK> mmQK_;
  Matmul<A_TYPE_dOV, B_TYPE_dOV, C_TYPE_dOV> mmdOV_;
  Matmul<A_TYPE_dSK, B_TYPE_dSK, C_TYPE_dSK> mmdSK_;
  FlashAttentionBwd_dQ_TilingData tilingData_;
  uint32_t batchNum_;
  uint32_t batchOffset_;

  static constexpr event_t EVENT_V_MTE3_ID = EVENT_ID0;
};

__aicore__ inline void FlashAttentionBwd_dQ_Kernel::Init(GM_ADDR Q, GM_ADDR O,
                                                         GM_ADDR dO, GM_ADDR K,
                                                         GM_ADDR V, GM_ADDR L,
                                                         GM_ADDR D, GM_ADDR dQ,
                                                         GM_ADDR tiling) {
  LoadTilingFromGM(tiling, tilingData_);

  uint32_t blockIdx = GetBlockIdx();
  uint32_t blockNum = GetBlockNum();
  uint32_t T_q = (tilingData_.N_q + tilingData_.B_q - 1) / tilingData_.B_q;
  batchNum_ = T_q / blockNum;
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
  dqGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(dQ),
                        totalBatch * tilingData_.N_q * tilingData_.d);

  pipe_.InitBuffer(sQue_, 1, tilingData_.B_q * tilingData_.B_k * sizeof(half));
  pipe_.InitBuffer(dsQue_, 1, tilingData_.B_q * tilingData_.B_k * sizeof(half));
  pipe_.InitBuffer(tmpMmResQue_, 1,
                   tilingData_.B_q * tilingData_.d * sizeof(half));
  pipe_.InitBuffer(dpQue_, 1, tilingData_.B_q * tilingData_.B_k * sizeof(half));

  pipe_.InitBuffer(dqBuf_, tilingData_.B_q * tilingData_.d * sizeof(half));
  pipe_.InitBuffer(lBuf_, tilingData_.B_q * sizeof(half));
  pipe_.InitBuffer(dBuf_, tilingData_.B_q * sizeof(half));
  pipe_.InitBuffer(mmWs1Buf_, 16384);
  pipe_.InitBuffer(mmWs2Buf_, 16384);
  pipe_.InitBuffer(mmWs3Buf_, 16384);

  mmQK_.Init(&tilingData_.cubeTilingQK, &pipe_);
  mmQK_.SetOrgShape(tilingData_.cubeTilingQK.M, tilingData_.cubeTilingQK.N,
                    tilingData_.cubeTilingQK.Ka, tilingData_.cubeTilingQK.Kb);

  mmdOV_.Init(&tilingData_.cubeTilingdOV, &pipe_);
  mmdOV_.SetOrgShape(tilingData_.cubeTilingdOV.M, tilingData_.cubeTilingdOV.N,
                     tilingData_.cubeTilingdOV.Ka,
                     tilingData_.cubeTilingdOV.Kb);

  mmdSK_.Init(&tilingData_.cubeTilingdSK, &pipe_);
  mmdSK_.SetOrgShape(tilingData_.cubeTilingdSK.M, tilingData_.cubeTilingdSK.N,
                     tilingData_.cubeTilingdSK.Ka,
                     tilingData_.cubeTilingdSK.Kb);

  LocalTensor<uint8_t> mmWs1 = mmWs1Buf_.Get<uint8_t>();
  LocalTensor<uint8_t> mmWs2 = mmWs2Buf_.Get<uint8_t>();
  LocalTensor<uint8_t> mmWs3 = mmWs3Buf_.Get<uint8_t>();
  mmQK_.SetLocalWorkspace(mmWs1);
  mmdOV_.SetLocalWorkspace(mmWs2);
  mmdSK_.SetLocalWorkspace(mmWs3);
}

__aicore__ inline void FlashAttentionBwd_dQ_Kernel::Process() {
  uint32_t totalBatch = (tilingData_.batchSize > 0) ? tilingData_.batchSize : 1;
  uint32_t T_k = (tilingData_.N_k + tilingData_.B_k - 1) / tilingData_.B_k;

  for (uint32_t h = 0; h < totalBatch; ++h) {
    uint64_t headOffsetQ = static_cast<uint64_t>(h) * tilingData_.N_q * tilingData_.d;
    uint64_t headOffsetKV = static_cast<uint64_t>(h) * tilingData_.N_k * tilingData_.d;
    uint64_t headOffset1D = static_cast<uint64_t>(h) * tilingData_.N_q;

    for (uint32_t batch = 0; batch < batchNum_; ++batch) {
      CopyInAndInit(batch, headOffset1D);
      uint64_t offsetQ = headOffsetQ + (batchOffset_ + batch) * tilingData_.B_q * tilingData_.d;
      for (uint32_t i = 0; i < T_k; ++i) {
        InnerCycle(i, offsetQ, headOffsetKV);
      }
      SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
      WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
      LocalTensor<half> dq = dqBuf_.Get<half>();
      DataCopy(dqGm_[offsetQ], dq, tilingData_.B_q * tilingData_.d);
      SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
      WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
  }
}

__aicore__ inline void
FlashAttentionBwd_dQ_Kernel::CopyInAndInit(uint32_t batch, uint64_t headOffset1D) {
  LocalTensor<half> dqLocal = dqBuf_.Get<half>();
  LocalTensor<half> lLocal = lBuf_.Get<half>();
  LocalTensor<half> dLocal = dBuf_.Get<half>();

  Duplicate(dqLocal, static_cast<half>(0.0f), tilingData_.B_q * tilingData_.d);

  uint64_t offset1D = headOffset1D + (batchOffset_ + batch) * tilingData_.B_q;
  DataCopy(lLocal, lGm_[offset1D], tilingData_.B_q);
  DataCopy(dLocal, dGm_[offset1D], tilingData_.B_q);
  SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
  WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
}

__aicore__ inline void
FlashAttentionBwd_dQ_Kernel::InnerCycle(uint32_t i, uint64_t offsetQ, uint64_t headOffsetKV) {
  InnerCycle_Compute(i, offsetQ, headOffsetKV);
}

__aicore__ inline void
FlashAttentionBwd_dQ_Kernel::InnerCycle_Compute(uint32_t i, uint64_t offsetQ, uint64_t headOffsetKV) {
  LocalTensor<half> lLocal = lBuf_.Get<half>();
  LocalTensor<half> dLocal = dBuf_.Get<half>();
  LocalTensor<half> dqLocal = dqBuf_.Get<half>();
  LocalTensor<uint8_t> mmWs1 = mmWs1Buf_.Get<uint8_t>();
  LocalTensor<uint8_t> mmWs2 = mmWs2Buf_.Get<uint8_t>();
  LocalTensor<uint8_t> mmWs3 = mmWs3Buf_.Get<uint8_t>();

  uint64_t offsetKV = headOffsetKV + i * tilingData_.B_k * tilingData_.d;

  // 1. Q @ K^T -> S
  mmQK_.SetLocalWorkspace(mmWs1);
  mmQK_.SetTensorA(qGm_[offsetQ], false);
  mmQK_.SetTensorB(kGm_[offsetKV], true);
  mmQK_.Iterate();
  LocalTensor<half> sRes = sQue_.AllocTensor<half>();
  mmQK_.GetTensorC(sRes);
  sQue_.EnQue(sRes);

  // 2. 异步下发 dO @ V^T -> dP（Cube 后台运行）
  mmdOV_.SetLocalWorkspace(mmWs2);
  mmdOV_.SetTensorA(doGm_[offsetQ], false);
  mmdOV_.SetTensorB(vGm_[offsetKV], true);
  mmdOV_.Iterate();

  // 3. 与 mmdOV_ 并发：Vector 计算 Softmax P = exp(S * scale - l)
  LocalTensor<half> sLocal = sQue_.DeQue<half>();
  Muls(sLocal, sLocal, static_cast<half>(tilingData_.scale),
       tilingData_.B_q * tilingData_.B_k);
  for (uint32_t r = 0; r < tilingData_.B_q; r++) {
    half l = lLocal.GetValue(r);
    uint64_t rowOffset = r * tilingData_.B_k;
    Adds(sLocal[rowOffset], sLocal[rowOffset],
         static_cast<half>(-static_cast<float>(l)), tilingData_.B_k);
  }
  Exp(sLocal, sLocal, tilingData_.B_q * tilingData_.B_k);

  // 4. 等待并取出 dP
  LocalTensor<half> dpRes = dpQue_.AllocTensor<half>();
  mmdOV_.GetTensorC(dpRes);
  dpQue_.EnQue(dpRes);

  // 5. Vector 计算常数折叠后的 dS = (dP - D) * P * scale
  LocalTensor<half> dpLocal = dpQue_.DeQue<half>();
  for (uint32_t r = 0; r < tilingData_.B_q; r++) {
    half d = dLocal.GetValue(r);
    uint64_t rowOffset = r * tilingData_.B_k;
    Adds(dpLocal[rowOffset], dpLocal[rowOffset],
         static_cast<half>(-static_cast<float>(d)), tilingData_.B_k);
  }
  LocalTensor<half> dsLocal = dsQue_.AllocTensor<half>();
  Mul(dpLocal, sLocal, dpLocal, tilingData_.B_q * tilingData_.B_k);
  sQue_.FreeTensor(sLocal);
  PipeBarrier<PIPE_V>();
  Muls(dsLocal, dpLocal, static_cast<half>(tilingData_.scale),
       tilingData_.B_q * tilingData_.B_k);
  dpQue_.FreeTensor(dpLocal);
  dsQue_.EnQue(dsLocal);

  // 6. dS @ K -> dQ
  LocalTensor<half> dsCube = dsQue_.DeQue<half>();
  mmdSK_.SetLocalWorkspace(mmWs3);
  mmdSK_.SetTensorA(dsCube, false);
  mmdSK_.SetTensorB(kGm_[offsetKV], false);
  mmdSK_.Iterate();
  LocalTensor<half> tmpMmRes = tmpMmResQue_.AllocTensor<half>();
  mmdSK_.GetTensorC(tmpMmRes);
  dsQue_.FreeTensor(dsCube);
  tmpMmResQue_.EnQue(tmpMmRes);
  LocalTensor<half> tmpMmLocal = tmpMmResQue_.DeQue<half>();

  Add(dqLocal, dqLocal, tmpMmLocal, tilingData_.d * tilingData_.B_q);
  tmpMmResQue_.FreeTensor(tmpMmLocal);
  PipeBarrier<PIPE_V>();
}

extern "C" __global__ __aicore__ void
flash_attention_bwd_dq_kernel(GM_ADDR q, GM_ADDR o, GM_ADDR do_grad, GM_ADDR k,
                              GM_ADDR v, GM_ADDR l, GM_ADDR d, GM_ADDR dq,
                              GM_ADDR tiling) {
  if ASCEND_IS_AIV {
    return;
  }
  FlashAttentionBwd_dQ_Kernel op;
  op.Init(q, o, do_grad, k, v, l, d, dq, tiling);
  op.Process();
  op.End();
}
