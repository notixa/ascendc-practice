// clang-format off
#include "kernel_operator.h"
#include "flash_attention_fwd_tiling.h"
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

class FlashAttentionFwdKernel {
  using A_TYPE_QK = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
  using B_TYPE_QK = MatmulType<TPosition::GM, CubeFormat::ND, half, true>;
  using C_TYPE_QK = MatmulType<TPosition::VECCALC, CubeFormat::ND, half>;
  using A_TYPE_PV = MatmulType<TPosition::VECCALC, CubeFormat::ND, half, false>;
  using B_TYPE_PV = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
  using C_TYPE_PV = MatmulType<TPosition::VECCALC, CubeFormat::ND, half>;

public:
  __aicore__ inline FlashAttentionFwdKernel() {}
  __aicore__ inline void Init(GM_ADDR Q, GM_ADDR K, GM_ADDR V, GM_ADDR O,
                              GM_ADDR L, GM_ADDR tiling);
  __aicore__ inline void Process();
  __aicore__ inline void End() {
    mmQK_.End();
    mmPV_.End();
  }

private:
  __aicore__ inline void InnerCycle(int32_t j, uint32_t batch,
                                    uint64_t batchOffset, uint64_t headOffsetQ,
                                    uint64_t headOffsetK, uint64_t headOffsetV);
  __aicore__ inline void Compute(uint64_t offsetV, int32_t j);

private:
  TPipe pipe_;
  uint32_t batchNum_;
  uint64_t batchOffset_;

  FlashAttentionFwdTilingData tilingData_;
  GlobalTensor<half> qGm_;
  GlobalTensor<half> kGm_;
  GlobalTensor<half> vGm_;
  GlobalTensor<half> oGm_;
  GlobalTensor<half> lGm_;
  TQue<TPosition::VECIN, 1> sQue_;
  TQue<TPosition::VECIN, 1> pQue_;
  TBuf<TPosition::VECCALC> L;
  TBuf<TPosition::VECCALC> O;
  TBuf<TPosition::VECCALC> mBuf_;
  TBuf<TPosition::VECCALC> prevMBuf_;
  TBuf<TPosition::VECCALC> rowMaxBuf_;
  TBuf<TPosition::VECCALC> rowSumBuf_;
  TBuf<TPosition::VECCALC> invLBuf_;
  TBuf<TPosition::VECCALC> scaleBuf_;
  TBuf<TPosition::VECCALC> reduceBuf_;
  TQue<TPosition::VECIN, 1> pvResQue_;
  TBuf<TPosition::VECCALC> mmWsQKBuf_;
  TBuf<TPosition::VECCALC> mmWsPVBuf_;
  Matmul<A_TYPE_QK, B_TYPE_QK, C_TYPE_QK> mmQK_;
  Matmul<A_TYPE_PV, B_TYPE_PV, C_TYPE_PV> mmPV_;
};

__aicore__ inline void FlashAttentionFwdKernel::Init(GM_ADDR Q, GM_ADDR K,
                                                     GM_ADDR V, GM_ADDR O,
                                                     GM_ADDR L,
                                                     GM_ADDR tiling) {
  LoadTilingFromGM(tiling, tilingData_);

  uint32_t blockNum = GetBlockNum();
  uint32_t blockIdx = GetBlockIdx();
  uint32_t T_q = (tilingData_.N_q + tilingData_.B_q - 1) / tilingData_.B_q;
  batchNum_ = T_q / blockNum;
  if (blockIdx < tilingData_.largeCoreNum) {
    batchNum_ += 1;
  }
  if (blockIdx < tilingData_.largeCoreNum) {
    batchOffset_ = batchNum_ * blockIdx;
  } else {
    batchOffset_ = tilingData_.largeCoreNum * (batchNum_ + 1) +
                   (blockIdx - tilingData_.largeCoreNum) * batchNum_;
  }

  uint32_t totalBatch = (tilingData_.batchSize > 0) ? tilingData_.batchSize : 1;

  qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(Q),
                       totalBatch * tilingData_.N_q * tilingData_.d);
  kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(K),
                       totalBatch * tilingData_.N_k * tilingData_.d);
  vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(V),
                       totalBatch * tilingData_.N_k * tilingData_.d);
  oGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(O),
                       totalBatch * tilingData_.N_q * tilingData_.d);
  lGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(L),
                       totalBatch * tilingData_.N_q);

  pipe_.InitBuffer(sQue_, 1, tilingData_.B_q * tilingData_.B_k * sizeof(half));
  pipe_.InitBuffer(pQue_, 1, tilingData_.B_q * tilingData_.B_k * sizeof(half));
  pipe_.InitBuffer(this->L, tilingData_.B_q * sizeof(half));
  pipe_.InitBuffer(this->O, tilingData_.B_q * tilingData_.d * sizeof(half));
  pipe_.InitBuffer(mBuf_, tilingData_.B_q * sizeof(half));
  pipe_.InitBuffer(prevMBuf_, tilingData_.B_q * sizeof(half));
  pipe_.InitBuffer(rowMaxBuf_, tilingData_.B_q * sizeof(half));
  pipe_.InitBuffer(rowSumBuf_, tilingData_.B_q * sizeof(half));
  pipe_.InitBuffer(invLBuf_, tilingData_.B_q * sizeof(half));
  pipe_.InitBuffer(scaleBuf_, tilingData_.B_q * sizeof(half));
  pipe_.InitBuffer(reduceBuf_, 4096);
  pipe_.InitBuffer(pvResQue_, 1,
                   tilingData_.B_q * tilingData_.d * sizeof(half));
  pipe_.InitBuffer(mmWsQKBuf_, 16384);
  pipe_.InitBuffer(mmWsPVBuf_, 16384);

  mmQK_.Init(&tilingData_.cubeTilingQK, &pipe_);
  mmQK_.SetOrgShape(tilingData_.cubeTilingQK.M, tilingData_.cubeTilingQK.N,
                    tilingData_.cubeTilingQK.Ka, tilingData_.cubeTilingQK.Kb);

  mmPV_.Init(&tilingData_.cubeTilingPV, &pipe_);
  mmPV_.SetOrgShape(tilingData_.cubeTilingPV.M, tilingData_.cubeTilingPV.N,
                    tilingData_.cubeTilingPV.Ka, tilingData_.cubeTilingPV.Kb);

  LocalTensor<uint8_t> mmWsQK = mmWsQKBuf_.Get<uint8_t>();
  LocalTensor<uint8_t> mmWsPV = mmWsPVBuf_.Get<uint8_t>();
  mmQK_.SetLocalWorkspace(mmWsQK);
  mmPV_.SetLocalWorkspace(mmWsPV);
}

__aicore__ inline void FlashAttentionFwdKernel::Process() {
  uint32_t totalBatch = (tilingData_.batchSize > 0) ? tilingData_.batchSize : 1;
  int32_t T_k = (tilingData_.N_k + tilingData_.B_k - 1) / tilingData_.B_k;

  for (uint32_t h = 0; h < totalBatch; ++h) {
    uint64_t headOffsetQ =
        static_cast<uint64_t>(h) * tilingData_.N_q * tilingData_.d;
    uint64_t headOffsetK =
        static_cast<uint64_t>(h) * tilingData_.N_k * tilingData_.d;
    uint64_t headOffsetV =
        static_cast<uint64_t>(h) * tilingData_.N_k * tilingData_.d;
    uint64_t headOffsetO =
        static_cast<uint64_t>(h) * tilingData_.N_q * tilingData_.d;
    uint64_t headOffsetL = static_cast<uint64_t>(h) * tilingData_.N_q;

    for (uint32_t batch = 0; batch < batchNum_; ++batch) {
      for (int32_t j = 0; j < T_k; ++j) {
        InnerCycle(j, batch, batchOffset_, headOffsetQ, headOffsetK,
                   headOffsetV);
      }
      LocalTensor<half> o = O.Get<half>();
      LocalTensor<half> l = L.Get<half>();
      LocalTensor<half> m = mBuf_.Get<half>();
      LocalTensor<half> logL = rowSumBuf_.Get<half>();
      LocalTensor<half> invL = invLBuf_.Get<half>();

      Reciprocal(invL, l, tilingData_.B_q);
      SetFlag<HardEvent::V_S>(EVENT_ID0);
      WaitFlag<HardEvent::V_S>(EVENT_ID0);
      for (int32_t i = 0; i < tilingData_.B_q; ++i) {
        Muls(o[i * tilingData_.d], o[i * tilingData_.d], invL.GetValue(i),
             tilingData_.d);
      }
      Ln(logL, l, tilingData_.B_q);
      PipeBarrier<PIPE_V>();
      Add(l, logL, m, tilingData_.B_q);

      uint64_t offsetO = headOffsetO + (batchOffset_ + batch) *
                                           tilingData_.B_q * tilingData_.d;
      uint64_t offsetL = headOffsetL + (batchOffset_ + batch) * tilingData_.B_q;
      SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
      WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
      DataCopy(oGm_[offsetO], o, tilingData_.B_q * tilingData_.d);
      DataCopy(lGm_[offsetL], l, tilingData_.B_q);
      SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
      WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
  }
}

__aicore__ inline void FlashAttentionFwdKernel::InnerCycle(
    int32_t j, uint32_t batch, uint64_t batchOffset, uint64_t headOffsetQ,
    uint64_t headOffsetK, uint64_t headOffsetV) {
  uint64_t offsetQ =
      headOffsetQ + (batchOffset + batch) * tilingData_.B_q * tilingData_.d;
  uint64_t offsetK = headOffsetK + j * tilingData_.B_k * tilingData_.d;
  uint64_t offsetV = headOffsetV + j * tilingData_.B_k * tilingData_.d;

  LocalTensor<uint8_t> mmWsQK = mmWsQKBuf_.Get<uint8_t>();
  LocalTensor<half> sLocal = sQue_.AllocTensor<half>();

  mmQK_.SetLocalWorkspace(mmWsQK);
  mmQK_.SetTensorA(qGm_[offsetQ], false);
  mmQK_.SetTensorB(kGm_[offsetK], true);
  mmQK_.Iterate();
  mmQK_.GetTensorC(sLocal);
  sQue_.EnQue(sLocal);

  Compute(offsetV, j);
}

__aicore__ inline void FlashAttentionFwdKernel::Compute(uint64_t offsetV,
                                                        int32_t j) {
  LocalTensor<half> rowMaxLocal = rowMaxBuf_.Get<half>();
  LocalTensor<half> reduceLocal = reduceBuf_.Get<half>();
  LocalTensor<half> mLocal = mBuf_.Get<half>();
  LocalTensor<half> prevMLocal = prevMBuf_.Get<half>();
  LocalTensor<half> sumLocal = rowSumBuf_.Get<half>();
  LocalTensor<half> scaleLocal = scaleBuf_.Get<half>();

  half scale = tilingData_.scale;
  LocalTensor<half> sLocal = sQue_.DeQue<half>();
  Muls(sLocal, sLocal, scale, tilingData_.B_q * tilingData_.B_k);
  PipeBarrier<PIPE_V>();

  for (int32_t i = 0; i < tilingData_.B_q; ++i) {
    ReduceMax(rowMaxLocal[i], sLocal[i * tilingData_.B_k], reduceLocal,
              tilingData_.B_k);
  }

  if (j == 0) {
    Adds(mLocal, rowMaxLocal, static_cast<half>(0.0f), tilingData_.B_q);
  } else {
    Adds(prevMLocal, mLocal, static_cast<half>(0.0f), tilingData_.B_q);
    Max(mLocal, mLocal, rowMaxLocal, tilingData_.B_q);
    PipeBarrier<PIPE_V>();
    Sub(scaleLocal, prevMLocal, mLocal, tilingData_.B_q);
    PipeBarrier<PIPE_V>();
    Exp(scaleLocal, scaleLocal, tilingData_.B_q);
  }

  SetFlag<HardEvent::V_S>(EVENT_ID0);
  WaitFlag<HardEvent::V_S>(EVENT_ID0);

  for (int32_t i = 0; i < tilingData_.B_q; ++i) {
    half m = mLocal.GetValue(i);
    Adds(sLocal[i * tilingData_.B_k], sLocal[i * tilingData_.B_k],
         static_cast<half>(-static_cast<float>(m)), tilingData_.B_k);
  }

  LocalTensor<half> pLocal = pQue_.AllocTensor<half>();
  Exp(pLocal, sLocal, tilingData_.B_q * tilingData_.B_k);
  sQue_.FreeTensor(sLocal);
  pQue_.EnQue(pLocal);

  LocalTensor<half> pCube = pQue_.DeQue<half>();

  for (int32_t i = 0; i < tilingData_.B_q; ++i) {
    ReduceSum(sumLocal[i], pCube[i * tilingData_.B_k], reduceLocal,
              tilingData_.B_k);
  }

  LocalTensor<half> l = L.Get<half>();
  LocalTensor<half> o = O.Get<half>();

  if (j == 0) {
    Adds(l, sumLocal, static_cast<half>(0.0f), tilingData_.B_q);
  } else {
    Mul(l, scaleLocal, l, tilingData_.B_q);
    PipeBarrier<PIPE_V>();
    Add(l, l, sumLocal, tilingData_.B_q);

    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);

    for (int32_t i = 0; i < tilingData_.B_q; ++i) {
      half scl = scaleLocal.GetValue(i);
      Muls(o[i * tilingData_.d], o[i * tilingData_.d], scl, tilingData_.d);
    }
  }

  // 2. Vector 计算完毕后，安全下发 P @ V -> pvRes (Cube)
  LocalTensor<half> pvRes = pvResQue_.AllocTensor<half>();
  LocalTensor<uint8_t> mmWsPV = mmWsPVBuf_.Get<uint8_t>();
  mmPV_.SetLocalWorkspace(mmWsPV);
  mmPV_.SetTensorA(pCube, false);
  mmPV_.SetTensorB(vGm_[offsetV], false);
  mmPV_.Iterate();

  // 3. 等待并取出 pvRes，累加到 o
  mmPV_.GetTensorC(pvRes);
  pQue_.FreeTensor(pCube);
  pvResQue_.EnQue(pvRes);
  LocalTensor<half> pvLocal = pvResQue_.DeQue<half>();
  if (j == 0) {
    Adds(o, pvLocal, static_cast<half>(0.0f), tilingData_.B_q * tilingData_.d);
  } else {
    Add(o, o, pvLocal, tilingData_.B_q * tilingData_.d);
  }
  pvResQue_.FreeTensor(pvLocal);
}

extern "C" __global__ __aicore__ void
flash_attention_fwd_kernel(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR o,
                           GM_ADDR l, GM_ADDR tiling) {
  if ASCEND_IS_AIV {
    return;
  }
  FlashAttentionFwdKernel op;
  op.Init(q, k, v, o, l, tiling);
  op.Process();
  op.End();
}
