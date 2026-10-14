// clang-format off
#include "kernel_operator.h"
#include "flash_attention_fwd_tiling.h"
#include "lib/matmul_intf.h"
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
  using C_TYPE_QK = MatmulType<TPosition::VECIN, CubeFormat::ND, half>;
  using A_TYPE_PV = MatmulType<TPosition::VECOUT, CubeFormat::ND, half, false>;
  using B_TYPE_PV = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
  using C_TYPE_PV = MatmulType<TPosition::VECIN, CubeFormat::ND, half>;

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
                                    uint64_t batchOffset);
  __aicore__ inline void Compute(uint64_t offsetV);

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
  TQue<TPosition::VECIN, FLASH_QUEUE_DEPTH> sQue_;
  TQue<TPosition::VECOUT, FLASH_QUEUE_DEPTH> pQue_;
  TBuf<TPosition::VECCALC> L;
  TBuf<TPosition::VECCALC> O;
  TBuf<TPosition::VECCALC> mBuf_;
  TBuf<TPosition::VECCALC> prevMBuf_;
  TBuf<TPosition::VECCALC> rowMaxBuf_;
  TBuf<TPosition::VECCALC> rowSumBuf_;
  TBuf<TPosition::VECCALC> scaleBuf_;
  TBuf<TPosition::VECCALC> reduceBuf_;
  TBuf<TPosition::VECCALC> pvResBuf_;
  TBuf<TPosition::VECCALC> scalarBuf_;
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

  qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(Q),
                       tilingData_.N_q * tilingData_.d);
  kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(K),
                       tilingData_.N_k * tilingData_.d);
  vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(V),
                       tilingData_.N_k * tilingData_.d);
  oGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(O),
                       tilingData_.N_q * tilingData_.d);
  lGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(L), tilingData_.N_q);

  pipe_.InitBuffer(sQue_, FLASH_BUFFER_SIZE,
                   tilingData_.B_q * tilingData_.B_k * sizeof(half));
  pipe_.InitBuffer(pQue_, FLASH_BUFFER_SIZE,
                   tilingData_.B_k * tilingData_.B_q * sizeof(half));
  pipe_.InitBuffer(this->L, tilingData_.B_q * sizeof(half));
  pipe_.InitBuffer(this->O, tilingData_.B_q * tilingData_.d * sizeof(half));
  pipe_.InitBuffer(mBuf_, tilingData_.B_q * sizeof(half));
  pipe_.InitBuffer(prevMBuf_, tilingData_.B_q * sizeof(half));
  pipe_.InitBuffer(rowMaxBuf_, tilingData_.B_q * sizeof(half));
  pipe_.InitBuffer(rowSumBuf_, tilingData_.B_q * sizeof(half));
  pipe_.InitBuffer(scaleBuf_, tilingData_.B_q * sizeof(half));
  pipe_.InitBuffer(reduceBuf_, 4096);
  pipe_.InitBuffer(pvResBuf_, tilingData_.B_q * tilingData_.d * sizeof(half));
  pipe_.InitBuffer(scalarBuf_, 32);

  mmQK_.Init(&tilingData_.cubeTilingQK, &pipe_);
  mmQK_.SetOrgShape(tilingData_.cubeTilingQK.M, tilingData_.cubeTilingQK.N,
                    tilingData_.cubeTilingQK.Ka, tilingData_.cubeTilingQK.Kb);

  mmPV_.Init(&tilingData_.cubeTilingPV, &pipe_);
  mmPV_.SetOrgShape(tilingData_.cubeTilingPV.M, tilingData_.cubeTilingPV.N,
                    tilingData_.cubeTilingPV.Ka, tilingData_.cubeTilingPV.Kb);
}

__aicore__ inline void FlashAttentionFwdKernel::Process() {
  for (uint32_t batch = 0; batch < batchNum_; ++batch) {
    LocalTensor<half> oLocal = this->O.Get<half>();
    Duplicate(oLocal, static_cast<half>(0.0f), tilingData_.B_q * tilingData_.d);
    LocalTensor<half> lLocal = this->L.Get<half>();
    Duplicate(lLocal, static_cast<half>(0.0f), tilingData_.B_q);
    LocalTensor<half> mLocal = mBuf_.Get<half>();
    Duplicate(mLocal, static_cast<half>(-65504.0f), tilingData_.B_q);

    int32_t T_k = (tilingData_.N_k + tilingData_.B_k - 1) / tilingData_.B_k;
    for (int32_t j = 0; j < T_k; ++j) {
      InnerCycle(j, batch, batchOffset_);
    }
    LocalTensor<half> o = O.Get<half>();
    LocalTensor<half> l = L.Get<half>();
    LocalTensor<half> m = mBuf_.Get<half>();
    LocalTensor<half> logL = rowSumBuf_.Get<half>();

    for (int32_t i = 0; i < tilingData_.B_q; ++i) {
      half scale = static_cast<half>(1.0f / static_cast<float>(l.GetValue(i)));
      Muls(o[i * tilingData_.d], o[i * tilingData_.d], scale, tilingData_.d);
    }
    Ln(logL, l, tilingData_.B_q);
    Add(l, logL, m, tilingData_.B_q);
    uint64_t offsetO = (batchOffset_ + batch) * tilingData_.B_q * tilingData_.d;
    uint64_t offsetL = (batchOffset_ + batch) * tilingData_.B_q;
    DataCopy(oGm_[offsetO], o, tilingData_.B_q * tilingData_.d);
    DataCopy(lGm_[offsetL], l, tilingData_.B_q);
  }
}

__aicore__ inline void
FlashAttentionFwdKernel::InnerCycle(int32_t j, uint32_t batch,
                                    uint64_t batchOffset) {
  PRINTF("[FWD] InnerCycle start: j=%d, batch=%d\n", j, batch);
  uint64_t offsetQ = (batchOffset + batch) * tilingData_.B_q * tilingData_.d;
  uint64_t offsetK = j * tilingData_.B_k * tilingData_.d;
  uint64_t offsetV = offsetK;
  mmQK_.SetTensorA(qGm_[offsetQ], false);
  mmQK_.SetTensorB(kGm_[offsetK], true);

  LocalTensor<half> sLocal = sQue_.AllocTensor<half>();
  mmQK_.IterateAll(sLocal);
  sQue_.EnQue(sLocal);
  PRINTF("[FWD] mmQK completed\n");

  Compute(offsetV);
  PRINTF("[FWD] Compute completed\n");
}

__aicore__ inline void FlashAttentionFwdKernel::Compute(uint64_t offsetV) {
  PRINTF("[FWD] Compute: DeQue sLocal\n");
  LocalTensor<half> sLocal = sQue_.DeQue<half>();
  LocalTensor<half> pLocal = pQue_.AllocTensor<half>();
  LocalTensor<half> rowMaxLocal = rowMaxBuf_.Get<half>();
  LocalTensor<half> reduceLocal = reduceBuf_.Get<half>();
  LocalTensor<half> mLocal = mBuf_.Get<half>();
  LocalTensor<half> prevMLocal = prevMBuf_.Get<half>();
  LocalTensor<half> sumLocal = rowSumBuf_.Get<half>();
  LocalTensor<half> scaleLocal = scaleBuf_.Get<half>();
  LocalTensor<half> scalarLocal = scalarBuf_.Get<half>();

  half scale = tilingData_.scale;
  Muls(sLocal, sLocal, scale, tilingData_.B_q * tilingData_.B_k);
  for (int32_t i = 0; i < tilingData_.B_q; ++i) {
    ReduceMax(scalarLocal, sLocal[i * tilingData_.B_k], reduceLocal,
              tilingData_.B_k);
    rowMaxLocal.SetValue(i, scalarLocal.GetValue(0));
  }
  Adds(prevMLocal, mLocal, static_cast<half>(0.0f), tilingData_.B_q);
  Max(mLocal, mLocal, rowMaxLocal, tilingData_.B_q);
  Sub(scaleLocal, prevMLocal, mLocal, tilingData_.B_q);
  Exp(scaleLocal, scaleLocal, tilingData_.B_q);
  for (int32_t i = 0; i < tilingData_.B_q; ++i) {
    half m = mLocal.GetValue(i);
    Adds(sLocal[i * tilingData_.B_k], sLocal[i * tilingData_.B_k],
         static_cast<half>(-static_cast<float>(m)), tilingData_.B_k);
  }
  Exp(pLocal, sLocal, tilingData_.B_q * tilingData_.B_k);
  sQue_.FreeTensor(sLocal);

  pQue_.EnQue(pLocal);
  LocalTensor<half> pMatmul = pQue_.DeQue<half>();

  PRINTF("[FWD] Starting mmPV\n");
  LocalTensor<half> pvRes = pvResBuf_.Get<half>();
  mmPV_.SetTensorA(pMatmul);
  mmPV_.SetTensorB(vGm_[offsetV], false);
  mmPV_.IterateAll(pvRes);
  PRINTF("[FWD] mmPV completed\n");

  for (int32_t i = 0; i < tilingData_.B_q; ++i) {
    ReduceSum(scalarLocal, pMatmul[i * tilingData_.B_k], reduceLocal,
              tilingData_.B_k);
    sumLocal.SetValue(i, scalarLocal.GetValue(0));
  }

  LocalTensor<half> l = L.Get<half>();
  LocalTensor<half> o = O.Get<half>();
  Mul(l, scaleLocal, l, tilingData_.B_q);
  Add(l, l, sumLocal, tilingData_.B_q);

  for (int32_t i = 0; i < tilingData_.B_q; ++i) {
    half scl = scaleLocal.GetValue(i);
    Muls(o[i * tilingData_.d], o[i * tilingData_.d], scl, tilingData_.d);
  }

  Add(o, o, pvRes, tilingData_.B_q * tilingData_.d);
  pQue_.FreeTensor(pMatmul);
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
