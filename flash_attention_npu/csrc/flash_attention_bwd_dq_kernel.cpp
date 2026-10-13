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
  using A_TYPE_QK = MatmulType<TPosition::VECIN, CubeFormat::ND, half, false>;
  using B_TYPE_QK = MatmulType<TPosition::VECIN, CubeFormat::ND, half, true>;
  using C_TYPE_QK = MatmulType<TPosition::VECIN, CubeFormat::ND, half>;
  using A_TYPE_dOV = MatmulType<TPosition::VECIN, CubeFormat::ND, half, false>;
  using B_TYPE_dOV = MatmulType<TPosition::VECIN, CubeFormat::ND, half, true>;
  using C_TYPE_dOV = MatmulType<TPosition::VECIN, CubeFormat::ND, half>;
  using A_TYPE_dSK = MatmulType<TPosition::VECIN, CubeFormat::ND, half, false>;
  using B_TYPE_dSK = MatmulType<TPosition::VECIN, CubeFormat::ND, half, false>;
  using C_TYPE_dSK = MatmulType<TPosition::VECIN, CubeFormat::ND, half>;

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
  __aicore__ inline void CopyInAndInit(uint32_t batch);
  __aicore__ inline void InnerCycle(uint32_t i);
  __aicore__ inline void InnerCycle_CopyIn(uint32_t i);
  __aicore__ __attribute__((noinline)) void InnerCycle_Compute(uint32_t i);

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
  TBuf<TPosition::VECCALC> qBuf_;
  TBuf<TPosition::VECCALC> lBuf_;
  TBuf<TPosition::VECCALC> dBuf_;
  TBuf<TPosition::VECCALC> doBuf_;
  TBuf<TPosition::VECCALC> dqBuf_;
  TBuf<TPosition::VECCALC> pBuf_;
  TBuf<TPosition::VECCALC> tmpMmResBuf_;
  TBuf<TPosition::VECCALC> dpBuf_;
  TBuf<TPosition::VECCALC> dsBuf_;
  TBuf<TPosition::VECCALC> scalarBuf_;
  TQue<TPosition::VECIN, FLASH_QUEUE_DEPTH> kQue_;
  TQue<TPosition::VECIN, FLASH_QUEUE_DEPTH> vQue_;
  TQue<TPosition::VECIN, FLASH_QUEUE_DEPTH> sQue_;
  Matmul<A_TYPE_QK, B_TYPE_QK, C_TYPE_QK> mmQK_;
  Matmul<A_TYPE_dOV, B_TYPE_dOV, C_TYPE_dOV> mmdOV_;
  Matmul<A_TYPE_dSK, B_TYPE_dSK, C_TYPE_dSK> mmdSK_;
  FlashAttentionBwd_dQ_TilingData tilingData_;
  uint32_t batchNum_;
  uint32_t batchOffset_;
};

__aicore__ inline void
FlashAttentionBwd_dQ_Kernel::Init(GM_ADDR Q, GM_ADDR O, GM_ADDR dO, GM_ADDR K,
                                  GM_ADDR V, GM_ADDR L, GM_ADDR D, GM_ADDR dQ,
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

  qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(Q),
                       tilingData_.N_q * tilingData_.d);
  oGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(O),
                       tilingData_.N_q * tilingData_.d);
  doGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(dO),
                        tilingData_.N_q * tilingData_.d);
  kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(K),
                       tilingData_.N_k * tilingData_.d);
  vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(V),
                       tilingData_.N_k * tilingData_.d);
  lGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(L), tilingData_.N_q);
  dGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(D), tilingData_.N_q);
  dqGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(dQ),
                        tilingData_.N_q * tilingData_.d);

  pipe_.InitBuffer(kQue_, FLASH_QUEUE_DEPTH,
                   tilingData_.B_k * tilingData_.d * sizeof(half));
  pipe_.InitBuffer(vQue_, FLASH_QUEUE_DEPTH,
                   tilingData_.B_k * tilingData_.d * sizeof(half));
  pipe_.InitBuffer(sQue_, FLASH_QUEUE_DEPTH,
                   tilingData_.B_q * tilingData_.B_k * sizeof(half));

  pipe_.InitBuffer(qBuf_, tilingData_.B_q * tilingData_.d * sizeof(half));
  pipe_.InitBuffer(doBuf_, tilingData_.B_q * tilingData_.d * sizeof(half));
  pipe_.InitBuffer(dqBuf_, tilingData_.B_q * tilingData_.d * sizeof(half));
  pipe_.InitBuffer(lBuf_, tilingData_.B_q * sizeof(half));
  pipe_.InitBuffer(dBuf_, tilingData_.B_q * sizeof(half));
  pipe_.InitBuffer(pBuf_, tilingData_.B_q * tilingData_.B_k * sizeof(half));
  pipe_.InitBuffer(tmpMmResBuf_, tilingData_.B_q * tilingData_.d * sizeof(half));
  pipe_.InitBuffer(dpBuf_, tilingData_.B_q * tilingData_.B_k * sizeof(half));
  pipe_.InitBuffer(dsBuf_, tilingData_.B_q * tilingData_.B_k * sizeof(half));
  pipe_.InitBuffer(scalarBuf_, 32);

  mmQK_.Init(&tilingData_.cubeTilingQK, &pipe_);
  mmQK_.SetOrgShape(tilingData_.cubeTilingQK.M, tilingData_.cubeTilingQK.N,
                    tilingData_.cubeTilingQK.Ka, tilingData_.cubeTilingQK.Kb);

  mmdOV_.Init(&tilingData_.cubeTilingdOV, &pipe_);
  mmdOV_.SetOrgShape(tilingData_.cubeTilingdOV.M, tilingData_.cubeTilingdOV.N,
                     tilingData_.cubeTilingdOV.Ka, tilingData_.cubeTilingdOV.Kb);

  mmdSK_.Init(&tilingData_.cubeTilingdSK, &pipe_);
  mmdSK_.SetOrgShape(tilingData_.cubeTilingdSK.M, tilingData_.cubeTilingdSK.N,
                     tilingData_.cubeTilingdSK.Ka, tilingData_.cubeTilingdSK.Kb);
}

__aicore__ inline void FlashAttentionBwd_dQ_Kernel::Process() {
  uint32_t T_k = (tilingData_.N_k + tilingData_.B_k - 1) / tilingData_.B_k;
  for (uint32_t batch = 0; batch < batchNum_; ++batch) {
    CopyInAndInit(batch);
    for (uint32_t i = 0; i < T_k; ++i) {
      InnerCycle(i);
    }
    uint64_t offset = (batchOffset_ + batch) * tilingData_.d * tilingData_.B_q;
    LocalTensor<half> dq = dqBuf_.Get<half>();
    DataCopy(dqGm_[offset], dq, tilingData_.B_q * tilingData_.d);
    PipeBarrier<PIPE_MTE3>();
  }
}

__aicore__ inline void
FlashAttentionBwd_dQ_Kernel::CopyInAndInit(uint32_t batch) {
  LocalTensor<half> qLocal = qBuf_.Get<half>();
  LocalTensor<half> doLocal = doBuf_.Get<half>();
  LocalTensor<half> dqLocal = dqBuf_.Get<half>();
  LocalTensor<half> lLocal = lBuf_.Get<half>();
  LocalTensor<half> dLocal = dBuf_.Get<half>();

  Duplicate(dqLocal, static_cast<half>(0.0f), tilingData_.B_q * tilingData_.d);

  uint64_t offset = (batchOffset_ + batch) * tilingData_.B_q * tilingData_.d;
  uint64_t offset1D = (batchOffset_ + batch) * tilingData_.B_q;
  DataCopy(qLocal, qGm_[offset], tilingData_.B_q * tilingData_.d);
  DataCopy(doLocal, doGm_[offset], tilingData_.B_q * tilingData_.d);
  DataCopy(lLocal, lGm_[offset1D], tilingData_.B_q);
  DataCopy(dLocal, dGm_[offset1D], tilingData_.B_q);
  PipeBarrier<PIPE_MTE2>();
}

__aicore__ inline void FlashAttentionBwd_dQ_Kernel::InnerCycle(uint32_t i) {
  InnerCycle_CopyIn(i);
  InnerCycle_Compute(i);
}

__aicore__ inline void
FlashAttentionBwd_dQ_Kernel::InnerCycle_CopyIn(uint32_t i) {
  LocalTensor<half> kLocal = kQue_.AllocTensor<half>();
  LocalTensor<half> vLocal = vQue_.AllocTensor<half>();

  uint64_t offset2D = i * tilingData_.B_k * tilingData_.d;
  DataCopy(kLocal, kGm_[offset2D], tilingData_.B_k * tilingData_.d);
  DataCopy(vLocal, vGm_[offset2D], tilingData_.B_k * tilingData_.d);

  kQue_.EnQue(kLocal);
  vQue_.EnQue(vLocal);
}

__aicore__ __attribute__((noinline)) void
FlashAttentionBwd_dQ_Kernel::InnerCycle_Compute(uint32_t i) {
  LocalTensor<half> kLocal = kQue_.DeQue<half>();
  LocalTensor<half> vLocal = vQue_.DeQue<half>();
  LocalTensor<half> lLocal = lBuf_.Get<half>();
  LocalTensor<half> dLocal = dBuf_.Get<half>();
  LocalTensor<half> qLocal = qBuf_.Get<half>();
  LocalTensor<half> doLocal = doBuf_.Get<half>();
  LocalTensor<half> dqLocal = dqBuf_.Get<half>();
  LocalTensor<half> pLocal = pBuf_.Get<half>();
  LocalTensor<half> dpLocal = dpBuf_.Get<half>();
  LocalTensor<half> dsLocal = dsBuf_.Get<half>();
  LocalTensor<half> sMatmul = sQue_.AllocTensor<half>();
  LocalTensor<half> tmpMmRes = tmpMmResBuf_.Get<half>();
  LocalTensor<half> scalar = scalarBuf_.Get<half>();

  mmQK_.SetTensorA(qLocal, false);
  mmQK_.SetTensorB(kLocal, true);
  mmQK_.IterateAll(sMatmul);
  sQue_.EnQue(sMatmul);
  LocalTensor<half> sLocal = sQue_.DeQue<half>();
  Muls(sLocal, sLocal, static_cast<half>(tilingData_.scale),
       tilingData_.B_q * tilingData_.B_k);
  for (uint32_t r = 0; r < tilingData_.B_q; r++) {
    half l = lLocal.GetValue(r);
    uint64_t offset = r * tilingData_.B_k;
    Adds(sLocal[offset], sLocal[offset],
         static_cast<half>(-static_cast<float>(l)), tilingData_.B_k);
  }

  Exp(pLocal, sLocal, tilingData_.B_q * tilingData_.B_k);
  sQue_.FreeTensor(sLocal);

  mmdOV_.SetTensorA(doLocal, false);
  mmdOV_.SetTensorB(vLocal, true);
  mmdOV_.IterateAll(dpLocal);
  vQue_.FreeTensor(vLocal);
  for (uint32_t r = 0; r < tilingData_.B_q; r++) {
    half d = dLocal.GetValue(r);
    uint64_t offset = r * tilingData_.B_k;
    Adds(dpLocal[offset], dpLocal[offset],
         static_cast<half>(-static_cast<float>(d)), tilingData_.B_k);
  }

  Mul(dsLocal, pLocal, dpLocal, tilingData_.B_q * tilingData_.B_k);

  mmdSK_.SetTensorA(dsLocal, false);
  mmdSK_.SetTensorB(kLocal, false);
  mmdSK_.IterateAll(tmpMmRes);
  Muls(tmpMmRes, tmpMmRes, static_cast<half>(tilingData_.scale),
       tilingData_.d * tilingData_.B_q);
  Add(dqLocal, dqLocal, tmpMmRes, tilingData_.d * tilingData_.B_q);
  kQue_.FreeTensor(kLocal);
}

extern "C" __global__ __aicore__ void
flash_attention_bwd_dq_kernel(GM_ADDR q, GM_ADDR o, GM_ADDR do_grad, GM_ADDR k,
                              GM_ADDR v, GM_ADDR l, GM_ADDR d, GM_ADDR dq,
                              GM_ADDR tiling) {
  FlashAttentionBwd_dQ_Kernel op;
  op.Init(q, o, do_grad, k, v, l, d, dq, tiling);
  op.Process();
  op.End();
}

