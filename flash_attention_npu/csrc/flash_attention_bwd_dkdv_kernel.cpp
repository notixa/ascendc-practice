// clang-format off
#include "kernel_operator.h"
#include "flash_attention_bwd_tiling.h"
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

class FlashAttentionBwd_dKdV_Kernel {
  using A_TYPE_QK = MatmulType<TPosition::VECIN, CubeFormat::ND, half, false>;
  using B_TYPE_QK = MatmulType<TPosition::VECIN, CubeFormat::ND, half, true>;
  using C_TYPE_QK = MatmulType<TPosition::VECIN, CubeFormat::ND, half>;
  using A_TYPE_PdO = MatmulType<TPosition::VECOUT, CubeFormat::ND, half, true>;
  using B_TYPE_PdO = MatmulType<TPosition::VECIN, CubeFormat::ND, half, false>;
  using C_TYPE_PdO = MatmulType<TPosition::VECIN, CubeFormat::ND, half>;
  using A_TYPE_dOV = MatmulType<TPosition::VECIN, CubeFormat::ND, half, false>;
  using B_TYPE_dOV = MatmulType<TPosition::VECIN, CubeFormat::ND, half, true>;
  using C_TYPE_dOV = MatmulType<TPosition::VECIN, CubeFormat::ND, half>;
  using A_TYPE_dSQ = MatmulType<TPosition::VECIN, CubeFormat::ND, half, true>;
  using B_TYPE_dSQ = MatmulType<TPosition::VECIN, CubeFormat::ND, half, false>;
  using C_TYPE_dSQ = MatmulType<TPosition::VECIN, CubeFormat::ND, half>;

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
  __aicore__ inline void InnerCycle(uint32_t i);
  __aicore__ inline void InnerCycle_CopyIn(uint32_t i);
  __aicore__ inline void InnerCycle_Compute(uint32_t i);

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
  TBuf<TPosition::VECCALC> kBuf_;
  TBuf<TPosition::VECCALC> vBuf_;
  TBuf<TPosition::VECCALC> dkBuf_;
  TBuf<TPosition::VECCALC> dvBuf_;
  TBuf<TPosition::VECCALC> pBuf_;
  TBuf<TPosition::VECCALC> tmpMmResBuf_;
  TBuf<TPosition::VECCALC> dpBuf_;
  TBuf<TPosition::VECCALC> dsBuf_;
  TBuf<TPosition::VECCALC> scalarBuf_;
  TQue<TPosition::VECIN, FLASH_QUEUE_DEPTH> qQue_;
  TQue<TPosition::VECIN, FLASH_QUEUE_DEPTH> doQue_;
  TQue<TPosition::VECIN, FLASH_QUEUE_DEPTH> lQue_;
  TQue<TPosition::VECIN, FLASH_QUEUE_DEPTH> dQue_;
  TQue<TPosition::VECIN, FLASH_QUEUE_DEPTH> sQue_;
  Matmul<A_TYPE_QK, B_TYPE_QK, C_TYPE_QK> mmQK_;
  Matmul<A_TYPE_PdO, B_TYPE_PdO, C_TYPE_PdO> mmPdO_;
  Matmul<A_TYPE_dOV, B_TYPE_dOV, C_TYPE_dOV> mmdOV_;
  Matmul<A_TYPE_dSQ, B_TYPE_dSQ, C_TYPE_dSQ> mmdSQ_;
  FlashAttentionBwd_dKdV_TilingData tilingData_;
  uint32_t batchNum_;
  uint32_t batchOffset_;
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
  dkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(dK),
                        tilingData_.N_k * tilingData_.d);
  dvGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(dV),
                        tilingData_.N_k * tilingData_.d);

  pipe_.InitBuffer(qQue_, FLASH_QUEUE_DEPTH,
                   tilingData_.B_q * tilingData_.d * sizeof(half));
  pipe_.InitBuffer(doQue_, FLASH_QUEUE_DEPTH,
                   tilingData_.B_q * tilingData_.d * sizeof(half));
  pipe_.InitBuffer(lQue_, FLASH_QUEUE_DEPTH, tilingData_.B_q * sizeof(half));
  pipe_.InitBuffer(dQue_, FLASH_QUEUE_DEPTH, tilingData_.B_q * sizeof(half));
  pipe_.InitBuffer(sQue_, FLASH_QUEUE_DEPTH,
                   tilingData_.B_q * tilingData_.B_k * sizeof(half));

  pipe_.InitBuffer(kBuf_, tilingData_.B_k * tilingData_.d * sizeof(half));
  pipe_.InitBuffer(vBuf_, tilingData_.B_k * tilingData_.d * sizeof(half));
  pipe_.InitBuffer(dkBuf_, tilingData_.B_k * tilingData_.d * sizeof(half));
  pipe_.InitBuffer(dvBuf_, tilingData_.B_k * tilingData_.d * sizeof(half));
  pipe_.InitBuffer(pBuf_, tilingData_.B_q * tilingData_.B_k * sizeof(half));
  pipe_.InitBuffer(tmpMmResBuf_, tilingData_.B_k * tilingData_.d * sizeof(half));
  pipe_.InitBuffer(dpBuf_, tilingData_.B_q * tilingData_.B_k * sizeof(half));
  pipe_.InitBuffer(dsBuf_, tilingData_.B_q * tilingData_.B_k * sizeof(half));
  pipe_.InitBuffer(scalarBuf_, 32);

  mmQK_.Init(&tilingData_.cubeTilingQK, &pipe_);
  mmQK_.SetOrgShape(tilingData_.cubeTilingQK.M, tilingData_.cubeTilingQK.N,
                    tilingData_.cubeTilingQK.Ka, tilingData_.cubeTilingQK.Kb);

  mmPdO_.Init(&tilingData_.cubeTilingPdO, &pipe_);
  mmPdO_.SetOrgShape(tilingData_.cubeTilingPdO.M, tilingData_.cubeTilingPdO.N,
                     tilingData_.cubeTilingPdO.Ka, tilingData_.cubeTilingPdO.Kb);

  mmdOV_.Init(&tilingData_.cubeTilingdOV, &pipe_);
  mmdOV_.SetOrgShape(tilingData_.cubeTilingdOV.M, tilingData_.cubeTilingdOV.N,
                     tilingData_.cubeTilingdOV.Ka, tilingData_.cubeTilingdOV.Kb);

  mmdSQ_.Init(&tilingData_.cubeTilingdSQ, &pipe_);
  mmdSQ_.SetOrgShape(tilingData_.cubeTilingdSQ.M, tilingData_.cubeTilingdSQ.N,
                     tilingData_.cubeTilingdSQ.Ka, tilingData_.cubeTilingdSQ.Kb);
}
__aicore__ inline void FlashAttentionBwd_dKdV_Kernel::Process() {
  uint32_t T_q = (tilingData_.N_q + tilingData_.B_q - 1) / tilingData_.B_q;
  for (uint32_t batch = 0; batch < batchNum_; ++batch) {
    CopyInAndInit(batch);
    for (uint32_t i = 0; i < T_q; ++i) {
      InnerCycle(i);
    }
    uint64_t offset = (batchOffset_ + batch) * tilingData_.d * tilingData_.B_k;
    LocalTensor<half> dk = dkBuf_.Get<half>();
    LocalTensor<half> dv = dvBuf_.Get<half>();
    DataCopy(dkGm_[offset], dk, tilingData_.B_k * tilingData_.d);
    DataCopy(dvGm_[offset], dv, tilingData_.B_k * tilingData_.d);
    PipeBarrier<PIPE_MTE3>();
  }
}

__aicore__ inline void
FlashAttentionBwd_dKdV_Kernel::CopyInAndInit(uint32_t batch) {
  LocalTensor<half> kLocal = kBuf_.Get<half>();
  LocalTensor<half> vLocal = vBuf_.Get<half>();
  LocalTensor<half> dkLocal = dkBuf_.Get<half>();
  LocalTensor<half> dvLocal = dvBuf_.Get<half>();
  Duplicate(dkLocal, static_cast<half>(0.0f), tilingData_.B_k * tilingData_.d);
  Duplicate(dvLocal, static_cast<half>(0.0f), tilingData_.B_k * tilingData_.d);
  uint64_t offset = (batchOffset_ + batch) * tilingData_.B_k * tilingData_.d;
  DataCopy(kLocal, kGm_[offset], tilingData_.B_k * tilingData_.d);
  DataCopy(vLocal, vGm_[offset], tilingData_.B_k * tilingData_.d);
  PipeBarrier<PIPE_MTE2>();
}

__aicore__ inline void FlashAttentionBwd_dKdV_Kernel::InnerCycle(uint32_t i) {
  InnerCycle_CopyIn(i);
  InnerCycle_Compute(i);
}

__aicore__ inline void
FlashAttentionBwd_dKdV_Kernel::InnerCycle_CopyIn(uint32_t i) {
  LocalTensor<half> qLocal = qQue_.AllocTensor<half>();
  LocalTensor<half> doLocal = doQue_.AllocTensor<half>();
  LocalTensor<half> lLocal = lQue_.AllocTensor<half>();
  LocalTensor<half> dLocal = dQue_.AllocTensor<half>();
  uint64_t offset2D = i * tilingData_.B_q * tilingData_.d;
  uint64_t offset1D = i * tilingData_.B_q;
  DataCopy(qLocal, qGm_[offset2D], tilingData_.B_q * tilingData_.d);
  DataCopy(doLocal, doGm_[offset2D], tilingData_.B_q * tilingData_.d);
  DataCopy(lLocal, lGm_[offset1D], tilingData_.B_q);
  DataCopy(dLocal, dGm_[offset1D], tilingData_.B_q);
  qQue_.EnQue(qLocal);
  doQue_.EnQue(doLocal);
  lQue_.EnQue(lLocal);
  dQue_.EnQue(dLocal);
}

__aicore__ inline void
FlashAttentionBwd_dKdV_Kernel::InnerCycle_Compute(uint32_t i) {
  LocalTensor<half> qLocal = qQue_.DeQue<half>();
  LocalTensor<half> doLocal = doQue_.DeQue<half>();
  LocalTensor<half> lLocal = lQue_.DeQue<half>();
  LocalTensor<half> dLocal = dQue_.DeQue<half>();
  LocalTensor<half> kLocal = kBuf_.Get<half>();
  LocalTensor<half> vLocal = vBuf_.Get<half>();
  LocalTensor<half> dkLocal = dkBuf_.Get<half>();
  LocalTensor<half> dvLocal = dvBuf_.Get<half>();
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
  lQue_.FreeTensor(lLocal);
  Exp(pLocal, sLocal, tilingData_.B_q * tilingData_.B_k);
  sQue_.FreeTensor(sLocal);

  mmPdO_.SetTensorA(pLocal, true);
  mmPdO_.SetTensorB(doLocal, false);
  mmPdO_.IterateAll(tmpMmRes);
  Add(dvLocal, dvLocal, tmpMmRes, tilingData_.d * tilingData_.B_k);

  mmdOV_.SetTensorA(doLocal, false);
  mmdOV_.SetTensorB(vLocal, true);
  mmdOV_.IterateAll(dpLocal);
  doQue_.FreeTensor(doLocal);
  for (uint32_t r = 0; r < tilingData_.B_q; r++) {
    half d = dLocal.GetValue(r);
    uint64_t offset = r * tilingData_.B_k;
    Adds(dpLocal[offset], dpLocal[offset],
         static_cast<half>(-static_cast<float>(d)), tilingData_.B_k);
  }
  dQue_.FreeTensor(dLocal);
  Mul(dsLocal, pLocal, dpLocal, tilingData_.B_q * tilingData_.B_k);

  mmdSQ_.SetTensorA(dsLocal, true);
  mmdSQ_.SetTensorB(qLocal, false);
  mmdSQ_.IterateAll(tmpMmRes);
  Muls(tmpMmRes, tmpMmRes, static_cast<half>(tilingData_.scale),
       tilingData_.d * tilingData_.B_k);
  Add(dkLocal, dkLocal, tmpMmRes, tilingData_.d * tilingData_.B_k);
  qQue_.FreeTensor(qLocal);
}

extern "C" __global__ __aicore__ void
flash_attention_bwd_dkdv_kernel(GM_ADDR q, GM_ADDR o, GM_ADDR do_grad, GM_ADDR k,
                                GM_ADDR v, GM_ADDR l, GM_ADDR d, GM_ADDR dk,
                                GM_ADDR dv, GM_ADDR tiling) {
  if ASCEND_IS_AIV {
    return;
  }
  FlashAttentionBwd_dKdV_Kernel op;
  op.Init(q, o, do_grad, k, v, l, d, dk, dv, tiling);
  op.Process();
  op.End();
}
