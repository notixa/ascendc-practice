#include "rmsnorm_kernel.h"
#include "kernel_operator.h"
#include "rmsnorm_tiling.h"
#include <cmath>
#include <cstdint>

constexpr uint32_t QUEUE_DEPTH = 2;
constexpr float EPS = 1e-5;

using namespace AscendC;

template <class dtype> class RMSNormKernel {
public:
  __aicore__ inline RMSNormKernel() {}
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                              const RMSNormTiling &tiling);
  __aicore__ inline void Process();

private:
  __aicore__ inline void CopyIn(int32_t process);
  __aicore__ inline void Compute(int32_t process);
  __aicore__ inline void CopyOut(int32_t process);

private:
  TPipe pipe;
  TQue<TPosition::VECIN, QUEUE_DEPTH> inQueX;
  TQue<TPosition::VECOUT, QUEUE_DEPTH> outQueZ;
  TBuf<TPosition::VECCALC> gammaBuf;
  TBuf<TPosition::VECCALC> rdcBuf;
  TBuf<TPosition::VECCALC> sumBuf;
  GlobalTensor<dtype> xGm;
  GlobalTensor<dtype> yGm;
  GlobalTensor<dtype> zGm;
  uint32_t rowNum;
  uint32_t hiddenSize;
};

template <class dtype>
__aicore__ inline void RMSNormKernel<dtype>::Init(GM_ADDR x, GM_ADDR y,
                                                  GM_ADDR z,
                                                  const RMSNormTiling &tiling) {
  this->hiddenSize = tiling.hiddenSize;
  uint32_t blockIdx = GetBlockIdx();
  bool fullCore = blockIdx < tiling.fullCoreNum;
  if (fullCore) {
    this->rowNum = tiling.rowsPerFullCore;
  } else {
    this->rowNum = tiling.rowsPerTailCore;
  }
  pipe.InitBuffer(inQueX, 2, hiddenSize * sizeof(dtype));
  pipe.InitBuffer(gammaBuf, hiddenSize * sizeof(dtype));
  pipe.InitBuffer(outQueZ, 2, hiddenSize * sizeof(dtype));
  pipe.InitBuffer(rdcBuf, tiling.reduceTBufBytes);
  pipe.InitBuffer(sumBuf, 256);

  uint32_t off_set = 0;
  if (fullCore) {
    off_set = blockIdx * hiddenSize * tiling.rowsPerFullCore;
  } else {
    off_set =
        tiling.fullCoreNum * hiddenSize * tiling.rowsPerFullCore +
        (blockIdx - tiling.fullCoreNum) * hiddenSize * tiling.rowsPerTailCore;
  }
  xGm.SetGlobalBuffer((__gm__ dtype *)x + off_set, this->rowNum * hiddenSize);
  yGm.SetGlobalBuffer((__gm__ dtype *)y, hiddenSize);
  zGm.SetGlobalBuffer((__gm__ dtype *)z + off_set, this->rowNum * hiddenSize);

  LocalTensor<dtype> yLocal = gammaBuf.Get<dtype>();
  DataCopy(yLocal, yGm[0], hiddenSize);
}

template <class dtype> __aicore__ inline void RMSNormKernel<dtype>::Process() {
  int32_t loopCnt = this->rowNum;
  for (int32_t i = 0; i < loopCnt; ++i) {
    CopyIn(i);
    Compute(i);
    CopyOut(i);
  }
}

template <class dtype>
__aicore__ inline void RMSNormKernel<dtype>::CopyIn(int32_t process) {
  LocalTensor<dtype> xLocal = inQueX.AllocTensor<dtype>();
  DataCopy(xLocal, xGm[process * hiddenSize], hiddenSize);
  inQueX.EnQue(xLocal);
}

template <class dtype>
__aicore__ inline void RMSNormKernel<dtype>::Compute(int32_t process) {
  LocalTensor<dtype> xLocal = inQueX.DeQue<dtype>();
  LocalTensor<dtype> yLocal = gammaBuf.Get<dtype>();
  LocalTensor<dtype> zLocal = outQueZ.AllocTensor<dtype>();
  LocalTensor<dtype> sum = sumBuf.Get<dtype>();
  LocalTensor<dtype> reduceBuf = rdcBuf.Get<dtype>();

  Mul(zLocal, xLocal, xLocal, hiddenSize);
  ReduceSum(sum, zLocal, reduceBuf, hiddenSize);
  Muls(sum, sum, static_cast<dtype>(1.0f / hiddenSize), 1);
  Adds(sum, sum, static_cast<dtype>(EPS), 1);
  Rsqrt(sum, sum, 1);

  BrcbRepeatParams brcbParams;
  Brcb(sum, sum, 1, brcbParams);
  
  constexpr uint32_t ELEM_PER_REP = 256 / sizeof(dtype);
  uint8_t repeatTimes = static_cast<uint8_t>(hiddenSize / ELEM_PER_REP);
  BinaryRepeatParams repeatParams(1, 1, 0, 8, 8, 0);
  if constexpr (std::is_same_v<dtype, float>) {
    Mul(zLocal, xLocal, sum, 64, repeatTimes, repeatParams);
  } else {
    uint64_t mask[2] = {0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL};
    Mul(zLocal, xLocal, sum, mask, repeatTimes, repeatParams);
  }
  Mul(zLocal, zLocal, yLocal, hiddenSize);

  outQueZ.EnQue(zLocal);
  inQueX.FreeTensor(xLocal);
}

template <class dtype>
__aicore__ inline void RMSNormKernel<dtype>::CopyOut(int32_t process) {
  LocalTensor<dtype> zLocal = outQueZ.DeQue<dtype>();
  DataCopy(zGm[process * hiddenSize], zLocal, hiddenSize);
  outQueZ.FreeTensor(zLocal);
}

#define INSTANTIATE_RMSNORM_KERNEL(NAME, DTYPE)                                \
  extern "C" __global__ __aicore__ void NAME(GM_ADDR x, GM_ADDR y, GM_ADDR z,  \
                                             RMSNormTiling tiling) {           \
    RMSNormKernel<DTYPE> op;                                                   \
    op.Init(x, y, z, tiling);                                                  \
    op.Process();                                                              \
  }

INSTANTIATE_RMSNORM_KERNEL(rmsnorm_fp32, float);
INSTANTIATE_RMSNORM_KERNEL(rmsnorm_fp16, half);
INSTANTIATE_RMSNORM_KERNEL(rmsnorm_bf16, bfloat16_t);

void RMSNorm_fp32_do(uint32_t blockDim, void *stream, void *x, void *y, void *z,
                     const RMSNormTiling &tiling) {
  rmsnorm_fp32<<<blockDim, nullptr, stream>>>(
      static_cast<GM_ADDR>(x), static_cast<GM_ADDR>(y), static_cast<GM_ADDR>(z),
      const_cast<RMSNormTiling *>(&tiling));
}

void RMSNorm_fp16_do(uint32_t blockDim, void *stream, void *x, void *y, void *z,
                     const RMSNormTiling &tiling) {
  rmsnorm_fp16<<<blockDim, nullptr, stream>>>(
      static_cast<GM_ADDR>(x), static_cast<GM_ADDR>(y), static_cast<GM_ADDR>(z),
      const_cast<RMSNormTiling *>(&tiling));
}

void RMSNorm_bf16_do(uint32_t blockDim, void *stream, void *x, void *y, void *z,
                     const RMSNormTiling &tiling) {
  rmsnorm_bf16<<<blockDim, nullptr, stream>>>(
      static_cast<GM_ADDR>(x), static_cast<GM_ADDR>(y), static_cast<GM_ADDR>(z),
      const_cast<RMSNormTiling *>(&tiling));
}
