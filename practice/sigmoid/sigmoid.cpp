#include "kernel_operator.h"
#include <cstdint>

constexpr uint32_t BUFFER_NUM = 2; // tensor num for each queue
constexpr uint32_t QUEUE_DEPTH = 2;

struct SigmoidCustomTilingData {
  uint32_t totalLength;
  uint32_t tileNum;
};

class KernelSigmoid {
public:
  __aicore__ inline KernelSigmoid() {}
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength,
                              uint32_t tileNum);
  __aicore__ inline void Process();

private:
  __aicore__ inline void CopyIn(int32_t progress);
  __aicore__ inline void Compute(int32_t progress);
  __aicore__ inline void CopyOut(int32_t progress);

private:
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::TPosition::VECIN, QUEUE_DEPTH> inQueue;
  AscendC::TQue<AscendC::TPosition::VECOUT, QUEUE_DEPTH> outQueue;
  AscendC::GlobalTensor<float> xGm;
  AscendC::GlobalTensor<float> yGm;
  uint32_t blockLength;
  uint32_t tileNum;
  uint32_t tileLength;
};

__aicore__ inline void KernelSigmoid::Init(GM_ADDR x, GM_ADDR y,
                                           uint32_t totalLength,
                                           uint32_t tileNum) {
  this->blockLength = totalLength / AscendC::GetBlockNum();
  this->tileNum = tileNum;
  this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
  pipe.InitBuffer(inQueue, BUFFER_NUM, this->tileLength * sizeof(float));
  pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileLength * sizeof(float));
  uint32_t off_set = this->blockLength * AscendC::GetBlockIdx();
  xGm.SetGlobalBuffer((__gm__ float *)x + off_set, this->blockLength);
  yGm.SetGlobalBuffer((__gm__ float *)y + off_set, this->blockLength);
}

__aicore__ inline void KernelSigmoid::Process() {
  int32_t loopCnt = this->tileNum * BUFFER_NUM;
  for (int32_t i = 0; i < loopCnt; ++i) {
    CopyIn(i);
    Compute(i);
    CopyOut(i);
  }
}

__aicore__ inline void KernelSigmoid::CopyIn(int32_t progress) {
  AscendC::LocalTensor<float> xLocal = inQueue.AllocTensor<float>();
  AscendC::DataCopy(xLocal, xGm[progress * tileLength], tileLength);
  inQueue.EnQue(xLocal);
}

__aicore__ inline void KernelSigmoid::Compute(int32_t progress) {
  AscendC::LocalTensor<float> xLocal = inQueue.DeQue<float>();
  AscendC::LocalTensor<float> yLocal = outQueue.AllocTensor<float>();
  AscendC::Muls(xLocal, xLocal, -1.0f, tileLength);
  AscendC::Exp(xLocal, xLocal, tileLength);
  AscendC::Adds(xLocal, xLocal, 1.0f, tileLength);
  AscendC::Duplicate(yLocal, 1.0f, tileLength);
  AscendC::Div(yLocal, yLocal, xLocal, tileLength);
  outQueue.EnQue(yLocal);
  inQueue.FreeTensor(xLocal);
}

__aicore__ inline void KernelSigmoid::CopyOut(int32_t progress) {
  AscendC::LocalTensor<float> yLocal = outQueue.DeQue<float>();
  AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
  outQueue.FreeTensor(yLocal);
}

extern "C" __global__ __aicore__ void
sigmoid_custom(GM_ADDR x, GM_ADDR y, SigmoidCustomTilingData tiling) {
  KernelSigmoid op;
  op.Init(x, y, tiling.totalLength, tiling.tileNum);
  op.Process();
}

void sigmoid_custom_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y,
                       SigmoidCustomTilingData tiling) {
  sigmoid_custom<<<blockDim, nullptr, stream>>>(x, y, &tiling);
}
