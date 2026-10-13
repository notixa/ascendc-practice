/**
 * @file vector_add.cpp
 *
 * Ascend C 矢量加法算子实现 (Device 侧)
 */
#include "kernel_operator.h"
#include <cstdint>

constexpr uint32_t BUFFER_NUM = 2;

struct VectorSubTilingData {
  uint32_t totalLength;
  uint32_t tileNum;
};

class KernelSub {
public:
  __aicore__ inline KernelSub() {}
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                              uint32_t totalLength, uint32_t tileNum);
  __aicore__ inline void Process();

private:
  __aicore__ inline void CopyIn(int32_t progress);
  __aicore__ inline void Compute(int32_t progress);
  __aicore__ inline void CopyOut(int32_t progress);

private:
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
  AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
  AscendC::GlobalTensor<float> xGm;
  AscendC::GlobalTensor<float> yGm;
  AscendC::GlobalTensor<float> zGm;

  uint32_t blockLength;
  uint32_t tileNum;
  uint32_t tileLength;
};

__aicore__ inline void KernelSub::Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                       uint32_t totalLength, uint32_t tileNum) {
  this->blockLength = totalLength / AscendC::GetBlockNum();
  this->tileNum = tileNum;
  this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
  pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
  pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
  pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(float));
  uint32_t off_set = blockLength * AscendC::GetBlockIdx();
  xGm.SetGlobalBuffer((__gm__ float *)x + off_set, this->blockLength);
  yGm.SetGlobalBuffer((__gm__ float *)y + off_set, this->blockLength);
  zGm.SetGlobalBuffer((__gm__ float *)z + off_set, this->blockLength);
}

__aicore__ inline void KernelSub::Process() {
  int32_t loopCnt = tileNum * BUFFER_NUM;
  for (int32_t i = 0; i < loopCnt; ++i) {
    CopyIn(i);
    Compute(i);
    CopyOut(i);
  }
}

__aicore__ inline void KernelSub::CopyIn(int32_t progress) {
  AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
  AscendC::LocalTensor<float> yLocal = inQueueY.AllocTensor<float>();
  AscendC::DataCopy(xLocal, this->xGm[progress * this->tileLength],
                    this->tileLength);
  AscendC::DataCopy(yLocal, this->yGm[progress * this->tileLength],
                    this->tileLength);
  inQueueX.EnQue(xLocal);
  inQueueY.EnQue(yLocal);
}

__aicore__ inline void KernelSub::Compute(int32_t progress) {
  AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
  AscendC::LocalTensor<float> yLocal = inQueueY.DeQue<float>();
  AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
  AscendC::Sub(zLocal, xLocal, yLocal, this->tileLength);
  inQueueX.FreeTensor(xLocal);
  inQueueY.FreeTensor(yLocal);
  outQueueZ.EnQue(zLocal);
}

__aicore__ inline void KernelSub::CopyOut(int32_t progress) {
  AscendC::LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
  AscendC::DataCopy(zGm[progress * tileLength], zLocal, this->tileLength);
  outQueueZ.FreeTensor(zLocal);
}

extern "C" __global__ __aicore__ void
vector_sub(GM_ADDR x, GM_ADDR y, GM_ADDR z, VectorSubTilingData tiling) {
  KernelSub op;
  op.Init(x, y, z, tiling.totalLength, tiling.tileNum);
  op.Process();
}

// 供 Host 调用的核函数启动桩
void vector_sub_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y,
                   uint8_t *z, VectorSubTilingData tiling) {
  vector_sub<<<blockDim, nullptr, stream>>>(x, y, z, &tiling);
}
