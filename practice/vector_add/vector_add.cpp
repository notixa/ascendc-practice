/**
 * @file vector_add.cpp
 *
 * Ascend C 矢量加法算子实现 (Device 侧)
 */
#include "kernel_operator.h"

constexpr uint32_t BUFFER_NUM = 2;

struct VectorAddTilingData {
  uint32_t totalLength;
  uint32_t tileNum;
};

class KernelAdd {
public:
  __aicore__ inline KernelAdd() {}
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

__aicore__ inline void KernelAdd::Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                       uint32_t totalLength, uint32_t tileNum) {
  this->blockLength = totalLength / AscendC::GetBlockNum();
  this->tileNum = tileNum;
  this->tileLength = blockLength / tileNum / BUFFER_NUM;
  uint32_t off_set = this->blockLength * AscendC::GetBlockIdx();

  xGm.SetGlobalBuffer((__gm__ float *)x + off_set, this->blockLength);
  yGm.SetGlobalBuffer((__gm__ float *)y + off_set, this->blockLength);
  zGm.SetGlobalBuffer((__gm__ float *)z + off_set, this->blockLength);

  pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
  pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
  pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(float));
}

__aicore__ inline void KernelAdd::Process() {
  int32_t loopCount = this->tileNum * BUFFER_NUM;

  for (int32_t i = 0; i < loopCount; ++i) {
    CopyIn(i);
    Compute(i);
    CopyOut(i);
  }
}

__aicore__ inline void KernelAdd::CopyIn(int32_t progress) {
  AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
  AscendC::LocalTensor<float> yLocal = inQueueY.AllocTensor<float>();

  AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
  AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);

  inQueueX.EnQue(xLocal);
  inQueueY.EnQue(yLocal);
}

__aicore__ inline void KernelAdd::Compute(int32_t progress) {
  AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
  AscendC::LocalTensor<float> yLocal = inQueueY.DeQue<float>();

  AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
  AscendC::Add(zLocal, xLocal, yLocal, this->tileLength);

  outQueueZ.EnQue(zLocal);

  inQueueX.FreeTensor(xLocal);
  inQueueY.FreeTensor(yLocal);
}

__aicore__ inline void KernelAdd::CopyOut(int32_t progress) {
  AscendC::LocalTensor<float> zLocal = outQueueZ.DeQue<float>();

  AscendC::DataCopy(zGm[progress * tileLength], zLocal, this->tileLength);

  outQueueZ.FreeTensor(zLocal);
}

extern "C" __global__ __aicore__ void vector_add(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                                 VectorAddTilingData tiling) {
  KernelAdd op;
  op.Init(x, y, z, tiling.totalLength, tiling.tileNum);
  op.Process();
}

// 供 Host 调用的核函数启动桩
void vector_add_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y, uint8_t *z,
                   VectorAddTilingData tiling) {
  vector_add<<<blockDim, nullptr, stream>>>(x, y, z, &tiling);
}
