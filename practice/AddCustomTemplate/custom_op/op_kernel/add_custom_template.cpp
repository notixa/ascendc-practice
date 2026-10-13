#include "kernel_operator.h"
#include <cstdint>

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t QUEUE_DEPTH = 2;

template <class dtypeX, class dtypeY, class dtypeZ> class KernelAdd {
public:
  __aicore__ inline KernelAdd() {}
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                              uint32_t totalLength, uint32_t tileNum) {
    this->blockLength = totalLength / AscendC::GetBlockNum();
    this->tileNum = tileNum;
    this->tileLength = this->blockLength / this->tileNum / BUFFER_NUM;
    pipe.InitBuffer(inQueX, 2, this->tileLength * sizeof(dtypeX));
    pipe.InitBuffer(inQueY, 2, this->tileLength * sizeof(dtypeY));
    pipe.InitBuffer(outQueZ, 2, this->tileLength * sizeof(dtypeZ));
    uint32_t off_set = AscendC::GetBlockIdx() * this->blockLength;
    xGm.SetGlobalBuffer((dtypeX *)x + off_set, this->blockLength);
    yGm.SetGlobalBuffer((dtypeY *)y + off_set, this->blockLength);
    zGm.SetGlobalBuffer((dtypeZ *)z + off_set, this->blockLength);
  }
  __aicore__ inline void Process() {
    int32_t loopCnt = tileNum * BUFFER_NUM;
    for (int32_t i = 0; i < loopCnt; ++i) {
      CopyIn(i);
      Compute(i);
      CopyOut(i);
    }
  }

private:
  __aicore__ inline void CopyIn(int32_t process) {
    AscendC::LocalTensor<dtypeX> xLocal = inQueX.AllocTensor<dtypeX>();
    AscendC::LocalTensor<dtypeY> yLocal = inQueY.AllocTensor<dtypeY>();
    AscendC::DataCopy(xLocal, xGm[process * tileLength], tileLength);
    AscendC::DataCopy(yLocal, yGm[process * tileLength], tileLength);
    inQueX.EnQue(xLocal);
    inQueY.EnQue(yLocal);
  }
  __aicore__ inline void Compute(int32_t process) {
    AscendC::LocalTensor<dtypeX> xLocal = inQueX.DeQue<dtypeX>();
    AscendC::LocalTensor<dtypeY> yLocal = inQueY.DeQue<dtypeY>();
    AscendC::LocalTensor<dtypeZ> zLocal = outQueZ.AllocTensor<dtypeZ>();
    AscendC::Add(zLocal, xLocal, yLocal, tileLength);
    inQueX.FreeTensor(xLocal);
    inQueY.FreeTensor(yLocal);
  }
  __aicore__ inline void CopyOut(int32_t process) {
    AscendC::LocalTensor<dtypeZ> zLocal = outQueZ.DeQue<dtypeZ>();
    AscendC::DataCopy(zGm[process * tileLength], zLocal, tileLength);
    outQueZ.FreeTensor(zLocal);
  }

private:
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::TPosition::VECIN, QUEUE_DEPTH> inQueX, inQueY;
  AscendC::TQue<AscendC::TPosition::VECOUT, QUEUE_DEPTH> outQueZ;
  AscendC::GlobalTensor<dtypeX> xGm;
  AscendC::GlobalTensor<dtypeY> yGm;
  AscendC::GlobalTensor<dtypeZ> zGm;
  uint32_t blockLength;
  uint32_t tileNum;
  uint32_t tileLength;
};

extern "C" __global__ __aicore__ void add_custom_template(GM_ADDR x, GM_ADDR y,
                                                          GM_ADDR z,
                                                          GM_ADDR workspace,
                                                          GM_ADDR tiling) {
  GET_TILING_DATA(tiling_data, tiling);
  // TODO: user kernel impl
  KernelAdd<DTYPE_X, DTYPE_Y, DTYPE_Z> op;
  op.Init(x, y, z, tiling_data.size, tiling_data.tile_num);
  op.Process();
}
