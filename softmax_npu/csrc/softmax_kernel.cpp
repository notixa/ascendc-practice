#include "kernel_operator.h"
#include "softmax_tiling.h"

using namespace AscendC;

constexpr uint32_t QUEUE_DEPTH = 2;
constexpr uint32_t BUFFER_SIZE = 2;

template <class dtype, uint32_t BATCH_ROWS> class Softmax {
public:
  __aicore__ inline Softmax() {}
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                              const SoftmaxTiling &tiling);
  __aicore__ inline void Process();

private:
  __aicore__ inline void CopyIn(int32_t process, uint32_t curBatchRows);
  __aicore__ inline void Compute(int32_t process, uint32_t curBatchRows);
  __aicore__ inline void CopyOut(int32_t process, uint32_t curBatchRows);

private:
  TPipe pipe;
  TQue<TPosition::VECIN, QUEUE_DEPTH> inQue;
  TQue<TPosition::VECOUT, QUEUE_DEPTH> outQue;
  TBuf<TPosition::VECCALC> maxBuf;
  TBuf<TPosition::VECCALC> sumBuf; // must be float32
  TBuf<TPosition::VECCALC> workSpaceBuf;
  GlobalTensor<dtype> inGm;
  GlobalTensor<dtype> outGm;

  uint32_t hiddenSize; // d_model
  uint32_t rowsNum;    // num of rows per ai core
  uint32_t batchRows;  // num of rows per batch(copy in, less than rowsNum)
};

template <class dtype, uint32_t BATCH_ROWS>
__aicore__ inline void
Softmax<dtype, BATCH_ROWS>::Init(GM_ADDR x, GM_ADDR y,
                                 const SoftmaxTiling &tiling) {
  uint32_t coreIdx = GetBlockIdx();
  hiddenSize = tiling.hiddenSize;
  uint64_t offset = 0;
  if (coreIdx < tiling.largeCoreNum) {
    rowsNum = tiling.rowsPerLargeCore;
    offset = static_cast<uint64_t>(coreIdx) * hiddenSize * rowsNum;
  } else {
    rowsNum = tiling.rowsPerTailCore;
    offset = static_cast<uint64_t>(tiling.largeCoreNum) * hiddenSize *
                 tiling.rowsPerLargeCore +
             static_cast<uint64_t>(coreIdx - tiling.largeCoreNum) * hiddenSize *
                 tiling.rowsPerTailCore;
  }
  batchRows = BATCH_ROWS;
  pipe.InitBuffer(inQue, BUFFER_SIZE, batchRows * hiddenSize * sizeof(dtype));
  pipe.InitBuffer(outQue, BUFFER_SIZE, batchRows * hiddenSize * sizeof(dtype));
  pipe.InitBuffer(maxBuf, 32);
  pipe.InitBuffer(sumBuf, 32);
  pipe.InitBuffer(workSpaceBuf, 4096);
  inGm.SetGlobalBuffer(reinterpret_cast<__gm__ dtype *>(x) + offset,
                       rowsNum * hiddenSize);
  outGm.SetGlobalBuffer(reinterpret_cast<__gm__ dtype *>(y) + offset,
                        rowsNum * hiddenSize);
}

template <class dtype, uint32_t BATCH_ROWS>
__aicore__ inline void Softmax<dtype, BATCH_ROWS>::Process() {
  if (rowsNum == 0) {
    return;
  }
  int32_t loopCnt = (rowsNum + batchRows - 1) / batchRows;
  for (int32_t i = 0; i < loopCnt; i++) {
    uint32_t curBatchRows = (i == loopCnt - 1 && rowsNum % batchRows != 0)
                                ? (rowsNum % batchRows)
                                : batchRows;
    CopyIn(i, curBatchRows);
    Compute(i, curBatchRows);
    CopyOut(i, curBatchRows);
  }
}

template <class dtype, uint32_t BATCH_ROWS>
__aicore__ inline void
Softmax<dtype, BATCH_ROWS>::CopyIn(int32_t process, uint32_t curBatchRows) {
  LocalTensor<dtype> xLocal = inQue.AllocTensor<dtype>();
  uint64_t offset = static_cast<uint64_t>(batchRows) * hiddenSize * process;
  DataCopy(xLocal, inGm[offset], curBatchRows * hiddenSize);
  inQue.EnQue(xLocal);
}

template <class dtype, uint32_t BATCH_ROWS>
__aicore__ inline void
Softmax<dtype, BATCH_ROWS>::Compute(int32_t process, uint32_t curBatchRows) {
  LocalTensor<dtype> xLocal = inQue.DeQue<dtype>();
  LocalTensor<dtype> yLocal = outQue.AllocTensor<dtype>();
  LocalTensor<dtype> maxLocal = maxBuf.Get<dtype>();
  LocalTensor<dtype> sumLocal = sumBuf.Get<dtype>();
  LocalTensor<dtype> reduceBuf = workSpaceBuf.Get<dtype>();

  for (uint32_t i = 0; i < curBatchRows; ++i) {
    uint64_t offset = static_cast<uint64_t>(hiddenSize) * i;
    ReduceMax(maxLocal, xLocal[offset], reduceBuf, hiddenSize);
    dtype maxVal = maxLocal.GetValue(0);
    Adds(yLocal[offset], xLocal[offset],
         static_cast<dtype>(-static_cast<float>(maxVal)), hiddenSize);
    Exp(yLocal[offset], yLocal[offset], hiddenSize);
    ReduceSum(sumLocal, yLocal[offset], reduceBuf, hiddenSize);
    dtype factor =
        static_cast<float>(1.0f / static_cast<float>(sumLocal.GetValue(0)));
    Muls(yLocal[offset], yLocal[offset], factor, hiddenSize);
  }
  inQue.FreeTensor(xLocal);
  outQue.EnQue(yLocal);
}

template <class dtype, uint32_t BATCH_ROWS>
__aicore__ inline void
Softmax<dtype, BATCH_ROWS>::CopyOut(int32_t process, uint32_t curBatchRows) {
  LocalTensor<dtype> yLocal = outQue.DeQue<dtype>();
  uint64_t offset = static_cast<uint64_t>(batchRows) * hiddenSize * process;
  DataCopy(outGm[offset], yLocal, curBatchRows * hiddenSize);
  outQue.FreeTensor(yLocal);
}

#define INSTANTIATE_SOFTMAX_KERNEL(NAME, DTYPE, BATCH_ROWS)                    \
  extern "C" __global__ __aicore__ void NAME(GM_ADDR x, GM_ADDR y,             \
                                             SoftmaxTiling tiling) {           \
    Softmax<DTYPE, BATCH_ROWS> op;                                             \
    op.Init(x, y, tiling);                                                     \
    op.Process();                                                              \
  }

INSTANTIATE_SOFTMAX_KERNEL(softmax_fp32_b8, float, 8);
INSTANTIATE_SOFTMAX_KERNEL(softmax_fp16_b8, half, 8);


