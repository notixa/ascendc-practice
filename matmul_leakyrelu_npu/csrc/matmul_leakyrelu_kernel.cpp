#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "matmul_leakyrelu_tiling.h"
using namespace AscendC;
using namespace matmul;

constexpr uint32_t BUFFER_SIZE = 2;
constexpr uint32_t PIPELINE_QUEUE_DEPTH = 2;

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

class MatmulLeakyReluKernel {
  using A_TYPE = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
  using B_TYPE = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
  using C_TYPE = MatmulType<TPosition::VECIN, CubeFormat::ND, half>;
  using BIAS_TYPE = MatmulType<TPosition::GM, CubeFormat::ND, float>;

public:
  __aicore__ inline MatmulLeakyReluKernel() {}
  __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c,
                              GM_ADDR tiling) {
    LoadTilingFromGM(tiling, tilingData_);

    aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(a),
                         tilingData_.cubeTiling.M * tilingData_.cubeTiling.Ka);
    bGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(b),
                         tilingData_.cubeTiling.Kb * tilingData_.cubeTiling.N);
    cGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(c),
                         tilingData_.cubeTiling.M * tilingData_.cubeTiling.N);
    if (tilingData_.cubeTiling.isBias) {
      biasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bias),
                              tilingData_.cubeTiling.N);
    }

    pipe_.InitBuffer(mmQue_, BUFFER_SIZE,
                     tilingData_.cubeTiling.baseM *
                         tilingData_.cubeTiling.baseN * sizeof(half));
    pipe_.InitBuffer(outQue_, BUFFER_SIZE,
                     tilingData_.cubeTiling.baseM *
                         tilingData_.cubeTiling.baseN * sizeof(half));

    mm_.Init(&tilingData_.cubeTiling, &pipe_);
    mm_.SetOrgShape(tilingData_.cubeTiling.M, tilingData_.cubeTiling.N,
                    tilingData_.cubeTiling.Ka, tilingData_.cubeTiling.Kb);
  }
  __aicore__ inline void Process() {
    uint32_t blockIdx = GetBlockIdx();
    if (blockIdx >= tilingData_.totalBlock) {
      return;
    }
    uint32_t mIdx = blockIdx / tilingData_.nTotalCnt;
    uint32_t nIdx = blockIdx % tilingData_.nTotalCnt;
    mm_.SetSingleShape(tilingData_.cubeTiling.singleCoreM,
                       tilingData_.cubeTiling.singleCoreN,
                       tilingData_.cubeTiling.singleCoreK);
    uint64_t offsetA =
        mIdx * tilingData_.cubeTiling.singleCoreM * tilingData_.cubeTiling.Ka;
    uint64_t offsetB = nIdx * tilingData_.cubeTiling.singleCoreN;
    uint64_t offsetC =
        mIdx * tilingData_.cubeTiling.N * tilingData_.cubeTiling.singleCoreM +
        nIdx * tilingData_.cubeTiling.singleCoreN;
    mm_.SetTensorA(aGm_[offsetA], false);
    mm_.SetTensorB(bGm_[offsetB], false);
    if (tilingData_.cubeTiling.isBias) {
      uint64_t offsetBias = nIdx * tilingData_.cubeTiling.singleCoreN;
      mm_.SetBias(biasGm_[offsetBias]);
    }

    while (mm_.Iterate()) {
      LocalTensor<half> mmLocal = mmQue_.AllocTensor<half>();
      mm_.GetTensorC(mmLocal, false, true);
      mmQue_.EnQue(mmLocal);

      mmLocal = mmQue_.DeQue<half>();
      LocalTensor<half> outLocal = outQue_.AllocTensor<half>();
      LeakyRelu(outLocal, mmLocal, static_cast<half>(tilingData_.negativeSlope),
                tilingData_.cubeTiling.baseM * tilingData_.cubeTiling.baseN);
      mmQue_.FreeTensor(mmLocal);
      outQue_.EnQue(outLocal);

      outLocal = outQue_.DeQue<half>();
      DataCopyParams copyParam;
      copyParam.blockCount = tilingData_.cubeTiling.baseM;
      copyParam.blockLen = tilingData_.cubeTiling.baseN * sizeof(half) / 32;
      copyParam.srcStride = 0;
      copyParam.dstStride =
          (tilingData_.cubeTiling.N - tilingData_.cubeTiling.baseN) *
          sizeof(half) / 32;
      DataCopy(cGm_[offsetC], outLocal, copyParam);
      outQue_.FreeTensor(outLocal);
    }
    PipeBarrier<PIPE_ALL>();
  }
  __aicore__ inline void End() { mm_.End(); }

private:
  TPipe pipe_;
  TQue<TPosition::VECIN, PIPELINE_QUEUE_DEPTH> mmQue_;
  TQue<TPosition::VECOUT, PIPELINE_QUEUE_DEPTH> outQue_;
  GlobalTensor<half> aGm_;
  GlobalTensor<half> bGm_;
  GlobalTensor<float> biasGm_;
  GlobalTensor<half> cGm_;
  MatmulLeakyReluTilingData tilingData_;
  Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> mm_;
};

extern "C" __global__ __aicore__ void
matmul_leakyrelu_custom_fp16(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c,
                             GM_ADDR tiling) {
  MatmulLeakyReluKernel op;
  op.Init(a, b, bias, c, tiling);
  op.Process();
  op.End();
}
