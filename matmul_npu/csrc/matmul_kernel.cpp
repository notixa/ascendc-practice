#include "kernel_operator.h"
#define ASCENDC_CUBE_ONLY
#include "lib/matmul_intf.h"
#include "matmul_tiling.h"

using namespace AscendC;
using namespace matmul;

/**
 * 从 GM 加载 TilingHeader 到 Stack（标量循环读取）
 */
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

/**
 * 二维网格切分 Matmul Kernel
 * 请在下方实现你的二维网格切分逻辑！
 */
class MatmulKernel {
public:
  // 1. 定义 A、B、C 矩阵在 GM 上的 MatmulType
  using A_TYPE = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
  using B_TYPE = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
  using C_TYPE = MatmulType<TPosition::GM, CubeFormat::ND, half>;

  __aicore__ inline MatmulKernel() {}

  /**
   * 初始化函数：
   * TODO:
   * 1. 调用 LoadTilingFromGM 加载 Tiling 元数据
   * 2. 绑定 aGlobal_, bGlobal_, cGlobal_ 的全局显存地址与总长度
   * 3. 初始化 mm_ 引擎 (mm_.Init)
   * 4. 设置全局原始矩阵形状 (mm_.SetOrgShape)
   */
  __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR tiling) {
    // [请在这里编写 Init 代码]
    LoadTilingFromGM(tiling, tilingData_);
    aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(a),
                             tilingData_.cubeTiling.M *
                                 tilingData_.cubeTiling.Ka);
    bGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(b),
                             tilingData_.cubeTiling.N *
                                 tilingData_.cubeTiling.Kb);
    cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(c),
                             tilingData_.cubeTiling.M *
                                 tilingData_.cubeTiling.N);
    mm_.Init(&tilingData_.cubeTiling, &pipe_);
    mm_.SetOrgShape(tilingData_.cubeTiling.M, tilingData_.cubeTiling.N,
                    tilingData_.cubeTiling.Ka, tilingData_.cubeTiling.Kb);
  }

  /**
   * 二维网格计算与写回核心逻辑：
   * TODO:
   * 1. 获取当前一维核号 blockIdx = GetBlockIdx()，进行越界防御
   * 2. 将一维 blockIdx 解码为二维网格坐标 (mIdx, nIdx)：
   *    - mIdx = ? (行索引)
   *    - nIdx = ? (列索引)
   * 3. 设置当前网格块的计算形状 (mm_.SetSingleShape)
   * 4. 计算当前 (mIdx, nIdx) 网格块在 A、B 矩阵上的起始偏移：
   *    - A 是 [M, K]，第 mIdx 行切块的起始元素偏移是多少？
   *    - B 是 [K, N]，第 nIdx 列切块的起始元素偏移是多少？
   *    - mm_.SetTensorA(aGlobal_[offsetA], false);
   *    - mm_.SetTensorB(bGlobal_[offsetB], false);
   * 5. 执行计算并写回 C 矩阵对应的二维位置：
   *    - C 是 [M, N]，当前 (mIdx, nIdx) 块在 C 上的起始元素偏移是多少？
   *    - mm_.IterateAll(cGlobal_[offsetC], 0);
   * 6. 调用 PipeBarrier<PIPE_ALL>() 等待完成
   */
  __aicore__ inline void Process() {
    // [请在这里编写二维网格切分 Process 代码]
    uint32_t blockIdx = GetBlockIdx();
    if (blockIdx >= tilingData_.cubeTiling.usedCoreNum) {
      return;
    }
    uint32_t mIdx = blockIdx / tilingData_.nTotalCnt;
    uint32_t nIdx = blockIdx % tilingData_.nTotalCnt;
    mm_.SetSingleShape(tilingData_.cubeTiling.singleCoreM,
                       tilingData_.cubeTiling.singleCoreN,
                       tilingData_.cubeTiling.singleCoreK);
    uint64_t offsetA =
        mIdx * tilingData_.cubeTiling.Ka * tilingData_.cubeTiling.singleCoreM;
    uint64_t offsetB = nIdx * tilingData_.cubeTiling.singleCoreN;
    mm_.SetTensorA(aGlobal_[offsetA], false);
    mm_.SetTensorB(bGlobal_[offsetB], false);
    uint64_t offsetC =
        mIdx * tilingData_.cubeTiling.N * tilingData_.cubeTiling.singleCoreM +
        nIdx * tilingData_.cubeTiling.singleCoreN;
    mm_.IterateAll(cGlobal_[offsetC], 0);
    PipeBarrier<PIPE_ALL>();
  }

  __aicore__ inline void End() { mm_.End(); }

private:
  TPipe pipe_;
  MatmulTilingData tilingData_{};
  GlobalTensor<half> aGlobal_;
  GlobalTensor<half> bGlobal_;
  GlobalTensor<half> cGlobal_;
  Matmul<A_TYPE, B_TYPE, C_TYPE> mm_;
};

/**
 * Kernel 外部入口函数
 */
extern "C" __global__ __aicore__ void
matmul_custom_fp16(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR tiling) {
  if ASCEND_IS_AIV {
    return;
  }
  MatmulKernel op;
  op.Init(a, b, c, tiling);
  op.Process();
  op.End();
}
