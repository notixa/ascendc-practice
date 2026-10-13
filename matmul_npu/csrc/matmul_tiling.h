#ifndef MATMUL_TILING_H
#define MATMUL_TILING_H

#include <cstdint>

// 如果未包含 CANN 内置 kernel_tiling.h，则定义标准 TCubeTiling 结构体
#ifndef __TIKCFW_KERNEL_TILING_H_
#pragma pack(push, 8)
struct TCubeTiling {
    int32_t usedCoreNum = 1;
    int32_t M = 0;
    int32_t N = 0;
    int32_t Ka = 0;
    int32_t Kb = 0;
    int32_t singleCoreM = 0;
    int32_t singleCoreN = 0;
    int32_t singleCoreK = 0;
    int32_t baseM = 64;
    int32_t baseN = 64;
    int32_t baseK = 64;
    int32_t depthA1 = 1;
    int32_t depthB1 = 1;
    int32_t stepM = 1;
    int32_t stepN = 1;
    int32_t isBias = 0;
    int32_t transLength = 0;
    int32_t iterateOrder = 0;
    int32_t shareMode = 0;
    int32_t shareL1Size = 0;
    int32_t shareL0CSize = 0;
    int32_t shareUbSize = 0;
    int32_t batchM = 0;
    int32_t batchN = 0;
    int32_t singleBatchM = 0;
    int32_t singleBatchN = 0;
    int32_t stepKa = 1;
    int32_t stepKb = 1;
    int32_t dbL0A = 0;
    int32_t dbL0B = 0;
    int32_t dbL0C = 0;
    int32_t ALayoutInfoB = 0;
    int32_t ALayoutInfoS = 0;
    int32_t ALayoutInfoN = 0;
    int32_t ALayoutInfoG = 0;
    int32_t ALayoutInfoD = 0;
    int32_t BLayoutInfoB = 0;
    int32_t BLayoutInfoS = 0;
    int32_t BLayoutInfoN = 0;
    int32_t BLayoutInfoG = 0;
    int32_t BLayoutInfoD = 0;
    int32_t CLayoutInfoB = 0;
    int32_t CLayoutInfoS1 = 0;
    int32_t CLayoutInfoN = 0;
    int32_t CLayoutInfoG = 0;
    int32_t CLayoutInfoS2 = 0;
    int32_t BatchNum = 1;
    int32_t reserved = 0;
};
#pragma pack(pop)
#endif // __TIKCFW_KERNEL_TILING_H_

#pragma pack(push, 8)
// 传给 Kernel 的完整 Tiling 数据包（支持二维网格切分）
struct MatmulTilingData {
    TCubeTiling cubeTiling;
    int32_t mTotalCnt = 1;  // M 轴网格分块数 (ceil(M / singleCoreM))
    int32_t nTotalCnt = 1;  // N 轴网格分块数 (ceil(N / singleCoreN))
    int32_t totalBlock = 1; // 总 Grid 块数 (mTotalCnt * nTotalCnt)
};
#pragma pack(pop)

// 向上整除辅助函数
inline int32_t CeilDiv(int32_t a, int32_t b) {
    return (a + b - 1) / b;
}

// 向上对齐到 alignment 的倍数
inline int32_t AlignUp(int32_t val, int32_t alignment) {
    return ((val + alignment - 1) / alignment) * alignment;
}

/**
 * Host 侧计算 Cube 二维网格 Tiling 参数
 */
inline void ComputeMatmulTiling(int32_t M, int32_t N, int32_t K, MatmulTilingData &tilingData) {
    TCubeTiling &cube = tilingData.cubeTiling;

    cube.M = M;
    cube.N = N;
    cube.Ka = K;
    cube.Kb = K;

    // 二维网格单块尺寸：每个网格块负责 singleCoreM × singleCoreN
    // 默认选用 64×64（若 M 或 N 小于 64 则对齐到 16）
    cube.singleCoreM = (M < 64) ? AlignUp(M, 16) : 64;
    cube.singleCoreN = (N < 64) ? AlignUp(N, 16) : 64;
    cube.singleCoreK = K;

    // 二维网格分块数计算
    tilingData.mTotalCnt = CeilDiv(M, cube.singleCoreM);
    tilingData.nTotalCnt = CeilDiv(N, cube.singleCoreN);
    tilingData.totalBlock = tilingData.mTotalCnt * tilingData.nTotalCnt;
    cube.usedCoreNum = tilingData.totalBlock;

    // 基础计算块（脉动阵列 L0 粒度，等于 singleCore 块大小或 64）
    cube.baseM = cube.singleCoreM;
    cube.baseN = cube.singleCoreN;
    cube.baseK = (K < 64) ? AlignUp(K, 16) : 64;

    cube.stepM = 1;
    cube.stepN = 1;
    cube.stepKa = 1;
    cube.stepKb = 1;

    cube.depthA1 = 1;
    cube.depthB1 = 1;

    cube.isBias = 0;
    cube.iterateOrder = 0;
}

#endif // MATMUL_TILING_H
