/**
 * @file main.cpp
 *
 * 矢量加法算子调用、精度验证与 aclrtEvent Benchmark 基准性能测试 (Host 侧)
 */
#include "acl/acl.h"
#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <vector>

struct VectorAddTilingData {
  uint32_t totalLength;
  uint32_t tileNum;
};

// 声明核函数启动桩
extern void vector_add_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y,
                          uint8_t *z, VectorAddTilingData tiling);

/**
 * 基于 aclrtEvent 的 NPU 硬件精准性能测试函数
 */
void BenchmarkKernel(uint32_t numBlocks, aclrtStream stream, uint8_t *xDevice,
                     uint8_t *yDevice, uint8_t *zDevice, VectorAddTilingData tiling,
                     uint32_t totalLength, int warmupIters = 10, int benchIters = 10) {
  // 1. 创建硬件事件对象
  aclrtEvent startEvent = nullptr;
  aclrtEvent endEvent = nullptr;
  aclrtCreateEvent(&startEvent);
  aclrtCreateEvent(&endEvent);

  // 2. 预热 (Warm-up)：排除 Cache 缺失和初始指令加载等冷启动抖动
  for (int i = 0; i < warmupIters; ++i) {
    vector_add_do(numBlocks, stream, xDevice, yDevice, zDevice, tiling);
  }
  aclrtSynchronizeStream(stream);

  // 3. 循环测时
  float totalTimeMs = 0.0f;
  float minTimeMs = 1e9f;
  float maxTimeMs = 0.0f;

  for (int i = 0; i < benchIters; ++i) {
    // 在 Stream 中打上起始事件标记
    aclrtRecordEvent(startEvent, stream);

    // 下发待测核函数
    vector_add_do(numBlocks, stream, xDevice, yDevice, zDevice, tiling);

    // 在 Stream 中打上结束事件标记
    aclrtRecordEvent(endEvent, stream);

    // 等待该事件硬件完成并读取两个 Event 之间的耗时 (毫秒)
    aclrtSynchronizeEvent(endEvent);
    float iterTimeMs = 0.0f;
    aclrtEventElapsedTime(&iterTimeMs, startEvent, endEvent);

    totalTimeMs += iterTimeMs;
    if (iterTimeMs < minTimeMs) minTimeMs = iterTimeMs;
    if (iterTimeMs > maxTimeMs) maxTimeMs = iterTimeMs;
  }

  // 4. 统计与带宽计算 (Effective Bandwidth)
  float avgTimeMs = totalTimeMs / benchIters;
  float avgTimeUs = avgTimeMs * 1000.0f;
  // 矢量加法：读入 X(float32) + 读入 Y(float32) + 写出 Z(float32) = 3 * N * 4 字节
  double totalBytes = 3.0 * totalLength * sizeof(float);
  double bandwidthGBs = (totalBytes / (avgTimeMs / 1000.0f)) / 1e9;

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "\n==================== [Ascend C Kernel Benchmark] ====================\n";
  std::cout << " 数组元素总数 (Elements)     : " << totalLength << "\n";
  std::cout << " 搬运总数据量 (Total Bytes)  : " << (totalBytes / (1024.0 * 1024.0)) << " MB\n";
  std::cout << " 测试迭代轮数 (Iterations)   : " << benchIters << " (Warmup: " << warmupIters << ")\n";
  std::cout << " 单次平均耗时 (Average Time) : " << avgTimeUs << " us (" << avgTimeMs << " ms)\n";
  std::cout << " 最快 / 最慢  (Min / Max)    : " << minTimeMs * 1000.0f << " us / " << maxTimeMs * 1000.0f << " us\n";
  std::cout << " 有效吞吐带宽 (Bandwidth)    : " << bandwidthGBs << " GB/s\n";
  std::cout << "=====================================================================\n" << std::endl;

  // 5. 释放事件资源
  aclrtDestroyEvent(startEvent);
  aclrtDestroyEvent(endEvent);
}

std::vector<float> kernel_add(std::vector<float> &x, std::vector<float> &y) {
  constexpr uint32_t numBlocks = 8;
  uint32_t totalLength = x.size();
  size_t totalByteSize = totalLength * sizeof(float);
  int32_t deviceId = 0;
  aclrtStream stream = nullptr;
  VectorAddTilingData tiling{totalLength, numBlocks};

  uint8_t *xHost = reinterpret_cast<uint8_t *>(x.data());
  uint8_t *yHost = reinterpret_cast<uint8_t *>(y.data());
  uint8_t *zHost = nullptr;
  uint8_t *xDevice = nullptr;
  uint8_t *yDevice = nullptr;
  uint8_t *zDevice = nullptr;

  aclInit(nullptr);
  aclrtSetDevice(deviceId);
  aclrtCreateStream(&stream);

  aclrtMallocHost((void **)(&zHost), totalByteSize);
  aclrtMalloc((void **)&xDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMalloc((void **)&yDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMalloc((void **)&zDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

  aclrtMemcpy(xDevice, totalByteSize, xHost, totalByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
  aclrtMemcpy(yDevice, totalByteSize, yHost, totalByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

  // 执行 Benchmark 性能基准测试
  BenchmarkKernel(numBlocks, stream, xDevice, yDevice, zDevice, tiling, totalLength, 10, 100);

  // 下发最后一次并同步，用于精度校验
  vector_add_do(numBlocks, stream, xDevice, yDevice, zDevice, tiling);
  aclrtSynchronizeStream(stream);

  aclrtMemcpy(zHost, totalByteSize, zDevice, totalByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
  std::vector<float> z((float *)zHost, (float *)(zHost + totalByteSize));

  aclrtFree(xDevice);
  aclrtFree(yDevice);
  aclrtFree(zDevice);
  aclrtFreeHost(zHost);
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return z;
}

uint32_t VerifyResult(std::vector<float> &output, std::vector<float> &golden) {
  auto printTensor = [](std::vector<float> &tensor, const char *name) {
    constexpr size_t maxPrintSize = 20;
    std::cout << name << ": ";
    std::copy(tensor.begin(),
              tensor.begin() + std::min(tensor.size(), maxPrintSize),
              std::ostream_iterator<float>(std::cout, " "));
    if (tensor.size() > maxPrintSize) {
      std::cout << "...";
    }
    std::cout << std::endl;
  };
  printTensor(output, "Output");
  printTensor(golden, "Golden");
  if (std::equal(golden.begin(), golden.end(), output.begin())) {
    std::cout << "[Success] Case accuracy is verification passed." << std::endl;
    return 0;
  } else {
    std::cout << "[Failed] Case accuracy is verification failed!" << std::endl;
    return 1;
  }
}

int32_t main(int32_t argc, char *argv[]) {
  /* ==================== 原有的调用代码 (已注释) ====================
  constexpr uint32_t totalLength = 8 * 2048;
  constexpr float valueX = 1.2f;
  constexpr float valueY = 2.3f;
  std::vector<float> x(totalLength, valueX);
  std::vector<float> y(totalLength, valueY);

  std::vector<float> output = kernel_add(x, y);
  std::vector<float> golden(totalLength, valueX + valueY);
  return VerifyResult(output, golden);
  ================================================================ */

  std::cout << "[Host] ================= [Ascend C VectorAdd Pipeline] =================" << std::endl;

  // 1. 基本参数设置
  // Atlas 200I DK A2 包含 8 个 AI Vector Core，每个核处理 2048 个 float32
  constexpr uint32_t numBlocks = 8;
  constexpr uint32_t totalLength = numBlocks * 2048;
  constexpr size_t totalByteSize = totalLength * sizeof(float);
  constexpr int32_t deviceId = 0;
  VectorAddTilingData tiling{totalLength, numBlocks};

  // 2. 初始化 ACL 运行时环境
  std::cout << "[Host] 1. 正在初始化 ACL 运行时环境与设置设备..." << std::endl;
  aclInit(nullptr);
  aclrtSetDevice(deviceId);

  // 3. 创建异步执行流 (Stream)
  std::cout << "[Host] 2. 创建异步执行任务流 (Stream)..." << std::endl;
  aclrtStream stream = nullptr;
  aclrtCreateStream(&stream);

  // 4. 准备 Host 侧测试数据与 Golden 真值
  std::cout << "[Host] 3. 准备 Host 侧输入数据 (X=1.2, Y=2.3)..." << std::endl;
  constexpr float valueX = 1.2f;
  constexpr float valueY = 2.3f;
  std::vector<float> x(totalLength, valueX);
  std::vector<float> y(totalLength, valueY);
  std::vector<float> golden(totalLength, valueX + valueY);

  // 5. 分配 Device 内存与锁页 Host 内存
  std::cout << "[Host] 4. 分配 NPU Device 内存与 Host 锁页内存 (各 " 
            << (totalByteSize / 1024.0) << " KB)..." << std::endl;
  uint8_t *xDevice = nullptr;
  uint8_t *yDevice = nullptr;
  uint8_t *zDevice = nullptr;
  uint8_t *zHost = nullptr;

  aclrtMalloc((void **)&xDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMalloc((void **)&yDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMalloc((void **)&zDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMallocHost((void **)&zHost, totalByteSize);

  // 6. 将输入数据拷贝至 NPU (Host -> Device)
  std::cout << "[Host] 5. 数据拷贝至 Device (H2D)..." << std::endl;
  aclrtMemcpy(xDevice, totalByteSize, x.data(), totalByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
  aclrtMemcpy(yDevice, totalByteSize, y.data(), totalByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

  // 7. 执行基于 aclrtEvent 的 Benchmark 性能基准测试
  std::cout << "[Host] 6. 正在启动 aclrtEvent 基准性能评测 (Warmup: 10次, Benchmark: 100次)..." << std::endl;
  BenchmarkKernel(numBlocks, stream, xDevice, yDevice, zDevice, tiling, totalLength, 10, 100);

  // 8. 下发单次计算任务并取回数据用于精度验证
  std::cout << "[Host] 7. 执行单次计算并回拷结果 (D2H) 验证精度..." << std::endl;
  vector_add_do(numBlocks, stream, xDevice, yDevice, zDevice, tiling);
  aclrtSynchronizeStream(stream);

  aclrtMemcpy(zHost, totalByteSize, zDevice, totalByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
  std::vector<float> output((float *)zHost, (float *)(zHost + totalByteSize));

  // 9. 精度比对
  uint32_t verifyRet = VerifyResult(output, golden);

  // 10. 释放所有分配的资源
  std::cout << "[Host] 8. 正在清理与释放 Device/Host 资源..." << std::endl;
  aclrtFree(xDevice);
  aclrtFree(yDevice);
  aclrtFree(zDevice);
  aclrtFreeHost(zHost);
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  std::cout << "[Host] 算子测试全部完成，退出状态: " << verifyRet << std::endl;
  return verifyRet;
}
