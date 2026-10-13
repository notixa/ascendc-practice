/**
 * @file main.cpp
 *
 * 矢量加法算子调用、精度验证与 aclrtEvent Benchmark 基准性能测试 (Host 侧)
 */
#include "acl/acl.h"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <vector>

struct VectorSubTilingData {
  uint32_t totalLength;
  uint32_t tileNum;
};

// 声明核函数启动桩
extern void vector_sub_do(uint32_t blockDim, void *stream, uint8_t *x,
                          uint8_t *y, uint8_t *z, VectorSubTilingData tiling);

std::vector<float> kernel_sub(std::vector<float> &x, std::vector<float> &y) {
  constexpr uint32_t numBlocks = 8;
  uint32_t totalLength = x.size();
  size_t totalByteSize = totalLength * sizeof(float);
  int32_t deviceId = 0;
  aclrtStream stream = nullptr;
  VectorSubTilingData tiling{totalLength, numBlocks};

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

  aclrtMemcpy(xDevice, totalByteSize, xHost, totalByteSize,
              ACL_MEMCPY_HOST_TO_DEVICE);
  aclrtMemcpy(yDevice, totalByteSize, yHost, totalByteSize,
              ACL_MEMCPY_HOST_TO_DEVICE);

  vector_sub_do(numBlocks, stream, xDevice, yDevice, zDevice, tiling);
  aclrtSynchronizeStream(stream);

  aclrtMemcpy(zHost, totalByteSize, zDevice, totalByteSize,
              ACL_MEMCPY_DEVICE_TO_HOST);
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

  constexpr uint32_t totalLength = 8 * 2048;
  constexpr float valueX = 1.2f;
  constexpr float valueY = 2.3f;
  std::vector<float> x(totalLength, valueX);
  std::vector<float> y(totalLength, valueY);

  std::vector<float> output = kernel_sub(x, y);
  std::vector<float> golden(totalLength, valueX - valueY);
  return VerifyResult(output, golden);
}
