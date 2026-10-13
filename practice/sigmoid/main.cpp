#include "acl/acl.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <vector>

constexpr uint32_t BUFFER_NUM = 2; // tensor num for each queue
constexpr uint32_t QUEUE_DEPTH = 2;

struct SigmoidCustomTilingData {
  uint32_t totalLength;
  uint32_t tileNum;
};

void sigmoid_custom_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y,
                       SigmoidCustomTilingData tiling);

std::vector<float> kernel_sigmoid(std::vector<float> &x) {
  constexpr uint32_t numBlocks = 8;
  uint32_t totalLength = x.size();
  size_t totalByteSize = totalLength * sizeof(float);
  int32_t deviceId = 0;
  aclrtStream stream = nullptr;
  SigmoidCustomTilingData tiling{totalLength, numBlocks};

  uint8_t *xHost = reinterpret_cast<uint8_t *>(x.data());
  uint8_t *yHost = nullptr;
  uint8_t *xDevice = nullptr;
  uint8_t *yDevice = nullptr;

  aclInit(nullptr);
  aclrtSetDevice(deviceId);
  aclrtCreateStream(&stream);

  aclrtMallocHost((void **)(&yHost), totalByteSize);
  aclrtMalloc((void **)&xDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMalloc((void **)&yDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

  aclrtMemcpy(xDevice, totalByteSize, xHost, totalByteSize,
              ACL_MEMCPY_HOST_TO_DEVICE);

  sigmoid_custom_do(numBlocks, stream, xDevice, yDevice, tiling);
  aclrtSynchronizeStream(stream);

  aclrtMemcpy(yHost, totalByteSize, yDevice, totalByteSize,
              ACL_MEMCPY_DEVICE_TO_HOST);
  std::vector<float> y((float *)yHost, (float *)(yHost + totalByteSize));

  aclrtFree(xDevice);
  aclrtFree(yDevice);
  aclrtFreeHost(yHost);
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return y;
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
  return 0;
}

std::vector<float> sigmoid(const std::vector<float> &x) {
  std::vector<float> y(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    y[i] = 1.0f / (1.0f + std::exp(-x[i]));
  }
  return y;
}

int32_t main(int32_t argc, char *argv[]) {
  constexpr uint32_t totalLength = 8 * 2048;
  constexpr float valueX = 5.5f;
  std::vector<float> x(totalLength, valueX);
  std::vector<float> output = kernel_sigmoid(x);
  std::vector<float> golden = sigmoid(x);
  return VerifyResult(output, golden);
}
