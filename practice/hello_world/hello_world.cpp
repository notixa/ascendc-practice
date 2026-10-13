/**
 * @file hello_world.cpp
 *
 * Ascend C 算子核函数实现 (Device 侧)
 */
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void hello_world() {
  AscendC::printf("Hello World from Ascend C AI Core!\n");
}

// 封装启动核函数的 Host Stub 函数，供 main.cpp 调用
void hello_world_do(uint32_t blockDim, void *stream) {
  hello_world<<<blockDim, nullptr, stream>>>();
}
