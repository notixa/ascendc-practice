/**
 * @file main.cpp
 *
 * 算子调用宿主程序 (Host 侧)
 */
#include "acl/acl.h"
#include <cstdint>
#include <cstdio>

// 声明由 BiSheng / ascendc_library 生成并导出的核函数启动桩
extern void hello_world_do(uint32_t blockDim, void *stream);

int32_t main(int argc, char const *argv[])
{
    printf("[Host] 1. 正在初始化 ACL 运行时环境...\n");
    fflush(stdout);
    aclInit(nullptr);

    int32_t deviceId = 0;
    aclrtSetDevice(deviceId);

    printf("[Host] 2. 创建异步任务流 (Stream)...\n");
    fflush(stdout);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    // Atlas 200I DK A2 具有 8 个 AI Vector Core，此处 blockDim = 8
    constexpr uint32_t blockDim = 8;
    printf("[Host] 3. 正在下发核函数 hello_world 到 %u 个 AI Core...\n", blockDim);
    fflush(stdout);
    hello_world_do(blockDim, stream);

    printf("[Host] 4. 等待 NPU 执行完毕并冲刷日志 (aclrtSynchronizeStream)...\n");
    fflush(stdout);
    aclrtSynchronizeStream(stream);

    printf("[Host] 5. 释放 Stream 与 Device 资源...\n");
    fflush(stdout);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    printf("[Host] 全部完成，算子执行成功！\n");
    fflush(stdout);
    return 0;
}
