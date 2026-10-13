#include "softmax_kernel.h"
#include "softmax_tiling.h"
#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <cstdint>
#include <cstdlib>
#include <torch/extension.h>

// 检查环境是否支持 torch_npu 头文件
#if __has_include(<torch_npu/csrc/core/npu/NPUStream.h>)
#include <torch_npu/csrc/core/npu/DeviceUtils.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#define HAS_TORCH_NPU 1
#else
#define HAS_TORCH_NPU 0
#endif

// =========================================================================
// 模板脚手架：硬件信息与 ACL 环境初始化 (样板代码，无需修改)
// =========================================================================
uint32_t get_aiv_core() {
  static uint32_t aivCoreNum = 0;
  if (aivCoreNum > 0) {
    return aivCoreNum;
  }
  const char *env_cores = std::getenv("ASCEND_AIV_CORES");
  if (env_cores) {
    int val = std::atoi(env_cores);
    if (val > 0) {
      aivCoreNum = static_cast<uint32_t>(val);
      return aivCoreNum;
    }
  }
  // Atlas 200I DK A2 (Ascend 310B) 默认通常为 2 个 Vector 核心
  aivCoreNum = 2;
  return aivCoreNum;
}

static bool g_acl_initialized = false;
static aclrtStream g_stream = nullptr;

static void ensure_acl_ready() {
  if (!g_acl_initialized) {
    aclError ret = aclInit(nullptr);
    if (ret != ACL_ERROR_NONE && ret != 100002) { // 100002: 重复初始化已忽略
      TORCH_CHECK(false, "aclInit failed with error: ", ret);
    }
    int32_t deviceId = 0;
    ret = aclrtSetDevice(deviceId);
    TORCH_CHECK(ret == ACL_ERROR_NONE,
                "aclrtSetDevice failed with error: ", ret);

    ret = aclrtCreateStream(&g_stream);
    TORCH_CHECK(ret == ACL_ERROR_NONE,
                "aclrtCreateStream failed with error: ", ret);

    g_acl_initialized = true;
  }
}

// =========================================================================
// 核心算子入口函数: softmax_forward
// =========================================================================
at::Tensor softmax_forward(const at::Tensor &x) {
  ensure_acl_ready();

  TORCH_CHECK(x.is_floating_point(), "x must be a floating point tensor");
  TORCH_CHECK(x.dim() >= 2,
              "x must have at least 2 dimensions [..., hiddenSize]");

  auto x_contig = x.contiguous();
  size_t element_size = x_contig.element_size();

  // -----------------------------------------------------------------------
  // TODO [思考与编写 1]: 计算 Tiling 切分参数 (核心脑力活动)
  // -----------------------------------------------------------------------
  // 提示：
  // 1. x 最后一维是 hiddenSize (最后一维是连续被做 softmax 归约的向量维度)
  // 2. 其余所有前置维度的乘积是总行数 totalRows (例如 shape [B, S, D]，则
  // totalRows = B * S, hiddenSize = D)
  // 3. 可用核数通过 get_aiv_core() 获取
  // 4. 请思考：如何把 totalRows 均分到各个核上？
  //    - 如果 totalRows 能整除 coreNum，每个核分配几行？
  //    - 如果不能整除，大核分配几行？尾核分配几行？有几个大核？
  //    - 如果 totalRows < coreNum（行数比核数还少），该怎么切？

  SoftmaxTiling tilingData;
  /*
  uint32_t coreNum = get_aiv_core();
  uint32_t hiddenSize = ...;
  uint32_t totalRows = ...;

  // 请在下方完成你的多核切分逻辑，并填充 tilingData:
  tilingData.hiddenSize = ...;
  tilingData.largeCoreNum = ...;
  tilingData.rowsPerLargeCore = ...;
  tilingData.rowsPerTailCore = ...;
  */
  uint32_t coreNum = get_aiv_core();
  tilingData.hiddenSize = x_contig.size(-1);
  uint32_t rowsNum = x_contig.numel() / tilingData.hiddenSize;
  tilingData.largeCoreNum = rowsNum % coreNum;
  tilingData.rowsPerTailCore = rowsNum / coreNum;
  tilingData.rowsPerLargeCore = tilingData.rowsPerTailCore + 1;

  // -----------------------------------------------------------------------
  // 分配输出 Tensor (样板代码)
  // -----------------------------------------------------------------------
  auto out = at::empty_like(x_contig);

  // -----------------------------------------------------------------------
  // TODO [思考与编写 2]: 封装 Kernel 启动逻辑 (核心脑力活动)
  // -----------------------------------------------------------------------
  // 提示：
  // 1. 检查 x_contig.scalar_type()：
  //    - 若为 at::kFloat，调用 SoftMax_fp32_b8_do(...)
  //    - 若为 at::kHalf，调用 SoftMax_fp16_b8_do(...)
  //    - 其余类型通过 TORCH_CHECK 报错
  // 2. SoftMax_xxx_b8_do 的参数列表依次为:
  //    (blockDim, stream, x_device_ptr, out_device_ptr, tilingData)
  //    其中 blockDim 就是你规划启动的核心数 (通常是 coreNum)

  auto launch_kernel = [&](void *x_ptr, void *out_ptr, void *stream) {
    if (x_contig.scalar_type() == at::kFloat) {
      SoftMax_fp32_b8_do(coreNum, stream, x_ptr, out_ptr, tilingData);
    } else if (x_contig.scalar_type() == at::kHalf) {
      SoftMax_fp16_b8_do(coreNum, stream, x_ptr, out_ptr, tilingData);
    } else {
      TORCH_CHECK(false, "Unsupported dtype: ", x_contig.scalar_type());
    }
  };

  // -----------------------------------------------------------------------
  // 样板脚手架：根据 Tensor 位置执行 Launch (样板代码，无需修改)
  // -----------------------------------------------------------------------
  bool is_device_tensor = false;
#if HAS_TORCH_NPU
  if (torch_npu::utils::is_npu(x_contig)) {
    is_device_tensor = true;
  }
#endif

  if (is_device_tensor) {
#if HAS_TORCH_NPU
    // NPU Tensor 通路：直接在当前 NPU Stream 上发射，零拷贝
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    launch_kernel(x_contig.data_ptr(), out.data_ptr(), stream);
#endif
  } else {
    // CPU Tensor 通路 (Fallback)：自动通过 ACL H2D -> 启动 -> D2H 同步回写
    aclrtStream stream = g_stream;

    size_t x_bytes = x_contig.numel() * element_size;
    size_t out_bytes = out.numel() * element_size;

    void *x_dev = nullptr;
    void *out_dev = nullptr;

    aclError err;
    err = aclrtMalloc(&x_dev, x_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    TORCH_CHECK(err == ACL_ERROR_NONE, "aclrtMalloc for x failed: ", err);

    err = aclrtMalloc(&out_dev, out_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    TORCH_CHECK(err == ACL_ERROR_NONE, "aclrtMalloc for out failed: ", err);

    err = aclrtMemcpy(x_dev, x_bytes, x_contig.data_ptr(), x_bytes,
                      ACL_MEMCPY_HOST_TO_DEVICE);
    TORCH_CHECK(err == ACL_ERROR_NONE, "aclrtMemcpy H2D for x failed: ", err);

    launch_kernel(x_dev, out_dev, stream);

    err = aclrtSynchronizeStream(stream);
    TORCH_CHECK(err == ACL_ERROR_NONE, "aclrtSynchronizeStream failed: ", err);

    err = aclrtMemcpy(out.data_ptr(), out_bytes, out_dev, out_bytes,
                      ACL_MEMCPY_DEVICE_TO_HOST);
    TORCH_CHECK(err == ACL_ERROR_NONE, "aclrtMemcpy D2H for out failed: ", err);

    aclrtFree(x_dev);
    aclrtFree(out_dev);
  }

  return out;
}

// =========================================================================
// PyBind11 模块注册 (样板代码，无需修改)
// =========================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &softmax_forward, "Softmax forward (Ascend C)");
}
