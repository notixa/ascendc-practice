#include <torch/extension.h>
#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <cstdint>
#include <cstdlib>
#include "rmsnorm_kernel.h"
#include "rmsnorm_tiling.h"

#if __has_include(<torch_npu/csrc/core/npu/NPUStream.h>)
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/DeviceUtils.h>
#define HAS_TORCH_NPU 1
#else
#define HAS_TORCH_NPU 0
#endif

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
  // Default to 2 for Atlas 200I DK A2 (Ascend 310B)
  aivCoreNum = 2;
  return aivCoreNum;
}

static bool g_acl_initialized = false;
static aclrtStream g_stream = nullptr;

static void ensure_acl_ready() {
  if (!g_acl_initialized) {
    aclError ret = aclInit(nullptr);
    if (ret != ACL_ERROR_NONE && ret != 100002) { // 100002: repeat initialize
      TORCH_CHECK(false, "aclInit failed with error: ", ret);
    }
    int32_t deviceId = 0;
    ret = aclrtSetDevice(deviceId);
    TORCH_CHECK(ret == ACL_ERROR_NONE, "aclrtSetDevice failed with error: ", ret);

    ret = aclrtCreateStream(&g_stream);
    TORCH_CHECK(ret == ACL_ERROR_NONE, "aclrtCreateStream failed with error: ", ret);

    g_acl_initialized = true;
  }
}

at::Tensor rmsnorm_forward(const at::Tensor &x, const at::Tensor &gamma) {
  ensure_acl_ready();

  TORCH_CHECK(x.scalar_type() == gamma.scalar_type(),
              "x and gamma must have same type");
  TORCH_CHECK(x.device() == gamma.device(),
              "x and gamma must be on the same device");

  auto x_contig = x.contiguous();
  auto gamma_contig = gamma.contiguous();

  uint32_t element_size = x_contig.element_size();
  uint32_t align_elements = 32 / element_size;
  uint32_t hiddenSize = x_contig.size(-1);
  TORCH_CHECK(hiddenSize % align_elements == 0,
              "hiddenSize must be 32B aligned (multiple of ", align_elements,
              "), got ", hiddenSize);

  uint32_t totalRows = x_contig.numel() / hiddenSize;

  RMSNormTiling tilingData;
  tilingData.totalRows = totalRows;
  tilingData.hiddenSize = hiddenSize;
  tilingData.reduceTBufBytes = 512;

  uint32_t aivCoreNum = get_aiv_core();

  tilingData.coreNum = (totalRows < aivCoreNum) ? totalRows : aivCoreNum;
  if (tilingData.coreNum == 0) {
    tilingData.coreNum = 1;
  }

  tilingData.rowsPerTailCore = totalRows / tilingData.coreNum;
  tilingData.fullCoreNum = totalRows % tilingData.coreNum;
  tilingData.rowsPerFullCore = tilingData.rowsPerTailCore + 1;

  auto out = at::empty_like(x_contig);

  bool is_device_tensor = false;
#if HAS_TORCH_NPU
  if (torch_npu::utils::is_npu(x_contig)) {
    is_device_tensor = true;
  }
#endif

  auto launch_kernel = [&](void *x_ptr, void *gamma_ptr, void *out_ptr, void *stream) {
    if (x_contig.scalar_type() == at::kFloat) {
      RMSNorm_fp32_do(tilingData.coreNum, stream, x_ptr, gamma_ptr, out_ptr, tilingData);
    } else if (x_contig.scalar_type() == at::kHalf) {
      RMSNorm_fp16_do(tilingData.coreNum, stream, x_ptr, gamma_ptr, out_ptr, tilingData);
    } else if (x_contig.scalar_type() == at::kBFloat16) {
      RMSNorm_bf16_do(tilingData.coreNum, stream, x_ptr, gamma_ptr, out_ptr, tilingData);
    } else {
      TORCH_CHECK(false, "Unsupported dtype: ", x_contig.scalar_type());
    }
  };

  if (is_device_tensor) {
#if HAS_TORCH_NPU
    const c10_npu::NPUGuard guard(x_contig.device());
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(true);
    launch_kernel(x_contig.data_ptr(), gamma_contig.data_ptr(), out.data_ptr(), stream);
#endif
  } else {
    // Pure ACL Mode: CPU Tensor -> aclrtMalloc -> aclrtMemcpy H2D -> kernel -> sync -> aclrtMemcpy D2H -> free
    ensure_acl_ready();
    aclrtStream stream = g_stream;

    size_t x_bytes = x_contig.numel() * element_size;
    size_t gamma_bytes = gamma_contig.numel() * element_size;
    size_t out_bytes = out.numel() * element_size;

    void *x_dev = nullptr;
    void *gamma_dev = nullptr;
    void *out_dev = nullptr;

    aclError err;
    err = aclrtMalloc(&x_dev, x_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    TORCH_CHECK(err == ACL_ERROR_NONE, "aclrtMalloc for x failed: ", err);

    err = aclrtMalloc(&gamma_dev, gamma_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    TORCH_CHECK(err == ACL_ERROR_NONE, "aclrtMalloc for gamma failed: ", err);

    err = aclrtMalloc(&out_dev, out_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    TORCH_CHECK(err == ACL_ERROR_NONE, "aclrtMalloc for out failed: ", err);

    err = aclrtMemcpy(x_dev, x_bytes, x_contig.data_ptr(), x_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    TORCH_CHECK(err == ACL_ERROR_NONE, "aclrtMemcpy H2D for x failed: ", err);

    err = aclrtMemcpy(gamma_dev, gamma_bytes, gamma_contig.data_ptr(), gamma_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    TORCH_CHECK(err == ACL_ERROR_NONE, "aclrtMemcpy H2D for gamma failed: ", err);

    launch_kernel(x_dev, gamma_dev, out_dev, stream);

    err = aclrtSynchronizeStream(stream);
    TORCH_CHECK(err == ACL_ERROR_NONE, "aclrtSynchronizeStream failed: ", err);

    err = aclrtMemcpy(out.data_ptr(), out_bytes, out_dev, out_bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    TORCH_CHECK(err == ACL_ERROR_NONE, "aclrtMemcpy D2H for out failed: ", err);

    aclrtFree(x_dev);
    aclrtFree(gamma_dev);
    aclrtFree(out_dev);
  }

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &rmsnorm_forward, "RMSNorm forward (Ascend C)");
}
