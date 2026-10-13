#include "matmul_kernel.h"
#include "matmul_tiling.h"
#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <cstdint>
#include <cstdlib>
#include <torch/extension.h>

#if __has_include(<torch_npu/csrc/core/npu/NPUStream.h>)
#include <torch_npu/csrc/core/npu/DeviceUtils.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#define HAS_TORCH_NPU 1
#else
#define HAS_TORCH_NPU 0
#endif

static bool g_acl_initialized = false;
static aclrtStream g_stream = nullptr;

static void ensure_acl_ready() {
    if (!g_acl_initialized) {
        aclError ret = aclInit(nullptr);
        if (ret != ACL_ERROR_NONE && ret != 100002) {
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

at::Tensor matmul_forward(const at::Tensor &a, const at::Tensor &b) {
    ensure_acl_ready();

    TORCH_CHECK(a.scalar_type() == at::kHalf, "Input A must be Half (FP16)");
    TORCH_CHECK(b.scalar_type() == at::kHalf, "Input B must be Half (FP16)");
    TORCH_CHECK(a.dim() == 2, "Input A must be a 2D matrix [M, K]");
    TORCH_CHECK(b.dim() == 2, "Input B must be a 2D matrix [K, N]");

    auto a_contig = a.contiguous();
    auto b_contig = b.contiguous();

    int32_t M = static_cast<int32_t>(a_contig.size(0));
    int32_t K = static_cast<int32_t>(a_contig.size(1));
    int32_t K_b = static_cast<int32_t>(b_contig.size(0));
    int32_t N = static_cast<int32_t>(b_contig.size(1));

    TORCH_CHECK(K == K_b, "Matrix inner dimension mismatch: A is [", M, ", ", K, "], B is [", K_b, ", ", N, "]");

    auto out = at::empty({M, N}, a_contig.options());

    // 1. 计算 Cube Tiling 数据
    MatmulTilingData tilingData;
    ComputeMatmulTiling(M, N, K, tilingData);

    uint32_t blockDim = tilingData.totalBlock;

    auto launch_kernel = [&](void *a_ptr, void *b_ptr, void *out_ptr, aclrtStream stream) {
        Matmul_custom_fp16_do(blockDim, stream, a_ptr, b_ptr, out_ptr, tilingData);
    };

    // 2. 根据 Tensor 位置执行 Launch
    bool is_device_tensor = false;
#if HAS_TORCH_NPU
    if (torch_npu::utils::is_npu(a_contig) && torch_npu::utils::is_npu(b_contig)) {
        is_device_tensor = true;
    }
#endif

    if (is_device_tensor) {
#if HAS_TORCH_NPU
        auto stream = c10_npu::getCurrentNPUStream().stream(false);
        launch_kernel(a_contig.data_ptr(), b_contig.data_ptr(), out.data_ptr(), stream);
#endif
    } else {
        // CPU fallback: 通过显存分配 -> H2D -> 执行 -> D2H
        aclrtStream stream = g_stream;

        size_t a_bytes = a_contig.numel() * sizeof(uint16_t);
        size_t b_bytes = b_contig.numel() * sizeof(uint16_t);
        size_t out_bytes = out.numel() * sizeof(uint16_t);

        void *a_dev = nullptr;
        void *b_dev = nullptr;
        void *out_dev = nullptr;

        aclError err;
        err = aclrtMalloc(&a_dev, a_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        TORCH_CHECK(err == ACL_ERROR_NONE, "aclrtMalloc for A failed: ", err);

        err = aclrtMalloc(&b_dev, b_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        TORCH_CHECK(err == ACL_ERROR_NONE, "aclrtMalloc for B failed: ", err);

        err = aclrtMalloc(&out_dev, out_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        TORCH_CHECK(err == ACL_ERROR_NONE, "aclrtMalloc for Out failed: ", err);

        err = aclrtMemcpy(a_dev, a_bytes, a_contig.data_ptr(), a_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        TORCH_CHECK(err == ACL_ERROR_NONE, "aclrtMemcpy H2D A failed: ", err);

        err = aclrtMemcpy(b_dev, b_bytes, b_contig.data_ptr(), b_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        TORCH_CHECK(err == ACL_ERROR_NONE, "aclrtMemcpy H2D B failed: ", err);

        launch_kernel(a_dev, b_dev, out_dev, stream);

        err = aclrtSynchronizeStream(stream);
        TORCH_CHECK(err == ACL_ERROR_NONE, "aclrtSynchronizeStream failed: ", err);

        err = aclrtMemcpy(out.data_ptr(), out_bytes, out_dev, out_bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        TORCH_CHECK(err == ACL_ERROR_NONE, "aclrtMemcpy D2H Out failed: ", err);

        aclrtFree(a_dev);
        aclrtFree(b_dev);
        aclrtFree(out_dev);
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_forward, "Ascend C MatMul FP16 Forward");
}
