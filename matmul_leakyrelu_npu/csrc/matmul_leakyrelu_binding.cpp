#include "matmul_leakyrelu_kernel.h"
#include "matmul_leakyrelu_tiling.h"
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

at::Tensor matmul_leakyrelu_forward(const at::Tensor &a, const at::Tensor &b,
                                    double negative_slope = 0.01,
                                    const c10::optional<at::Tensor> &bias = c10::nullopt) {
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

    // 1. 计算 Cube-Vector Tiling 数据
    MatmulLeakyReluTilingData tilingData;
    ComputeMatmulLeakyReluTiling(M, N, K, static_cast<float>(negative_slope), tilingData);

    void *bias_ptr = nullptr;
    at::Tensor bias_contig;
    if (bias.has_value() && bias.value().defined()) {
        TORCH_CHECK(bias.value().scalar_type() == at::kFloat, "Bias must be Float (FP32)");
        TORCH_CHECK(bias.value().numel() == N, "Bias length must equal N = ", N);
        bias_contig = bias.value().contiguous();
        bias_ptr = bias_contig.data_ptr();
        tilingData.cubeTiling.isBias = 1;
    } else {
        tilingData.cubeTiling.isBias = 0;
    }

    uint32_t blockDim = tilingData.totalBlock;

    auto launch_kernel = [&](void *a_p, void *b_p, void *bias_p, void *out_p, aclrtStream stream) {
        Matmul_leakyrelu_custom_fp16_do(blockDim, stream, a_p, b_p, bias_p, out_p, tilingData);
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
        launch_kernel(a_contig.data_ptr(), b_contig.data_ptr(), bias_ptr, out.data_ptr(), stream);
#endif
    } else {
        // CPU fallback: 通过显存分配 -> H2D -> 执行 -> D2H
        aclrtStream stream = g_stream;

        size_t a_bytes = a_contig.numel() * sizeof(uint16_t);
        size_t b_bytes = b_contig.numel() * sizeof(uint16_t);
        size_t out_bytes = out.numel() * sizeof(uint16_t);

        void *a_dev = nullptr;
        void *b_dev = nullptr;
        void *bias_dev = nullptr;
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

        if (bias_ptr) {
            size_t bias_bytes = bias_contig.numel() * sizeof(float);
            err = aclrtMalloc(&bias_dev, bias_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
            TORCH_CHECK(err == ACL_ERROR_NONE, "aclrtMalloc for Bias failed: ", err);
            err = aclrtMemcpy(bias_dev, bias_bytes, bias_contig.data_ptr(), bias_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
            TORCH_CHECK(err == ACL_ERROR_NONE, "aclrtMemcpy H2D Bias failed: ", err);
        }

        launch_kernel(a_dev, b_dev, bias_dev, out_dev, stream);

        err = aclrtSynchronizeStream(stream);
        TORCH_CHECK(err == ACL_ERROR_NONE, "aclrtSynchronizeStream failed: ", err);

        err = aclrtMemcpy(out.data_ptr(), out_bytes, out_dev, out_bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        TORCH_CHECK(err == ACL_ERROR_NONE, "aclrtMemcpy D2H Out failed: ", err);

        aclrtFree(a_dev);
        aclrtFree(b_dev);
        if (bias_dev) {
            aclrtFree(bias_dev);
        }
        aclrtFree(out_dev);
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_leakyrelu_forward, "Ascend C MatMul + LeakyReLU FP16 Forward",
          py::arg("a"), py::arg("b"), py::arg("negative_slope") = 0.01, py::arg("bias") = py::none());
}
