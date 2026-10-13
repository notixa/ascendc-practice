#include "flash_attention_fwd_kernel.h"
#include "flash_attention_fwd_tiling.h"
#include "flash_attention_bwd_kernel.h"
#include "flash_attention_bwd_tiling.h"
#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <cstdint>
#include <cstdlib>
#include <vector>
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

std::vector<at::Tensor> flash_attention_forward(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    double scale = 0.0) {

    ensure_acl_ready();

    TORCH_CHECK(q.scalar_type() == at::kHalf, "Input Q must be Half (FP16)");
    TORCH_CHECK(k.scalar_type() == at::kHalf, "Input K must be Half (FP16)");
    TORCH_CHECK(v.scalar_type() == at::kHalf, "Input V must be Half (FP16)");

    TORCH_CHECK(q.dim() == 2, "Input Q must be 2D [N_q, d]");
    TORCH_CHECK(k.dim() == 2, "Input K must be 2D [N_k, d]");
    TORCH_CHECK(v.dim() == 2, "Input V must be 2D [N_k, d]");

    auto q_contig = q.contiguous();
    auto k_contig = k.contiguous();
    auto v_contig = v.contiguous();

    uint32_t N_q = static_cast<uint32_t>(q_contig.size(0));
    uint32_t d   = static_cast<uint32_t>(q_contig.size(1));
    uint32_t N_k = static_cast<uint32_t>(k_contig.size(0));
    uint32_t d_k = static_cast<uint32_t>(k_contig.size(1));
    uint32_t N_v = static_cast<uint32_t>(v_contig.size(0));
    uint32_t d_v = static_cast<uint32_t>(v_contig.size(1));

    TORCH_CHECK(d == d_k && d == d_v, "Feature dimension d mismatch: Q has ", d,
                ", K has ", d_k, ", V has ", d_v);
    TORCH_CHECK(N_k == N_v, "Sequence length mismatch: K has ", N_k, ", V has ", N_v);
    TORCH_CHECK(N_q % 16 == 0 && N_k % 16 == 0 && d % 16 == 0,
                "N_q, N_k, and d must be multiples of 16 for Ascend C alignment");

    auto o = at::empty({static_cast<int64_t>(N_q), static_cast<int64_t>(d)}, q_contig.options());
    auto l = at::empty({static_cast<int64_t>(N_q)}, q_contig.options());

    // 1. 计算 FlashAttention Tiling
    FlashAttentionFwdTilingData tilingData;
    float scale_val = static_cast<float>(scale);
    uint32_t targetCoreNum = 1; // 310B1 默认单核或多核调度
    ComputeFlashAttentionFwdTiling(N_q, N_k, d, targetCoreNum, tilingData, scale_val);

    uint32_t blockDim = tilingData.coreNum;

    auto launch_kernel = [&](void *q_p, void *k_p, void *v_p, void *o_p, void *l_p, aclrtStream stream) {
        FlashAttentionFwdKernel_do(blockDim, stream, q_p, k_p, v_p, o_p, l_p, tilingData);
    };

    // 2. 根据 Tensor 位置执行 Launch
    bool is_device_tensor = false;
#if HAS_TORCH_NPU
    if (torch_npu::utils::is_npu(q_contig) && torch_npu::utils::is_npu(k_contig) && torch_npu::utils::is_npu(v_contig)) {
        is_device_tensor = true;
    }
#endif

    if (is_device_tensor) {
#if HAS_TORCH_NPU
        auto stream = c10_npu::getCurrentNPUStream().stream(false);
        launch_kernel(q_contig.data_ptr(), k_contig.data_ptr(), v_contig.data_ptr(),
                      o.data_ptr(), l.data_ptr(), stream);
#endif
    } else {
        // CPU fallback 模式：Host -> Device -> Launch -> Device -> Host
        aclrtStream stream = g_stream;

        size_t q_bytes = q_contig.numel() * sizeof(uint16_t);
        size_t k_bytes = k_contig.numel() * sizeof(uint16_t);
        size_t v_bytes = v_contig.numel() * sizeof(uint16_t);
        size_t o_bytes = o.numel() * sizeof(uint16_t);
        size_t l_bytes = l.numel() * sizeof(uint16_t);

        void *d_q = nullptr;
        void *d_k = nullptr;
        void *d_v = nullptr;
        void *d_o = nullptr;
        void *d_l = nullptr;

        aclError ret = aclrtMalloc(&d_q, q_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        TORCH_CHECK(ret == ACL_ERROR_NONE, "aclrtMalloc d_q failed: ", ret);
        ret = aclrtMalloc(&d_k, k_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        TORCH_CHECK(ret == ACL_ERROR_NONE, "aclrtMalloc d_k failed: ", ret);
        ret = aclrtMalloc(&d_v, v_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        TORCH_CHECK(ret == ACL_ERROR_NONE, "aclrtMalloc d_v failed: ", ret);
        ret = aclrtMalloc(&d_o, o_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        TORCH_CHECK(ret == ACL_ERROR_NONE, "aclrtMalloc d_o failed: ", ret);
        ret = aclrtMalloc(&d_l, l_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        TORCH_CHECK(ret == ACL_ERROR_NONE, "aclrtMalloc d_l failed: ", ret);

        aclrtMemcpyAsync(d_q, q_bytes, q_contig.data_ptr(), q_bytes, ACL_MEMCPY_HOST_TO_DEVICE, stream);
        aclrtMemcpyAsync(d_k, k_bytes, k_contig.data_ptr(), k_bytes, ACL_MEMCPY_HOST_TO_DEVICE, stream);
        aclrtMemcpyAsync(d_v, v_bytes, v_contig.data_ptr(), v_bytes, ACL_MEMCPY_HOST_TO_DEVICE, stream);

        launch_kernel(d_q, d_k, d_v, d_o, d_l, stream);

        aclrtMemcpyAsync(o.data_ptr(), o_bytes, d_o, o_bytes, ACL_MEMCPY_DEVICE_TO_HOST, stream);
        aclrtMemcpyAsync(l.data_ptr(), l_bytes, d_l, l_bytes, ACL_MEMCPY_DEVICE_TO_HOST, stream);
        aclrtSynchronizeStream(stream);

        aclrtFree(d_q);
        aclrtFree(d_k);
        aclrtFree(d_v);
        aclrtFree(d_o);
        aclrtFree(d_l);
    }

    return {o, l};
}

std::vector<at::Tensor> flash_attention_backward(
    const at::Tensor &do_grad,
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &o,
    const at::Tensor &l,
    double scale = 0.0) {

    ensure_acl_ready();

    TORCH_CHECK(do_grad.scalar_type() == at::kHalf, "Input do_grad must be Half (FP16)");
    TORCH_CHECK(q.scalar_type() == at::kHalf, "Input Q must be Half (FP16)");
    TORCH_CHECK(k.scalar_type() == at::kHalf, "Input K must be Half (FP16)");
    TORCH_CHECK(v.scalar_type() == at::kHalf, "Input V must be Half (FP16)");
    TORCH_CHECK(o.scalar_type() == at::kHalf, "Input O must be Half (FP16)");
    TORCH_CHECK(l.scalar_type() == at::kHalf, "Input L must be Half (FP16)");

    TORCH_CHECK(do_grad.dim() == 2, "Input do_grad must be 2D [N_q, d]");
    TORCH_CHECK(q.dim() == 2, "Input Q must be 2D [N_q, d]");
    TORCH_CHECK(k.dim() == 2, "Input K must be 2D [N_k, d]");
    TORCH_CHECK(v.dim() == 2, "Input V must be 2D [N_k, d]");
    TORCH_CHECK(o.dim() == 2, "Input O must be 2D [N_q, d]");

    auto q_contig  = q.contiguous();
    auto k_contig  = k.contiguous();
    auto v_contig  = v.contiguous();
    auto o_contig  = o.contiguous();
    auto do_contig = do_grad.contiguous();
    auto l_contig  = l.contiguous();

    uint32_t N_q = static_cast<uint32_t>(q_contig.size(0));
    uint32_t d   = static_cast<uint32_t>(q_contig.size(1));
    uint32_t N_k = static_cast<uint32_t>(k_contig.size(0));
    uint32_t d_k = static_cast<uint32_t>(k_contig.size(1));
    uint32_t N_v = static_cast<uint32_t>(v_contig.size(0));
    uint32_t d_v = static_cast<uint32_t>(v_contig.size(1));

    TORCH_CHECK(d == d_k && d == d_v, "Feature dimension d mismatch: Q has ", d,
                ", K has ", d_k, ", V has ", d_v);
    TORCH_CHECK(N_k == N_v, "Sequence length mismatch: K has ", N_k, ", V has ", N_v);
    TORCH_CHECK(o_contig.size(0) == N_q && o_contig.size(1) == d,
                "O shape mismatch: expected [", N_q, ", ", d, "]");
    TORCH_CHECK(do_contig.size(0) == N_q && do_contig.size(1) == d,
                "do_grad shape mismatch: expected [", N_q, ", ", d, "]");
    TORCH_CHECK(l_contig.numel() == static_cast<int64_t>(N_q),
                "L numel mismatch: expected ", N_q, ", got ", l_contig.numel());
    TORCH_CHECK(N_q % 16 == 0 && N_k % 16 == 0 && d % 16 == 0,
                "N_q, N_k, and d must be multiples of 16 for Ascend C alignment");

    // D = rowsum(dO * O), computed with FP32 intermediate accumulation for accuracy
    auto d_tensor = (do_contig.to(at::kFloat) * o_contig.to(at::kFloat)).sum(-1).to(at::kHalf).contiguous();

    auto dq = at::empty({static_cast<int64_t>(N_q), static_cast<int64_t>(d)}, q_contig.options());
    auto dk = at::empty({static_cast<int64_t>(N_k), static_cast<int64_t>(d)}, k_contig.options());
    auto dv = at::empty({static_cast<int64_t>(N_k), static_cast<int64_t>(d)}, v_contig.options());

    // 1. 计算 FlashAttention Backward Tiling
    FlashAttentionBwd_dKdV_TilingData tilingData_dkdv;
    FlashAttentionBwd_dQ_TilingData tilingData_dq;
    float scale_val = static_cast<float>(scale);
    uint32_t targetCoreNum = 1;

    ComputeFlashAttentionBwd_dKdV_Tiling(N_q, N_k, d, targetCoreNum, tilingData_dkdv, scale_val);
    ComputeFlashAttentionBwd_dQ_Tiling(N_q, N_k, d, targetCoreNum, tilingData_dq, scale_val);

    uint32_t blockDim_dkdv = tilingData_dkdv.coreNum;
    uint32_t blockDim_dq   = tilingData_dq.coreNum;

    auto launch_bwd_kernels = [&](void *q_p, void *o_p, void *do_p, void *k_p, void *v_p,
                                  void *l_p, void *d_p, void *dq_p, void *dk_p, void *dv_p,
                                  aclrtStream stream) {
        FlashAttentionBwd_dKdV_Kernel_do(
            blockDim_dkdv, stream,
            q_p, o_p, do_p, k_p, v_p,
            l_p, d_p, dk_p, dv_p,
            tilingData_dkdv);
        FlashAttentionBwd_dQ_Kernel_do(
            blockDim_dq, stream,
            q_p, o_p, do_p, k_p, v_p,
            l_p, d_p, dq_p,
            tilingData_dq);
    };

    // 2. 根据 Tensor 位置执行 Launch
    bool is_device_tensor = false;
#if HAS_TORCH_NPU
    if (torch_npu::utils::is_npu(q_contig) && torch_npu::utils::is_npu(k_contig) &&
        torch_npu::utils::is_npu(v_contig) && torch_npu::utils::is_npu(o_contig) &&
        torch_npu::utils::is_npu(do_contig) && torch_npu::utils::is_npu(l_contig)) {
        is_device_tensor = true;
    }
#endif

    if (is_device_tensor) {
#if HAS_TORCH_NPU
        auto stream = c10_npu::getCurrentNPUStream().stream(false);
        launch_bwd_kernels(
            q_contig.data_ptr(), o_contig.data_ptr(), do_contig.data_ptr(),
            k_contig.data_ptr(), v_contig.data_ptr(), l_contig.data_ptr(),
            d_tensor.data_ptr(), dq.data_ptr(), dk.data_ptr(), dv.data_ptr(),
            stream);
#endif
    } else {
        // CPU fallback 模式：Host -> Device -> Launch -> Device -> Host
        aclrtStream stream = g_stream;

        size_t q_bytes   = q_contig.numel() * sizeof(uint16_t);
        size_t k_bytes   = k_contig.numel() * sizeof(uint16_t);
        size_t v_bytes   = v_contig.numel() * sizeof(uint16_t);
        size_t o_bytes   = o_contig.numel() * sizeof(uint16_t);
        size_t do_bytes  = do_contig.numel() * sizeof(uint16_t);
        size_t l_bytes   = l_contig.numel() * sizeof(uint16_t);
        size_t d_bytes   = d_tensor.numel() * sizeof(uint16_t);
        size_t dq_bytes  = dq.numel() * sizeof(uint16_t);
        size_t dk_bytes  = dk.numel() * sizeof(uint16_t);
        size_t dv_bytes  = dv.numel() * sizeof(uint16_t);

        void *dev_q   = nullptr;
        void *dev_k   = nullptr;
        void *dev_v   = nullptr;
        void *dev_o   = nullptr;
        void *dev_do  = nullptr;
        void *dev_l   = nullptr;
        void *dev_d   = nullptr;
        void *dev_dq  = nullptr;
        void *dev_dk  = nullptr;
        void *dev_dv  = nullptr;

        aclError ret;
        ret = aclrtMalloc(&dev_q, q_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        TORCH_CHECK(ret == ACL_ERROR_NONE, "aclrtMalloc dev_q failed: ", ret);
        ret = aclrtMalloc(&dev_k, k_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        TORCH_CHECK(ret == ACL_ERROR_NONE, "aclrtMalloc dev_k failed: ", ret);
        ret = aclrtMalloc(&dev_v, v_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        TORCH_CHECK(ret == ACL_ERROR_NONE, "aclrtMalloc dev_v failed: ", ret);
        ret = aclrtMalloc(&dev_o, o_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        TORCH_CHECK(ret == ACL_ERROR_NONE, "aclrtMalloc dev_o failed: ", ret);
        ret = aclrtMalloc(&dev_do, do_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        TORCH_CHECK(ret == ACL_ERROR_NONE, "aclrtMalloc dev_do failed: ", ret);
        ret = aclrtMalloc(&dev_l, l_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        TORCH_CHECK(ret == ACL_ERROR_NONE, "aclrtMalloc dev_l failed: ", ret);
        ret = aclrtMalloc(&dev_d, d_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        TORCH_CHECK(ret == ACL_ERROR_NONE, "aclrtMalloc dev_d failed: ", ret);
        ret = aclrtMalloc(&dev_dq, dq_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        TORCH_CHECK(ret == ACL_ERROR_NONE, "aclrtMalloc dev_dq failed: ", ret);
        ret = aclrtMalloc(&dev_dk, dk_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        TORCH_CHECK(ret == ACL_ERROR_NONE, "aclrtMalloc dev_dk failed: ", ret);
        ret = aclrtMalloc(&dev_dv, dv_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        TORCH_CHECK(ret == ACL_ERROR_NONE, "aclrtMalloc dev_dv failed: ", ret);

        aclrtMemcpyAsync(dev_q, q_bytes, q_contig.data_ptr(), q_bytes, ACL_MEMCPY_HOST_TO_DEVICE, stream);
        aclrtMemcpyAsync(dev_k, k_bytes, k_contig.data_ptr(), k_bytes, ACL_MEMCPY_HOST_TO_DEVICE, stream);
        aclrtMemcpyAsync(dev_v, v_bytes, v_contig.data_ptr(), v_bytes, ACL_MEMCPY_HOST_TO_DEVICE, stream);
        aclrtMemcpyAsync(dev_o, o_bytes, o_contig.data_ptr(), o_bytes, ACL_MEMCPY_HOST_TO_DEVICE, stream);
        aclrtMemcpyAsync(dev_do, do_bytes, do_contig.data_ptr(), do_bytes, ACL_MEMCPY_HOST_TO_DEVICE, stream);
        aclrtMemcpyAsync(dev_l, l_bytes, l_contig.data_ptr(), l_bytes, ACL_MEMCPY_HOST_TO_DEVICE, stream);
        aclrtMemcpyAsync(dev_d, d_bytes, d_tensor.data_ptr(), d_bytes, ACL_MEMCPY_HOST_TO_DEVICE, stream);

        launch_bwd_kernels(
            dev_q, dev_o, dev_do, dev_k, dev_v, dev_l, dev_d,
            dev_dq, dev_dk, dev_dv, stream);

        aclrtMemcpyAsync(dq.data_ptr(), dq_bytes, dev_dq, dq_bytes, ACL_MEMCPY_DEVICE_TO_HOST, stream);
        aclrtMemcpyAsync(dk.data_ptr(), dk_bytes, dev_dk, dk_bytes, ACL_MEMCPY_DEVICE_TO_HOST, stream);
        aclrtMemcpyAsync(dv.data_ptr(), dv_bytes, dev_dv, dv_bytes, ACL_MEMCPY_DEVICE_TO_HOST, stream);
        aclrtSynchronizeStream(stream);

        aclrtFree(dev_q);
        aclrtFree(dev_k);
        aclrtFree(dev_v);
        aclrtFree(dev_o);
        aclrtFree(dev_do);
        aclrtFree(dev_l);
        aclrtFree(dev_d);
        aclrtFree(dev_dq);
        aclrtFree(dev_dk);
        aclrtFree(dev_dv);
    }

    return {dq, dk, dv};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashAttention Forward & Backward FP16 PyTorch Extension Operator for Ascend C";
    m.def("forward", &flash_attention_forward,
          "FlashAttention forward computation (CUDA/NPU & CPU fallback)",
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("scale") = 0.0);
    m.def("backward", &flash_attention_backward,
          "FlashAttention backward computation (CUDA/NPU & CPU fallback)",
          py::arg("do_grad"), py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("o"), py::arg("l"),
          py::arg("scale") = 0.0);
}
