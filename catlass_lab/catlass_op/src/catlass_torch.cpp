/**
 * catlass_torch.cpp —— PyTorch (PrivateUse1/NPU) 侧的胶水
 *
 * 职责：
 *   1. 校验 dtype / device / 连续性，把 4-D [B,N_q,H,D] 归一成 3-D [B*N_q,H,D]
 *   2. 从 tensor shape 算 FATTiling，拷到 device
 *   3. 分配三块 workspace（S / P / oTemp）
 *   4. 调 catlass_kernel::fat(...) 启动 kernel
 *
 * 支持两种输入布局（内存排布必须一致，都是 token-major）：
 *   3-D: [B*N_q, H, D]
 *   4-D: [B, N_q, H, D]      → reshape 成 3-D（连续张量上零拷贝）
 *
 * mask：必须显式传入一张 [>=128, >=128] 的 fp16 上三角表（1 = 屏蔽）。
 *   kernel 只在对角块用它，坐标是"相对对角块起点"，所以一张 128x128 表
 *   可以服务所有 q block（平移不变性）。列起点恒为 0，故表不能被裁剪过。
 *
 * 不做的事（这是 FA 相对 matmul 模板的差异）：
 *   - 没有 Kernel::Arguments / ToUnderlyingArguments / GetWorkspaceSize，
 *     因为 FATKernel 只暴露 operator()(FATParams)，workspace 由这里手算
 *   - 没有 torch.autograd（第一版只做 forward）
 */

#include <torch/extension.h>
// 官方 torch-intf 模板用的这一组（顺序照抄）：
//   OpAdapter.h 提供 at_npu::native::OpCommand（RUN_NPU_FUNC 依赖）
//   NPUFormat.h / CalcuOpUtil.h 提供 torch_npu 的 layout/dtype 适配
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/utils/CalcuOpUtil.h>
#include <torch_npu/csrc/framework/utils/OpAdapter.h>
#include <acl/acl.h>      // AclAPI 总入口（类型/错误码）
#include <dlfcn.h>        // dlsym：绕开 aclrtGetHardwareSyncAddr 的头文件可见性问题
#include <tiling/platform/platform_ascendc.h>

#include <algorithm>
#include <cstring>

#include "catlass_kernel.h"
#include "fat_tiling.h"   // 只含 FATTiling（纯 POD）；FATParams 是设备侧的，不要在这里引

namespace {

// aclrtGetHardwareSyncAddr 在不同 CANN 版本 / 不同 acl 头里的可见性不稳定
// （torch_npu 也自带一份精简 acl 头，会遮蔽真声明）。它本身是 libascendcl 导出的
// C 符号，所以直接用 dlsym 取，绕开编译期头文件差异。
// 拿不到就明确报错 —— 传 0 会让 AIC/AIV 的跨核 flag 落在错误的同步区（表现为 kernel 卡死）。
uint64_t GetHardwareSyncAddrOrThrow()
{
    using Fn = int (*)(void**);          // aclError(*)(void**)，ACL_SUCCESS == 0
    static Fn fn = nullptr;
    static bool tried = false;
    if (!tried) {
        tried = true;
        fn = reinterpret_cast<Fn>(dlsym(RTLD_DEFAULT, "aclrtGetHardwareSyncAddr"));
        if (fn == nullptr) {
            void* h = dlopen("libascendcl.so", RTLD_NOW | RTLD_GLOBAL);
            if (h != nullptr) {
                fn = reinterpret_cast<Fn>(dlsym(h, "aclrtGetHardwareSyncAddr"));
            }
        }
    }
    TORCH_CHECK(fn != nullptr,
                "dlsym(aclrtGetHardwareSyncAddr) 失败：拿不到 AIC<->AIV 同步基地址。"
                "请确认 libascendcl.so 已加载（import torch_npu 之后才有）。");
    uint64_t addr = 0;
    TORCH_CHECK(fn(reinterpret_cast<void**>(&addr)) == 0, "aclrtGetHardwareSyncAddr 返回失败");
    return addr;
}

} // namespace

#define RUN_NPU_FUNC(func, ...)                                                                                    \
    do {                                                                                                           \
        if ((func) == nullptr) {                                                                                   \
            throw std::runtime_error(                                                                              \
                std::string("Function pointer is null at ") + __FILE__ + ":" + std::to_string(__LINE__) + " in " + \
                #func);                                                                                            \
        }                                                                                                          \
        at_npu::native::OpCommand::RunOpApiV2(#func, [=]() -> int {                                                \
            func(__VA_ARGS__);                                                                                     \
            return 0;                                                                                              \
        });                                                                                                        \
    } while (false)

namespace catlass_torch {

// 与 kernel/catlass_kernel.asc 里 L1TileShape = GemmShape<128,128,128> 保持一致
static constexpr int64_t B_Q = 128;
static constexpr int64_t B_K = 128;

at::Tensor fat(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
               const at::Tensor& mask, double scale, int64_t batch, int64_t blockDim)
{
    // ---------------- 1. 基础校验 ----------------
    TORCH_CHECK(q.device().type() == c10::DeviceType::PrivateUse1, "q must be on NPU");
    TORCH_CHECK(k.device().type() == c10::DeviceType::PrivateUse1, "k must be on NPU");
    TORCH_CHECK(v.device().type() == c10::DeviceType::PrivateUse1, "v must be on NPU");
    TORCH_CHECK(q.scalar_type() == at::kHalf, "only float16 is supported");
    TORCH_CHECK(k.scalar_type() == at::kHalf, "only float16 is supported");
    TORCH_CHECK(v.scalar_type() == at::kHalf, "only float16 is supported");
    TORCH_CHECK(q.dim() == 3 || q.dim() == 4,
                "q must be [B*N_q, H, D] or [B, N_q, H, D], got rank ", q.dim());

    // mask：1 = 屏蔽（上三角，diagonal=1），0 = 保留；dtype 必须 2 字节
    // 源码里 epilogue 用 (maskStride - columnNum) * 2 和 columnNum * 2 硬编码 2 字节，
    // 且列起点恒为 0（表须对齐对角块起点），故这里只校验 shape/stride/dtype。
    TORCH_CHECK(mask.defined(), "mask tensor is required (this kernel is causal)");
    TORCH_CHECK(mask.device().type() == c10::DeviceType::PrivateUse1, "mask must be on NPU");
    TORCH_CHECK(mask.device() == q.device(), "mask must be on the same device as q");
    TORCH_CHECK(mask.scalar_type() == at::kHalf, "mask must be float16 (2 bytes)");
    TORCH_CHECK(mask.dim() == 2, "mask must be 2-D [rows, cols]");
    TORCH_CHECK(mask.is_contiguous(), "mask must be contiguous");
    TORCH_CHECK(mask.size(0) >= B_Q && mask.size(1) >= B_K,
                "mask must be at least ", B_Q, "x", B_K, ", got ", mask.size(0), "x", mask.size(1));

    // ---------------- 2. 归一化成 3-D [tokens, heads, head_dim] ----------------
    // 连续张量上 reshape 只改元数据，不拷贝
    const bool is4d = (q.dim() == 4);
    at::Tensor q3, k3, v3;

    if (is4d) {
        TORCH_CHECK(k.dim() == 4 && v.dim() == 4, "q/k/v must have the same rank");
        TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0), "batch size mismatch");
        TORCH_CHECK(q.size(2) == k.size(2) && q.size(2) == v.size(2), "heads mismatch");
        TORCH_CHECK(q.size(3) == k.size(3) && q.size(3) == v.size(3), "head_dim mismatch");
        if (batch <= 0) {
            batch = q.size(0); // 4-D 时 batch 可以从 shape 推出来
        }
        TORCH_CHECK(batch == q.size(0), "batch(", batch, ") != q.size(0)(", q.size(0), ")");
        q3 = q.reshape({-1, q.size(2), q.size(3)});
        k3 = k.reshape({-1, k.size(2), k.size(3)});
        v3 = v.reshape({-1, v.size(2), v.size(3)});
    } else {
        TORCH_CHECK(batch >= 1, "batch is required (and must be >= 1) for 3-D input");
        q3 = q;
        k3 = k;
        v3 = v;
    }

    TORCH_CHECK(q3.is_contiguous() && k3.is_contiguous() && v3.is_contiguous(),
                "q/k/v must be contiguous (call .contiguous() first)");
    TORCH_CHECK(k3.sizes() == v3.sizes(), "k/v must have the same shape");
    TORCH_CHECK(q3.size(1) == k3.size(1) && q3.size(2) == k3.size(2),
                "q/k must share heads and head_dim (no GQA in this kernel)");

    const int64_t nQTokens = q3.size(0);
    const int64_t nKTokens = k3.size(0);
    const int64_t heads    = q3.size(1);
    const int64_t headDim  = q3.size(2);
    TORCH_CHECK(nQTokens % batch == 0 && nKTokens % batch == 0, "token count not divisible by batch");
    const int64_t N_q = nQTokens / batch;
    const int64_t N_k = nKTokens / batch;

    // ---------------- 3. tiling ----------------
    const int64_t blocksPerBatch = (N_q + B_Q - 1) / B_Q;
    const int64_t numTask        = batch * blocksPerBatch * heads;

    if (blockDim <= 0) {
        int64_t aic = static_cast<int64_t>(
            platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic());
        blockDim = std::min<int64_t>(numTask, aic);
    }
    TORCH_CHECK(blockDim > 0, "blockDim must be > 0");

    FATTiling hostTiling{};
    hostTiling.N_q       = static_cast<uint32_t>(N_q);
    hostTiling.N_k       = static_cast<uint32_t>(N_k);
    hostTiling.B_q       = static_cast<uint32_t>(B_Q);
    hostTiling.B_k       = static_cast<uint32_t>(B_K);
    hostTiling.head_num  = static_cast<uint32_t>(heads);
    hostTiling.head_dim  = static_cast<uint32_t>(headDim);
    hostTiling.batch_num = static_cast<uint32_t>(batch);
    hostTiling.numTask   = static_cast<uint32_t>(numTask);
    hostTiling.scale     = static_cast<float>(scale);

    // FATTiling 是 POD：用 from_blob 借 host 内存 → to(device) 走 torch 的 H2D 拷贝
    auto tilingDev = at::from_blob(&hostTiling, {static_cast<int64_t>(sizeof(FATTiling))},
                                   at::TensorOptions().dtype(at::kByte))
                         .to(q.device());

    // ---------------- 4. workspace（三块，与 kernel 的 layout stride 对应） ----------------
    //   s     : [coreNum, B_q, B_k]     float
    //   p     : [coreNum, B_q, B_k]     half
    //   oTemp : [coreNum, B_q, headDim] float
    auto sDev  = at::empty({blockDim * B_Q * B_K}, q.options().dtype(at::kFloat));
    auto pDev  = at::empty({blockDim * B_Q * B_K}, q.options().dtype(at::kHalf));
    auto otDev = at::empty({blockDim * B_Q * headDim}, q.options().dtype(at::kFloat));

    auto o3 = at::empty_like(q3);

    // ---------------- 5. launch ----------------
    // 用默认流：bench 场景不涉及自定义 stream，传 nullptr 最稳（不依赖 c10::npu:: 的版本差异）。
    // 如果以后要在自定义流/多流上用，改成 c10::npu::getCurrentNPUStream().stream()，
    // 并在服务器上用 grep 确认它的命名空间：
    //   grep -rn "getCurrentNPUStream" $TA/include/ | head
    void* stream = nullptr;

    // ★ AIC<->AIV 跨核 flag 的硬件同步基地址（fai.cpp 同样要做）
    const uint64_t hwSyncAddr = GetHardwareSyncAddrOrThrow();

    RUN_NPU_FUNC(catlass_kernel::fat, hwSyncAddr, static_cast<uint32_t>(blockDim), stream,
                 reinterpret_cast<uint8_t*>(q3.data_ptr()),
                 reinterpret_cast<uint8_t*>(k3.data_ptr()),
                 reinterpret_cast<uint8_t*>(v3.data_ptr()),
                 reinterpret_cast<uint8_t*>(mask.data_ptr()), // 1 = 屏蔽（上三角）
                 reinterpret_cast<uint8_t*>(o3.data_ptr()),
                 reinterpret_cast<uint8_t*>(sDev.data_ptr()),
                 reinterpret_cast<uint8_t*>(pDev.data_ptr()),
                 reinterpret_cast<uint8_t*>(otDev.data_ptr()),
                 reinterpret_cast<uint8_t*>(tilingDev.data_ptr()));

    // 输出形状跟输入保持一致
    return is4d ? o3.reshape({batch, N_q, heads, headDim}) : o3;
}

} // namespace catlass_torch

TORCH_LIBRARY(catlass, m)
{
    m.def("fat(Tensor q, Tensor k, Tensor v, Tensor mask, float scale, int batch, int blockDim) -> Tensor");
}

TORCH_LIBRARY_IMPL(catlass, PrivateUse1, m)
{
    m.impl("fat", catlass_torch::fat);
}
