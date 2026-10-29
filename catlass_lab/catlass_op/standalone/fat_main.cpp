/**
 * fat_main.cpp —— FATkernel 的【独立可执行】版本（照 CATLASS examples/23 的 main 写）
 *
 * 为什么要有它：
 *   同样的 kernel，官方 examples/23 编成 executable 能跑通（Compare Success），
 *   我们编成 shared library 塞进 torch 进程就卡死在 mmadPV 等 smReady 上，
 *   连 20 行的 pingpong（纯跨核 flag，无计算）也卡。
 *   ⇒ 嫌疑集中在「torch 进程 + shared library」这个嵌入方式上。
 *   这个 executable 用官方那套 ACL 初始化顺序，把变量降到最少。
 *
 * 编译：由 catlass_op/CMakeLists.txt 的 fat_bench 目标负责（LANGUAGE ASC）。
 * 运行：
 *   ./fat_bench                       # 默认 N=128 H=1 D=128 full
 *   ./fat_bench 512 8 1               # N=512 H=8 causal
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "acl/acl.h"
#include "kernel_operator.h"

// 直接复用 lib 里那份 kernel + host wrapper，保证和 torch 扩展用的是完全相同的代码
#include "../catlass_op/kernel/catlass_kernel.asc"

#define ACL_OK(expr)                                                                                 \
    do {                                                                                             \
        aclError _e = (expr);                                                                        \
        if (_e != ACL_SUCCESS) {                                                                      \
            printf("[FATAL] %s:%d  %s -> aclError %d\n", __FILE__, __LINE__, #expr, (int)_e);         \
            return -1;                                                                                \
        }                                                                                             \
    } while (0)

namespace {

// ---------- fp16 <-> fp32（只用于 0.25 的倍数，值域很小，直接暴力转换即可） ----------
// 为了简单：数据全部取 { -2, -1, 0, 1, 2 } * 0.25，这些数在 fp16 里是精确的。
uint16_t F32ToF16Exact(float v)
{
    // 标准 fp32 -> fp16（只覆盖规格数，足够本 bench 用：|v| 在 [1e-4, 100]）
    // 之前那张查表把 1.25 编成了 4.0、2.0 编成了 32.0，导致实际数据比预期大 16 倍。
    uint32_t x;
    std::memcpy(&x, &v, sizeof(x));
    const uint16_t sign = static_cast<uint16_t>((x >> 16) & 0x8000u);
    const uint32_t exp32 = (x >> 23) & 0xFFu;
    const uint32_t mant32 = x & 0x7FFFFFu;
    if (exp32 == 0) {
        return sign;                                  // 0 / 次正规 → 0
    }
    int he = static_cast<int>(exp32) - 127 + 15;
    if (he >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00u);  // inf
    }
    if (he <= 0) {
        return sign;                                  // 太小 → 0（本 bench 不会走到）
    }
    uint16_t mant16 = static_cast<uint16_t>(mant32 >> 13);
    // 就近舍入（round-half-up 到 fp16）
    if ((mant32 & 0x1FFFu) > 0x1000u) {
        mant16 = static_cast<uint16_t>(mant16 + 1u);
        if (mant16 == 0x400u) { mant16 = 0; if (++he >= 31) return static_cast<uint16_t>(sign | 0x7C00u); }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint16_t>(he) << 10) | mant16);
}

float F16ToF32Exact(uint16_t h)
{
    const int sign = (h & 0x8000) ? -1 : 1;
    const int exp = (h >> 10) & 0x1F;
    const int mant = h & 0x3FF;
    if (exp == 0 && mant == 0) {
        return 0.0f;
    }
    const float f = (1.0f + mant / 1024.0f) * std::pow(2.0f, static_cast<float>(exp) - 15.0f);
    return sign * f;
}

struct Options {
    uint32_t n = 128;
    uint32_t h = 1;
    uint32_t d = 128;
    uint32_t batch = 1;
    int causal = 0;
    uint32_t blockDim = 0; // 0 = 自动
};

void Golden(const Options& o,
            const std::vector<uint16_t>& q,
            const std::vector<uint16_t>& k,
            const std::vector<uint16_t>& v,
            std::vector<uint16_t>& out)
{
    const uint32_t N = o.n, H = o.h, D = o.d;
    const double scale = 1.0 / std::sqrt(static_cast<double>(D));
    std::vector<double> s(static_cast<size_t>(N) * N);
    out.assign(q.size(), 0);

    for (uint32_t h = 0; h < H; ++h) {
        // s[p][t] = sum_d q[p][h][d] * k[t][h][d] * scale
        for (uint32_t p = 0; p < N; ++p) {
            for (uint32_t t = 0; t < N; ++t) {
                if (o.causal && t > p) {
                    s[static_cast<size_t>(p) * N + t] = -1e30;
                    continue;
                }
                double acc = 0.0;
                for (uint32_t dd = 0; dd < D; ++dd) {
                    const size_t qi = (static_cast<size_t>(p) * H + h) * D + dd;
                    const size_t ki = (static_cast<size_t>(t) * H + h) * D + dd;
                    acc += static_cast<double>(F16ToF32Exact(q[qi])) * static_cast<double>(F16ToF32Exact(k[ki]));
                }
                s[static_cast<size_t>(p) * N + t] = acc * scale;
            }
            // 没被 mask 的做 softmax
            double mx = -1e30;
            for (uint32_t t = 0; t < N; ++t) {
                mx = std::max(mx, s[static_cast<size_t>(p) * N + t]);
            }
            double sum = 0.0;
            for (uint32_t t = 0; t < N; ++t) {
                const double e = std::exp(s[static_cast<size_t>(p) * N + t] - mx);
                s[static_cast<size_t>(p) * N + t] = e;
                sum += e;
            }
            for (uint32_t dd = 0; dd < D; ++dd) {
                double acc = 0.0;
                for (uint32_t t = 0; t < N; ++t) {
                    const size_t vi = (static_cast<size_t>(t) * H + h) * D + dd;
                    acc += s[static_cast<size_t>(p) * N + t] / sum * static_cast<double>(F16ToF32Exact(v[vi]));
                }
                out[(static_cast<size_t>(p) * H + h) * D + dd] = F32ToF16Exact(static_cast<float>(acc));
            }
        }
    }
}

} // namespace

int main(int argc, char** argv)
{
    Options o;
    if (argc > 1) o.n = static_cast<uint32_t>(atoi(argv[1]));
    if (argc > 2) o.h = static_cast<uint32_t>(atoi(argv[2]));
    if (argc > 3) o.causal = atoi(argv[3]);
    if (argc > 4) o.blockDim = static_cast<uint32_t>(atoi(argv[4]));
    const uint32_t B_q = 128, B_k = 128;
    const uint32_t B = o.batch, N = o.n, H = o.h, D = o.d;

    printf("=== fat_bench  N=%u H=%u D=%u B=%u causal=%d ===\n", N, H, D, B, o.causal);

    // ---------------- ACL 初始化（严格照 examples/23 的顺序） ----------------
    ACL_OK(aclInit(nullptr));
    ACL_OK(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    ACL_OK(aclrtCreateStream(&stream));
    uint64_t hardwareSyncAddr = 0;
    ACL_OK(aclrtGetHardwareSyncAddr(reinterpret_cast<void**>(&hardwareSyncAddr)));
    printf("[host] hardwareSyncAddr = 0x%llx\n", (unsigned long long)hardwareSyncAddr);

    // ---------------- 造数据（0.25 的整数倍，fp16 精确） ----------------
    const size_t elems = static_cast<size_t>(B) * N * H * D;
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(-8, 8); // -2 .. 2
    std::vector<uint16_t> hq(elems), hk(elems), hv(elems);
    for (size_t i = 0; i < elems; ++i) {
        hq[i] = F32ToF16Exact(dist(rng) * 0.25f);
        hk[i] = F32ToF16Exact(dist(rng) * 0.25f);
        hv[i] = F32ToF16Exact(dist(rng) * 0.25f);
    }
    // mask：[128,128] fp16，1 = 屏蔽。
    // ★ 必须跟 causal 开关一致：之前无条件造 causal mask，导致 causal=0 时
    //   golden 算全注意力、kernel 却按 causal 算，两边永远对不上。
    std::vector<uint16_t> hmask(128 * 128, 0);
    if (o.causal) {
        for (uint32_t i = 0; i < 128; ++i) {
            for (uint32_t j = 0; j < 128; ++j) {
                hmask[i * 128 + j] = (j > i) ? 0x3C00 /* 1.0 */ : 0x0000;
            }
        }
    }

    // ---------------- 设备内存 ----------------
    void* qDev = nullptr;
    void* kDev = nullptr;
    void* vDev = nullptr;
    void* oDev = nullptr;
    void* maskDev = nullptr;
    void* tilingDev = nullptr;
    ACL_OK(aclrtMalloc(&qDev, elems * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_OK(aclrtMalloc(&kDev, elems * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_OK(aclrtMalloc(&vDev, elems * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_OK(aclrtMalloc(&oDev, elems * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_OK(aclrtMalloc(&maskDev, 128 * 128 * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_OK(aclrtMalloc(&tilingDev, sizeof(FATTiling), ACL_MEM_MALLOC_HUGE_FIRST));

    ACL_OK(aclrtMemcpy(qDev, elems * 2, hq.data(), elems * 2, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_OK(aclrtMemcpy(kDev, elems * 2, hk.data(), elems * 2, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_OK(aclrtMemcpy(vDev, elems * 2, hv.data(), elems * 2, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_OK(aclrtMemcpy(maskDev, 128 * 128 * 2, hmask.data(), 128 * 128 * 2, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_OK(aclrtMemset(oDev, elems * 2, 0, elems * 2));

    const uint32_t numTask = B * ((N + B_q - 1) / B_q) * H;
    uint32_t coreNum = o.blockDim;
    if (coreNum == 0) {
        coreNum = (numTask < 20u) ? numTask : 20u;
        if (coreNum == 0) coreNum = 1;
    }

    FATTiling tiling{};
    tiling.N_q = N;
    tiling.N_k = N;
    tiling.B_q = B_q;
    tiling.B_k = B_k;
    tiling.head_num = H;
    tiling.head_dim = D;
    tiling.batch_num = B;
    tiling.numTask = numTask;
    tiling.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(D)));

    // ---------------- workspace（公式必须和 catlass_kernel.asc 里的一致） ----------------
    const size_t wsSize = catlass_kernel::get_fat_workspace_size(coreNum, B_q, B_k, D);
    void* wsDev = nullptr;
    ACL_OK(aclrtMalloc(&wsDev, wsSize, ACL_MEM_MALLOC_HUGE_FIRST));
    printf("[host] coreNum=%u  numTask=%u  workspace=%.1f KB\n", coreNum, numTask, wsSize / 1024.0);

    // 三段偏移必须与 kernel 里的 spOffset / oTmpOffset 寻址一致（公式见 fat_ws）
    size_t off = 0;
    const size_t sBytes = fat_ws::BytesS(coreNum, B_q, B_k);
    const size_t pBytes = fat_ws::BytesP(coreNum, B_q, B_k);
    uint8_t* pS = static_cast<uint8_t*>(wsDev) + off;
    off += sBytes;
    uint8_t* pP = static_cast<uint8_t*>(wsDev) + off;
    off += pBytes;
    uint8_t* pOt = static_cast<uint8_t*>(wsDev) + off;
    off += fat_ws::BytesOt(coreNum, B_q, D);
    ACL_OK(aclrtMemcpy(tilingDev, sizeof(FATTiling), &tiling, sizeof(FATTiling), ACL_MEMCPY_HOST_TO_DEVICE));

    // ---------------- launch ----------------
    printf("[host] launching ...\n");
    fflush(stdout);
    catlass_kernel::fat(hardwareSyncAddr, coreNum, stream,
                        static_cast<uint8_t*>(qDev), static_cast<uint8_t*>(kDev),
                        static_cast<uint8_t*>(vDev), static_cast<uint8_t*>(maskDev),
                        static_cast<uint8_t*>(oDev), pS, pP, pOt,
                        static_cast<uint8_t*>(tilingDev));
    printf("[host] launched, waiting ...\n");
    fflush(stdout);
    ACL_OK(aclrtSynchronizeStream(stream));
    printf("[host] kernel finished\n");

    // ---------------- 校验 ----------------
    std::vector<uint16_t> ho(elems, 0);
    ACL_OK(aclrtMemcpy(ho.data(), elems * 2, oDev, elems * 2, ACL_MEMCPY_DEVICE_TO_HOST));
    std::vector<uint16_t> hgold;
    Golden(o, hq, hk, hv, hgold);

    double maxErr = 0.0;
    size_t badIdx = 0;
    for (size_t i = 0; i < elems; ++i) {
        const double e = std::fabs(static_cast<double>(F16ToF32Exact(ho[i])) -
                                   static_cast<double>(F16ToF32Exact(hgold[i])));
        if (e > maxErr) {
            maxErr = e;
            badIdx = i;
        }
    }
    printf("max_err = %.6f   (at %zu: got %.4f expect %.4f)\n", maxErr, badIdx,
           static_cast<double>(F16ToF32Exact(ho[badIdx])), static_cast<double>(F16ToF32Exact(hgold[badIdx])));

    // ---------------- 数值定位 dump：把 S / P / O 的第一行打出来 ----------------
    {
        std::vector<float> hS(8, -12345.f);
        std::vector<uint16_t> hP(8, 0xFC00);   // -inf 占位
        ACL_OK(aclrtMemcpy(hS.data(), 8 * sizeof(float), pS, 8 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
        ACL_OK(aclrtMemcpy(hP.data(), 8 * 2, pP, 8 * 2, ACL_MEMCPY_DEVICE_TO_HOST));
        printf("S[0][0..7] =");
        for (int i = 0; i < 8; ++i) printf(" %.4f", hS[i]);
        printf("\n");
        printf("P[0][0..7] =");
        for (int i = 0; i < 8; ++i) printf(" %.6f", (double)F16ToF32Exact(hP[i]));
        printf("\n");

        // golden 的 S 第一行（未 softmax，带 scale）
        printf("Sgold[0][0..7] =");
        for (int j = 0; j < 8; ++j) {
            double acc = 0;
            for (int d = 0; d < D; ++d)
                acc += (double)F16ToF32Exact(hq[0 * H * D + 0 * D + d]) *
                       (double)F16ToF32Exact(hk[j * H * D + 0 * D + d]);
            printf(" %.4f", acc / std::sqrt(static_cast<double>(D)));
        }
        printf("\n");
        printf("O[0][0..7]  =");
        for (int d = 0; d < 8; ++d) printf(" %.4f", (double)F16ToF32Exact(ho[0 * H * D + d]));
        printf("\n");
        printf("Ogold[0][0..7]=");
        for (int d = 0; d < 8; ++d) printf(" %.4f", (double)F16ToF32Exact(hgold[0 * H * D + d]));
        printf("\n");
    }

    aclrtFree(qDev); aclrtFree(kDev); aclrtFree(vDev); aclrtFree(oDev);
    aclrtFree(maskDev); aclrtFree(tilingDev); aclrtFree(wsDev);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return (maxErr < 2e-2) ? 0 : 1;
}
