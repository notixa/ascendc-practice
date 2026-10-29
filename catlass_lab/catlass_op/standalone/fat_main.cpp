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

static bool gBrokenMask = false;

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
                if (o.causal) {
                    // kernel 的 mask 表按【块内局部坐标】索引：行 = q 块内行号，列 = kv 块内列号
                    const uint32_t iBlk = static_cast<uint32_t>(p / 128), rLow = static_cast<uint32_t>(p % 128);
                    const uint32_t kBlk = static_cast<uint32_t>(t / 128), cLow = static_cast<uint32_t>(t % 128);
                    bool masked;
                    if (kBlk > iBlk) {
                        masked = true;                      // 未来的 kv 块，全屏蔽
                    } else if (kBlk == iBlk) {
                        const uint32_t pat = gBrokenMask ? (rLow % 64) : rLow;
                        masked = (cLow > pat);              // 对角块：列 > 行 就屏蔽
                    } else {
                        masked = false;                     // 过去的 kv 块，全可见
                    }
                    if (masked) { s[static_cast<size_t>(p) * N + t] = -1e30; continue; }
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
    int brokenMask = 0;
    // 用法: fat_bench N H causal blockDim [brokenMask]
    //   brokenMask=1 时把 mask 表的 64..127 行替换成 0..63 行（验证 sb1 是否读到 mask 前 64 行）
    if (argc > 5) brokenMask = atoi(argv[5]);
    if (argc > 4) o.blockDim = static_cast<uint32_t>(atoi(argv[4]));
    const uint32_t B_q = 128, B_k = 128;
    const uint32_t B = o.batch, N = o.n, H = o.h, D = o.d;

    gBrokenMask = (brokenMask != 0);
    printf("=== fat_bench  N=%u H=%u D=%u B=%u causal=%d brokenMask=%d ===\n", N, H, D, B, o.causal, brokenMask);

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
            const uint32_t pat = brokenMask ? (i % 64) : i;   // 验证：假设 sb1 读到的是 mask 0..63 行
            for (uint32_t j = 0; j < 128; ++j) {
                hmask[i * 128 + j] = (j > pat) ? 0x3C00 /* 1.0 */ : 0x0000;
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
    tiling.causal = (o.causal != 0) ? 1u : 0u;   // ★ 之前漏了：FATTiling tiling{} 零初始化 → 恒为 0

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

    // ---- 两 stack 分解的期望值：验证 oTmp(stack0)/oTmp(stack1)/gl 哪个错 ----
    if (o.causal == 0 && N >= 640 && H == 1) {   // 第二个 stack 从 key 512 开始，期望值同构
        const size_t split = 512;              // STACK_NUM*B_k：段1 第一次调用覆盖 keys 0..511
        auto dotp = [&](size_t t) {
            double a = 0.0;
            for (uint32_t d = 0; d < D; ++d) {
                a += static_cast<double>(F16ToF32Exact(hq[d])) *
                     static_cast<double>(F16ToF32Exact(hk[t * D + d]));
            }
            return a / std::sqrt(static_cast<double>(D));
        };
        double m0 = -1e30, m1 = -1e30;
        for (size_t t = 0; t < split; ++t) m0 = std::max(m0, dotp(t));
        for (size_t t = split; t < N; ++t) m1 = std::max(m1, dotp(t));
        double gl0 = 0.0, gl1 = 0.0, ot0 = 0.0, ot1 = 0.0;
        for (size_t t = 0; t < split; ++t) {
            const double e = std::exp(dotp(t) - m0);
            gl0 += e;
            ot0 += e * static_cast<double>(F16ToF32Exact(hv[t * D + 0]));
        }
        for (size_t t = split; t < N; ++t) {
            const double e = std::exp(dotp(t) - m1);
            gl1 += e;
            ot1 += e * static_cast<double>(F16ToF32Exact(hv[t * D + 0]));
        }
        const double M = std::max(m0, m1);
        const double w0 = std::exp(m0 - M), w1 = std::exp(m1 - M);
        const double glT = gl0 * w0 + gl1 * w1;
        const double goT = ot0 * w0 + ot1 * w1;
        printf("EXPECT: oTmp0[0]=%.6f (E6 应等于它)  gl0=%.6f\n", ot0, gl0);
        printf("EXPECT: oTmp1[0]=%.6f (E7 应等于它)  gl1=%.6f\n", ot1, gl1);
        printf("EXPECT: m0=%.6f m1=%.6f  dm=%.6f  gl_total=%.6f  go_total=%.6f  O[0]=%.6f\n",
               m0, m1, w0, glT, goT, goT / glT);
        // 第二个 stack 各行（0..3）的真实 local max，用于和内核 E1 的 lm[0..3] 对比
        printf("EXPECT m1(row0..3):");
        for (size_t r = 0; r < 4; ++r) {
            double mx = -1e30;
            for (size_t t = split; t < N; ++t) {
                double a = 0.0;
                for (uint32_t d = 0; d < D; ++d) {
                    a += static_cast<double>(F16ToF32Exact(hq[r * D + d])) *
                         static_cast<double>(F16ToF32Exact(hk[t * D + d]));
                }
                mx = std::max(mx, a / std::sqrt(static_cast<double>(D)));
            }
            printf(" %.6f", mx);
        }
        printf("\n");
        printf("GOLDEN: O[0][0]=%.6f\n", static_cast<double>(F16ToF32Exact(hgold[0])));

        // GM 里残留的 S 是【最后一个 stack】的：N=768 时 = keys 512..767（未乘 scale）
        printf("EXPECT S(keys 512..519, 未scale):");
        for (size_t j = 0; j < 8; ++j) {
            double a = 0.0;
            for (uint32_t d = 0; d < D; ++d) {
                a += static_cast<double>(F16ToF32Exact(hq[d])) *
                     static_cast<double>(F16ToF32Exact(hk[(split + j) * D + d]));
            }
            printf(" %.4f", a);
        }
        printf("\n");
        // P = exp(S - m1)，未归一化（fp16 存于 GM，最后一个 stack 的 P）
        printf("EXPECT P(512..519) = exp(s-m1):");
        for (size_t j = 0; j < 8; ++j) {
            double a = 0.0;
            for (uint32_t d = 0; d < D; ++d) {
                a += static_cast<double>(F16ToF32Exact(hq[d])) *
                     static_cast<double>(F16ToF32Exact(hk[(split + j) * D + d]));
            }
            printf(" %.6f", std::exp(a / std::sqrt(static_cast<double>(D)) - m1));
        }
        printf("\n");

        // 同时给一份 keys 0..7 的，作为"取错偏移"的对照
        printf("EXPECT S(keys 0..7, 未scale)   :");
        for (size_t j = 0; j < 8; ++j) {
            double a = 0.0;
            for (uint32_t d = 0; d < D; ++d) {
                a += static_cast<double>(F16ToF32Exact(hq[d])) *
                     static_cast<double>(F16ToF32Exact(hk[j * D + d]));
            }
            printf(" %.4f", a);
        }
        printf("\n");
    }

    // ---- 每个 q 块第一行的 O vs golden（配合设备侧 go/gl 打点算"应该是多少"）----
    {
        const size_t rowElems = static_cast<size_t>(H) * D;
        const size_t nBlk = (static_cast<size_t>(N) + 127) / 128;
        for (size_t b = 0; b < nBlk; ++b) {
            const size_t p = b * 128;
            if (p >= static_cast<size_t>(N)) break;
            printf("tok %4zu (i=%zu) O   [0..3] =", p, b);
            for (int d = 0; d < 4; ++d) printf(" %.6f", (double)F16ToF32Exact(ho[p * rowElems + d]));
            printf("\n");
            printf("tok %4zu (i=%zu) Ogold[0..3] =", p, b);
            for (int d = 0; d < 4; ++d) printf(" %.6f", (double)F16ToF32Exact(hgold[p * rowElems + d]));
            printf("\n");
        }
        // oTmp（GM，最后一个 stack 的值）：前 2 行各 4 列
        std::vector<float> ot(16, 0.f);
        ACL_OK(aclrtMemcpy(ot.data(), 8 * sizeof(float), pOt, 8 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
        ACL_OK(aclrtMemcpy(ot.data() + 8, 8 * sizeof(float), pOt + 128, 8 * sizeof(float),
                           ACL_MEMCPY_DEVICE_TO_HOST));
        printf("oTmp row0 [0..7] =");
        for (int d = 0; d < 8; ++d) printf(" %.6f", (double)ot[d]);
        printf("\n");
        printf("oTmp row1 [0..7] =");
        for (int d = 0; d < 8; ++d) printf(" %.6f", (double)ot[8 + d]);
        printf("\n");
    }

    // ---- 按 q 块（每 B_q=128 个 token）分段统计误差，定位是哪个 i 坏 ----
    {
        const size_t rowElems = static_cast<size_t>(H) * D;   // 每个 token 的元素数
        const size_t nBlk = (static_cast<size_t>(N) + 127) / 128;
        printf("per-q-block max_err:");
        for (size_t b = 0; b < nBlk; ++b) {
            double be = 0.0;
            const size_t p0 = b * 128, p1 = std::min<size_t>(p0 + 128, N);
            for (size_t p = p0; p < p1; ++p) {
                for (size_t d = 0; d < rowElems; ++d) {
                    const size_t i = p * rowElems + d;
                    const double e = std::fabs(static_cast<double>(F16ToF32Exact(ho[i])) -
                                               static_cast<double>(F16ToF32Exact(hgold[i])));
                    if (e > be) be = e;
                }
            }
            printf("  i=%zu: %.6f", b, be);

            // 每个 subblock（0..63 / 64..127）分别统计 + 坏行范围
            for (int sb = 0; sb < 2; ++sb) {
                double sbe = 0.0;
                size_t nBad = 0, firstBad = 0, lastBad = 0;
                const size_t r0 = p0 + sb * 64, r1 = std::min<size_t>(p0 + (sb + 1) * 64, p1);
                for (size_t p = r0; p < r1; ++p) {
                    double re = 0.0;
                    for (size_t d = 0; d < rowElems; ++d) {
                        const size_t i = p * rowElems + d;
                        const double e = std::fabs(static_cast<double>(F16ToF32Exact(ho[i])) -
                                                   static_cast<double>(F16ToF32Exact(hgold[i])));
                        if (e > re) re = e;
                    }
                    if (re > sbe) sbe = re;
                    if (re > 0.01) { if (!nBad) firstBad = p - p0; lastBad = p - p0; ++nBad; }
                }
                printf("   [sb%d: %.6f nbad=%zu rows %zu..%zu]", sb, sbe, nBad, firstBad, lastBad);
            }
        }
        printf("\n");
    }

    // ---------------- 数值定位 dump：把 S / P / O 的第一行打出来 ----------------
    {
        std::vector<float> hS(8, -12345.f);
        std::vector<uint16_t> hP(8, 0xFC00);   // -inf 占位
        ACL_OK(aclrtMemcpy(hS.data(), 8 * sizeof(float), pS, 8 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
        ACL_OK(aclrtMemcpy(hP.data(), 8 * 2, pP, 8 * 2, ACL_MEMCPY_DEVICE_TO_HOST));
        printf("S[0][0..7] =");
        for (int i = 0; i < 8; ++i) printf(" %.4f", hS[i]);
        printf("\n");
        // 关键对照：GM 里 S 的第 256..263 列（第二个 stack 时应为 keys 768..775）
        {
            std::vector<float> s256(8, 0.f);
            ACL_OK(aclrtMemcpy(s256.data(), 8 * sizeof(float), pS + 256, 8 * sizeof(float),
                               ACL_MEMCPY_DEVICE_TO_HOST));
            printf("S[0][256..263] 实际        =");
            for (int i = 0; i < 8; ++i) printf(" %.4f", s256[i]);
            printf("\n");
            printf("EXPECT S(keys 768..775)    =");
            for (size_t j = 0; j < 8; ++j) {
                double a = 0.0;
                for (uint32_t d = 0; d < D; ++d) {
                    a += static_cast<double>(F16ToF32Exact(hq[d])) *
                         static_cast<double>(F16ToF32Exact(hk[(split + 256 + j) * D + d]));
                }
                printf(" %.4f", a);
            }
            printf("\n");
            printf("（若与 EXPECT S(512..519) 相同，则说明第 256 列被 key512 覆盖）\n");
        }
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
