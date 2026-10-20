#pragma once

// ---------------------------------------------------------------------------
// 仅供 clangd（Zed LSP）使用的 AscendC 最小替身头文件，不参与任何真实编译。
//
// 为什么不直接 -include kernel_operator.h？
//   kernel_operator.h 依赖 bisheng 编译器内建的类型（pipe_t / event_t / PIPE_*），
//   clang 不认识它们，强制引入会立刻爆 1w+ 错误，把 clangd 的 AST 直接打断。
//
// 这里只补 CATLASS 头文件在 **声明层** 真正需要的东西：位置标记、同步事件、
// Tensor 类型、以及一批"签名随便"的运行时函数桩。目标不是跑得起来，
// 而是让 clangd 能把 CATLASS 的模板/类型全部索引起来 —— 跳转、hover、补全可用即可。
// ---------------------------------------------------------------------------

#if defined(__clang__) && !defined(__CCE__)

#include <cstdint>

// 技巧：预先占住 kernel_operator.h 的 include guard，让它在 clangd 下变成空操作。
// 否则真实 CANN 头会与下面的替身重复声明（TPosition / RoundMode / HardEvent 等），
// 而这些头本身又依赖 pipe_t、event_t 等 bisheng 内建，注定解析不完整。
#ifndef ASCENDC_MODULE_OPERATOR_H
#define ASCENDC_MODULE_OPERATOR_H
#endif

// bisheng 关键字 → clang 不认识，退化成普通 inline
#ifndef __forceinline__
#define __forceinline__ inline
#endif

// bisheng 把 half 当内建类型，clang 没有；_Float16 足够让类型推导走通
using half = _Float16;

// ---- 编译器内建：流水类型 / 事件 ID（bisheng 里是全局内建枚举）----

// ---- 编译器内建：全局地址类型 ----
using GM_ADDR = uint8_t*;

// ---- 低精度类型（bisheng 内建，clang 用近似类型代替，只为类型推导）----
// 注意：bfloat16_t 必须与 half 是"不同类型"，否则 CATLASS / CANN 里
// ElementAccumulatorSelector<half, half> 与 <bfloat16_t, bfloat16_t> 两套特化会撞车
using bfloat16_t = __bf16;
struct int4b_t {
    int8_t v;
};
struct fp8_e8m0_t {
    uint8_t v;
};
struct hifloat8_t {
    uint8_t v;
};
struct mx_fp8_e4m3_t {
    uint8_t v;
};
struct mx_fp8_e5m2_t {
    uint8_t v;
};

inline void pipe_barrier(int) {}

enum pipe_t {
    PIPE_S = 0,
    PIPE_V,
    PIPE_M,
    PIPE_MTE1,
    PIPE_MTE2,
    PIPE_MTE3,
    PIPE_FIX,
    PIPE_ALL,
};

enum event_t {
    EVENT_ID0 = 0,
    EVENT_ID1,
    EVENT_ID2,
    EVENT_ID3,
    EVENT_ID4,
    EVENT_ID5,
    EVENT_ID6,
    EVENT_ID7,
};

namespace AscendC {

// 低精度类型在 CANN 里既能全局用、也有 AscendC:: 前缀的引用
using ::bfloat16_t;
using ::fp8_e8m0_t;
using ::hifloat8_t;
using ::int4b_t;
using ::mx_fp8_e4m3_t;
using ::mx_fp8_e5m2_t;

// ---- 事件 ID（TEventID 是 event_t 的强类型版本）----
using TEventID = event_t;

// ---- 队列位置：CANN 里就是 TPosition 的别名 ----
// ---- 位置标记（CATLASS 用作模板非类型参数）----
enum TPosition {
    GM = 0,
    VECIN,
    VECOUT,
    VECCALC,
    A1,
    A2,
    B1,
    B2,
    C1,
    C2,
    C2PIPE2GM,
    CO1,
    CO2,
};

// CANN 里 QuePosition 就是 TPosition 的别名
using QuePosition = TPosition;

// ---- 量化模式 ----
enum QuantMode_t {
    NoQuant = 0,
    DEQF16,
    REQ8,
    VDEQF16,
    VREQ8,
    F322BF16,
    F322F16,
    QF322B8_PRE,
    QF322BF16_PRE,
    QF322F16_PRE,
    QF322F32_PRE,
    QS322BF16_PRE,
    VQF322B8_PRE,
    VQF322BF16_PRE,
    VQF322F16_PRE,
    VQF322F32_PRE,
    VQS322BF16_PRE,
};

// ---- 核心类型标记（g_coreType 的取值）----
enum CoreType { AIC = 0, AIV = 1 };
inline constexpr CoreType g_coreType = AIC;

// ---- 搬运 / 计算参数结构体（字段名对齐 CANN kernel_struct_*.h）----
struct LoadData2DParams {
    template <typename... Args>
    LoadData2DParams(Args&&...)
    {
    }
    uint16_t startIndex;
    uint8_t repeatTimes;
    uint16_t srcStride;
    uint8_t sid;
    uint16_t dstGap;
    bool ifTranspose;
    uint8_t addrMode;
};

struct LoadData2dTransposeParams {
    template <typename... Args>
    LoadData2dTransposeParams(Args&&...)
    {
    }
    uint16_t startIndex;
    uint8_t repeatTimes;
    uint16_t srcStride;
    uint16_t dstGap;
    uint16_t dstFracGap;
    uint8_t addrMode;
};

template <typename T>
struct LoadData3DParamsV2 {
    uint16_t channelSize;
    uint16_t enL1;
    uint16_t enL0;
    uint16_t kExtension;
    uint16_t mExtension;
    uint16_t strideW;
    uint16_t strideH;
    uint16_t filterW;
    uint16_t filterH;
    uint16_t dilationW;
    uint16_t dilationH;
};

struct DataCopyExtParams {
    template <typename... Args>
    DataCopyExtParams(Args&&...)
    {
    }
    uint16_t blockCount;
    uint32_t blockLen;
    uint32_t srcStride;
    uint32_t dstStride;
    uint32_t rsv;
};

template <typename T = uint8_t>
struct DataCopyPadExtParams {
    template <typename... Args>
    DataCopyPadExtParams(Args&&...)
    {
    }
    bool isPad;
    uint8_t leftPadding;
    uint8_t rightPadding;
    uint8_t paddingValue;
};

struct FixpipeParamsV220 {
    bool cmatrixInitVal;
    template <typename... Args>
    FixpipeParamsV220(Args&&...)
    {
    }
    uint16_t nSize;
    uint16_t mSize;
    uint16_t srcStride;
    uint16_t dstStride;
    bool unitFlag;
    uint8_t quantPre;
};

struct MmadParams {
    bool reluEn;
    template <typename... Args>
    MmadParams(Args&&...)
    {
    }
    bool isBias;
    uint16_t m;
    uint16_t n;
    uint16_t k;
    bool unitFlag;
};

enum class FmatrixMode { FMATRIX_LEFT, FMATRIX_RIGHT };
enum class PaddingMode { PAD_NONE = 0, PAD_MODE0, PAD_MODE1, PAD_MODE2, PAD_MODE3 };
enum class MaskMode { MASK_NONE = 0, MASK_NORMAL, MASK_COUNT };
enum class ReduceOrder { ORDER_VALUE_INDEX = 0, ORDER_INDEX_VALUE };
enum class CacheMode { CACHE_NONE = 0, CACHE_HALF, CACHE_ALL };

// ---- 同步事件（与 CANN impl/kernel_event.h 保持同名）----
enum class HardEvent : uint8_t {
    MTE2_MTE1,
    MTE1_MTE2,
    MTE1_M,
    M_MTE1,
    MTE2_V,
    V_MTE2,
    MTE3_V,
    V_MTE3,
    M_V,
    V_M,
    V_V,
    MTE3_MTE1,
    MTE1_MTE3,
    MTE1_V,
    MTE2_M,
    M_MTE2,
    V_MTE1,
    M_FIX,
    FIX_M,
    MTE3_MTE2,
    MTE2_MTE3,
    S_V,
    V_S,
    S_MTE2,
    MTE2_S,
    S_MTE3,
    MTE3_S,
    MTE2_FIX,
    FIX_MTE2,
    FIX_S,
    M_S,
    FIX_MTE3,
};

enum RoundMode { CAST_NONE = 0, CAST_RINT, CAST_FLOOR, CAST_CEIL, CAST_ROUND, CAST_TRUNC, CAST_ODD };
enum ReduceMode { REDUCE_SUM = 0, REDUCE_MAX, REDUCE_MIN };

// ---- ShapeInfo（Tensor 的 shape 描述，CATLASS 只当类型用）----
struct ShapeInfo {
    uint32_t shape[8];
    uint32_t originalShape[8];
};

// ---- 参数结构体（CATLASS 里只当作类型传递）----
struct Nd2NzParams {
    template <typename... Args>
    Nd2NzParams(Args&&...)
    {
    }
    uint16_t ndNum;
    uint16_t nValue;
    uint16_t dValue;
    uint16_t srcNdMatrixStride;
    uint16_t srcDValue;
    uint16_t dstNzC0Stride;
    uint16_t dstNzNStride;
    uint16_t dstNzMatrixStride;
};

struct DataCopyParams {
    template <typename... Args>
    DataCopyParams(Args&&...)
    {
    }
    uint16_t blockCount;
    uint16_t blockLen;
    uint16_t srcStride;
    uint16_t dstStride;
};

// ---- Tensor：CATLASS 主要拿它当类型用 ----
template <typename T>
class LocalTensor {
public:
    T* data() const
    {
        return nullptr;
    }
    T* GetPhyAddr() const
    {
        return nullptr;
    }
    constexpr uint32_t GetSize() const
    {
        return 0;
    }
    template <typename U>
    LocalTensor<U> ReinterpretCast() const
    {
        return LocalTensor<U>{};
    }
    // AscendC 的 operator[] 是"切片"，返回子张量视图（不是元素引用）
    LocalTensor<T> operator[](uint32_t) const
    {
        return LocalTensor<T>{};
    }
    T GetValue(uint32_t) const
    {
        return T{};
    }
    void SetValue(uint32_t, T)
    {
    }
};

template <typename T>
class GlobalTensor {
public:
    GlobalTensor() = default;
    GlobalTensor(T*) {}

    T* data() const
    {
        return nullptr;
    }
    T* GetPhyAddr() const
    {
        return nullptr;
    }
    template <typename... Args>
    void SetGlobalBuffer(Args&&...)
    {
    }
    T GetValue(uint32_t) const
    {
        return T{};
    }
    void SetValue(uint32_t, T)
    {
    }
    GlobalTensor<T> operator[](uint32_t) const
    {
        return GlobalTensor<T>{};
    }
};


// ---- 其余 repeat / fixpipe 参数结构体：只保证能通过语法检查 ----
struct CopyRepeatParams { uint32_t dstStride; uint32_t srcStride; uint16_t repeatTimes; };
struct UnaryRepeatParams { uint32_t dstBlkStride; uint32_t srcBlkStride; uint32_t dstRepStride; uint32_t srcRepStride; };
struct BinaryRepeatParams {
    uint32_t dstBlkStride;
    uint32_t src0BlkStride;
    uint32_t src1BlkStride;
    uint32_t dstRepStride;
    uint32_t src0RepStride;
    uint32_t src1RepStride;
};
struct BrcbRepeatParams { uint32_t dstBlkStride; uint32_t dstRepStride; };
struct SelectWithBytesMaskShapeInfo { uint32_t firstDim; uint32_t lastDim; uint32_t srcLastDim; };
struct SoftMaxShapeInfo { uint32_t srcM; uint32_t srcK; uint32_t oriSrcM; uint32_t oriSrcK; };
struct FixpipeConfig { bool isNz2nd; };
struct DataCopyCO12DstParams { uint16_t nSize; uint16_t mSize; uint32_t dstStride; uint32_t srcStride; };
struct InitConstValueParams { uint16_t repeatTimes; uint16_t blockNum; uint16_t dstGap; };
struct Dn2NzParams { uint16_t dnNum; uint16_t nValue; uint16_t dValue; uint16_t srcDValue; uint16_t dstNzC0Stride; };
struct Nz2DnParams { uint16_t ndNum; uint16_t nValue; uint16_t dValue; uint16_t srcNzC0Stride; uint16_t dstDValue; };
struct LoadData2DParamsV2 {
    uint16_t mStartPosition;
    uint16_t kStartPosition;
    uint16_t mStep;
    uint16_t kStep;
    uint8_t srcStride;
    uint8_t dstStride;
    uint8_t ifTranspose;
};
struct LoadData2DMxParams { uint16_t startIndex; uint8_t repeatTimes; uint16_t srcStride; uint16_t dstGap; };
struct LoadData3DParamsV2Pro {
    uint16_t channelSize;
    uint16_t kExtension;
    uint16_t mExtension;
};
struct TransDataTo5HDParams { uint16_t repeatTimes; uint8_t dstRepStride; uint8_t srcRepStride; bool dstHighHalf; bool srcHighHalf; };
struct FixpipeParamsC310 {
    uint16_t nSize;
    uint16_t mSize;
    uint16_t srcStride;
    bool unitFlag;
    uint8_t quantPre;
};


template <TPosition pos = TPosition::VECCALC, int32_t depth = 2, typename T = uint8_t>
class TQue {
public:
    template <typename... Args>
    LocalTensor<T> AllocTensor(Args&&...)
    {
        return LocalTensor<T>{};
    }
    template <typename U = T, typename... Args>
    void EnQue(LocalTensor<U>, Args&&...)
    {
    }
    template <typename U = T, typename... Args>
    LocalTensor<U> DeQue(Args&&...)
    {
        return LocalTensor<U>{};
    }
    template <typename U = T, typename... Args>
    void FreeTensor(LocalTensor<U>, Args&&...)
    {
    }
    template <typename... Args>
    void Init(Args&&...)
    {
    }
};

// ---- 片上 buffer / pipe：CATLASS 的 LocalTensorBuffer 用 ----
template <TPosition pos = TPosition::VECCALC, typename T = uint8_t>
class TBuf {
public:
    template <typename U = T>
    LocalTensor<U> Get()
    {
        return LocalTensor<U>{};
    }
    template <typename U = T>
    LocalTensor<U> GetWithOffset(uint32_t)
    {
        return LocalTensor<U>{};
    }
};

class TPipe {
public:
    template <typename... Args>
    void InitBuffer(Args&&...)
    {
    }
    template <typename... Args>
    void Init(Args&&...)
    {
    }
    void Destroy() {}
    ~TPipe() = default;
};

inline TPipe* GetTPipePtr()
{
    static TPipe pipe;
    return &pipe;
}

// ---- 流水同步：两个可变参数包，既支持 SetFlag<HardEvent::X>(EVENT_ID0) 也支持裸调用 ----
template <auto... Es, typename... Args>
inline void SetFlag(Args&&...)
{
}
template <auto... Es, typename... Args>
inline void WaitFlag(Args&&...)
{
}
template <auto... Es, typename... Args>
inline void PipeBarrier(Args&&...)
{
}
template <auto... Es, typename... Args>
inline void CrossCoreSetFlag(Args&&...)
{
}
template <auto... Es, typename... Args>
inline void CrossCoreWaitFlag(Args&&...)
{
}
template <auto... Es, typename... Args>
inline void DataCopy(Args&&...)
{
}
template <auto... Es, typename... Args>
inline void DataCopyPad(Args&&...)
{
}
template <auto... Es, typename... Args>
inline void Cast(Args&&...)
{
}
template <auto... Es, typename... Args>
inline void LoadData(Args&&...)
{
}
template <auto... Es, typename... Args>
inline void SetVectorMask(Args&&...)
{
}


template <auto... Es, typename... Args>
inline void Mmad(Args&&...)
{
}
template <auto... Es, typename... Args>
inline void Fixpipe(Args&&...)
{
}
template <auto... Es, typename... Args>
inline void LoadDataWithTranspose(Args&&...)
{
}
template <auto... Es, typename... Args>
inline void SetFmatrix(Args&&...)
{
}
template <auto... Es, typename... Args>
inline void SetAtomicAdd(Args&&...)
{
}
template <auto... Es, typename... Args>
inline void SetAtomicNone(Args&&...)
{
}
template <auto... Es, typename... Args>
inline void SetAtomicType(Args&&...)
{
}
template <auto... Es, typename... Args>
inline void Muls(Args&&...)
{
}
template <auto... Es, typename... Args>
inline void IsResetLoad3dConfig(Args&&...)
{
}

// ---- 常用常量（CANN 里的 block / fractal 尺寸）----
constexpr uint32_t ONE_BLK_SIZE = 32;
constexpr uint32_t ONE_BLOCK_SIZE = 32;
constexpr uint32_t ONE_REPEAT_BYTE_SIZE = 256;
constexpr uint32_t VECTOR_REG_WIDTH = 256;
constexpr uint32_t BLOCK_CUBE = 16;
constexpr uint32_t DEFAULT_REPEAT_STRIDE = 8;
constexpr uint32_t HALF_REPEAT_STRIDE = 4;
constexpr uint32_t B16_BYTE_SIZE = 2;
constexpr uint32_t B32_BYTE_SIZE = 4;

template <auto... Es, typename... Args>
inline void SetSyncBaseAddr(Args&&...)
{
}

inline uint32_t GetBlockIdx()
{
    return 0;
}
inline uint32_t GetBlockNum()
{
    return 1;
}
inline uint32_t GetSubBlockIdx()
{
    return 0;
}
inline uint32_t GetSubBlockNum()
{
    return 2;
}

} // namespace AscendC

// 这些符号在 bisheng 下是全局可见的，CATLASS 头里也有不带 AscendC:: 的写法
using AscendC::GetTPipePtr;

template <typename Dst, typename... Args>
inline Dst ReinterpretCast(Args&&...)
{
    return {};
}

using QuantMode_t = AscendC::QuantMode_t;
using CoreType = AscendC::CoreType;
inline constexpr AscendC::CoreType g_coreType = AscendC::AIC;

#endif
