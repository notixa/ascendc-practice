/*
 * clangd_prelude.h —— 只给 clangd 用，不参与任何真实编译。
 *
 * 背景：
 *   bisheng（ccec）编译器会隐式预包含 kernel_operator.h，所以 CATLASS 的
 *   examples 里敢于先 include "catlass/gemm/kernel/basic_matmul.hpp"（它用到
 *   AscendC::TPosition）、后 include "catlass/catlass.hpp"。用普通 clang 解析时
 *   就会报 "use of undeclared identifier 'AscendC'"。
 *
 *   解决办法就是模仿 bisheng：用 -include<本文件> 强行排到 TU 最前面，
 *   让 AscendC 命名空间在所有 CATLASS 头之前就绪。
 *
 * 为什么走 ASCENDC_CPU_DEBUG=1 这条路：
 *   CANN 的 CPU 仿真模式（tikicpulib）本来就是为「用 host 编译器解析 kernel」
 *   设计的，它会把 bisheng 内建符号补齐：
 *     - half / bfloat16_t            -> kernel_fp16.h / kernel_bf16.h
 *     - pipe_t / event_t / PIPE_*    -> stub_fun.h
 *     - __global__ / __aicore__ 等   -> stub_def.h
 *   于是不需要我们再手写一份打桩头，clangd 的 hover / 跳转直接落在 CANN 真实
 *   声明上，比假类型有用得多。
 */
#ifndef CATLASS_LAB_CLANGD_PRELUDE_H
#define CATLASS_LAB_CLANGD_PRELUDE_H

#if defined(__clang__) && !defined(__CCE__)
#include <kernel_operator.h>
#endif

#endif // CATLASS_LAB_CLANGD_PRELUDE_H
