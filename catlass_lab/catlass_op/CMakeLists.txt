cmake_minimum_required(VERSION 3.18)

# ---------------------------------------------------------------------------
# CANN 环境
# ---------------------------------------------------------------------------
if(NOT DEFINED ENV{ASCEND_HOME_PATH})
  message(FATAL_ERROR "ASCEND_HOME_PATH not set（先 source set_env.sh）")
endif()
set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})

find_package(ASC REQUIRED)
project(catlass_fat LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)

# ---------------------------------------------------------------------------
# 架构：910B = 2201
# ---------------------------------------------------------------------------
set(NPU_ARCH "2201" CACHE STRING "NPU architecture")

# 设备侧打印：AscendC::PRINTF 需要 --cce-enable-print，【否则会被编译成空操作】
option(FAT_ENABLE_PRINT "Enable device-side AscendC::PRINTF" OFF)

# 调试用：绕过 torch_npu 的 OpCommand，直接调 kernel wrapper。
# 用来区分「kernel 内部死锁」和「OpCommand 等错了流」。
option(FAT_NO_OPCOMMAND "Bypass torch_npu OpCommand (debug stream deadlock)" OFF)

# 调试用：去掉 AIV 侧的两个跨核 wait（让 AIV 不再等 AIC）
# 目的是判断「AIV 到底有没有被调度」：
#   - 开了它，kernel 能跑完且 O 非 0  → AIV 跑过 ⇒ 之前是跨核 flag 收不到
#   - 开了它，仍然卡死 / O 全 0      → AIV 根本没被调度
option(FAT_NO_SYNC "Drop AIV cross-core waits (debug: is AIV scheduled?)" OFF)

# 调试用：AIC 只做 QK，跳过 PV（配合 FAT_ENABLE_PRINT 用）
# 目的：确认「QK + AIV softmax」这条链路是否完全正常，把问题隔离在 PV
option(FAT_QK_ONLY "AIC does QK only, skip PV (debug)" OFF)

# 调试用：在自己的 include 目录里放了一份带插桩的 block_mmad_fa_pv.hpp，
# 用 include 路径优先来「影子替换」CATLASS 原版（不改 CATLASS 仓库）
option(FAT_TRACE_PV "Use shadowed block_mmad_fa_pv.hpp with step traces (debug)" OFF)

# 诊断：AIV 在 softmax【之前】就置 smReady（判断 AIV->AIC flag 机制是否可用）
option(FAT_SM_EARLY "AIV sets smReady before softmax (debug)" OFF)
if(FAT_SM_EARLY)
  message(STATUS "smReady set BEFORE softmax: ON")
  add_compile_definitions(FAT_SM_EARLY=1)
endif()
if(FAT_TRACE_PV)
  message(STATUS "shadowed FAPV header with traces: ON")
  add_compile_definitions(FAT_TRACE_PV=1)
endif()
if(FAT_QK_ONLY)
  message(STATUS "QK only (PV skipped)")
  add_compile_definitions(FAT_QK_ONLY=1)
endif()
if(FAT_NO_SYNC)
  message(STATUS "AIV cross-core wait disabled")
  add_compile_definitions(FAT_NO_SYNC=1)
endif()
if(FAT_NO_OPCOMMAND)
  message(STATUS "OpCommand bypass: ON")
  add_compile_definitions(FAT_NO_OPCOMMAND=1)
endif()
if(FAT_ENABLE_PRINT)
  message(STATUS "device print: ON (--cce-enable-print)")
  add_compile_options($<$<COMPILE_LANGUAGE:ASC>:--cce-enable-print>)
endif()
add_compile_definitions(CATLASS_ARCH=${NPU_ARCH})
add_compile_options($<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-${NPU_ARCH}>)
add_compile_options(
  $<$<COMPILE_LANGUAGE:ASC>:-Wno-macro-redefined>
  $<$<COMPILE_LANGUAGE:ASC>:-Wno-ignored-attributes>
  $<$<COMPILE_LANGUAGE:ASC>:-Wno-unused-value>)
add_link_options($<$<COMPILE_LANGUAGE:ASC>:-Wno-unused-command-line-argument>)

# ---------------------------------------------------------------------------
# PyTorch / torch_npu
# ---------------------------------------------------------------------------
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
list(APPEND CMAKE_PREFIX_PATH "${Python3_SITELIB}/torch/share/cmake")
find_package(Torch REQUIRED)

# ---------------------------------------------------------------------------
# 包含路径
# ---------------------------------------------------------------------------
# CATLASS 本体（只读参考仓）
set(CATLASS_INCLUDE_DIR
    "/home/notixa/codes/opensource-project/catlass/include"
    CACHE PATH "CATLASS include dir")
# 你的 kernel（fat_kernel.h / fat_types.h / fat_tiling.h 所在目录）
set(FA_KERNEL_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../flash_attention_train"
    CACHE PATH "FATkernel dir")

# ---- fail-fast：路径不对就在这里报，而不是等 make 时报一堆 "file not found" ----
if(NOT EXISTS "${CATLASS_INCLUDE_DIR}/catlass/arch/arch.hpp")
  message(FATAL_ERROR
    "CATLASS_INCLUDE_DIR 无效（找不到 catlass/arch/arch.hpp）: ${CATLASS_INCLUDE_DIR}\n"
    "  请在 cmake 时显式指定： -DCATLASS_INCLUDE_DIR=<catlass 仓库>/include")
endif()
if(NOT EXISTS "${FA_KERNEL_DIR}/fat_types.h")
  message(FATAL_ERROR "FA_KERNEL_DIR 无效（找不到 fat_types.h）: ${FA_KERNEL_DIR}")
endif()

# ---- CANN 的按架构分目录：x86_64-linux / aarch64-linux ----
# find_package(ASC) 会给出 ASCEND_CANN_PACKAGE_LINUX_PATH；没给就自己按目录存在性判断。
if(NOT DEFINED ASCEND_CANN_PACKAGE_LINUX_PATH OR ASCEND_CANN_PACKAGE_LINUX_PATH STREQUAL "")
  if(EXISTS "${ASCEND_HOME_PATH}/aarch64-linux")
    set(ASCEND_CANN_PACKAGE_LINUX_PATH "${ASCEND_HOME_PATH}/aarch64-linux")
  else()
    set(ASCEND_CANN_PACKAGE_LINUX_PATH "${ASCEND_HOME_PATH}/x86_64-linux")
  endif()
endif()
message(STATUS "CANN arch dir: ${ASCEND_CANN_PACKAGE_LINUX_PATH}")

# ⚠️ 顺序有讲究：CANN 的 include 必须排在 torch_npu 之前。
#    原因：torch_npu 的 include 目录里也带了一份 acl/acl.h（精简版），
#    排在前面会抢先匹配，导致 aclrtGetHardwareSyncAddr 等接口"未声明"。
include_directories(
  ${ASCEND_CANN_PACKAGE_LINUX_PATH}/include
  ${ASCEND_HOME_PATH}/include
  ${ASCEND_CANN_PACKAGE_LINUX_PATH}/include/aclnn
  ${ASCEND_CANN_PACKAGE_LINUX_PATH}/asc/include
  ${Python3_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
  ${Python3_SITELIB}/torch_npu/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CATLASS_INCLUDE_DIR}
  ${FA_KERNEL_DIR})

# 就诊：确认编译时到底会用到哪一份 acl/acl.h
foreach(_d ${ASCEND_CANN_PACKAGE_LINUX_PATH}/include ${ASCEND_HOME_PATH}/include
           ${Python3_SITELIB}/torch_npu/include)
  if(EXISTS "${_d}/acl/acl.h")
    message(STATUS "acl/acl.h 可见于: ${_d}/acl/acl.h")
  endif()
endforeach()

link_directories(
  ${ASCEND_HOME_PATH}/lib64
  ${ASCEND_CANN_PACKAGE_LINUX_PATH}/lib64
  ${ASCEND_CANN_PACKAGE_LINUX_PATH}/devlib
  ${Python3_SITELIB}/torch_npu/lib)

# ---------------------------------------------------------------------------
# 目标
# ---------------------------------------------------------------------------
set_source_files_properties(kernel/catlass_kernel.asc PROPERTIES LANGUAGE ASC)

add_library(catlass_fat SHARED
  src/catlass_torch.cpp
  kernel/catlass_kernel.asc)

# 再显式挂一遍（target_include_directories 对所有语言都生效，包括 ASC，
# 避免某些 toolchain 下 include_directories 没传给 .asc 的编译命令）
target_include_directories(catlass_fat PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CATLASS_INCLUDE_DIR}
  ${FA_KERNEL_DIR})

# ---------------------------------------------------------------------------
# 独立可执行（对照 examples/23 的 main）：
#   同样的 kernel，exe 里能跑、shared library + torch 里卡死 → 定位嵌入方式的问题，
#   同时也提供一个「无 Python launcher」的纯净性能口径。
# ---------------------------------------------------------------------------
add_executable(fat_bench standalone/fat_main.cpp)
set_source_files_properties(standalone/fat_main.cpp PROPERTIES LANGUAGE ASC)
target_include_directories(fat_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CATLASS_INCLUDE_DIR}
  ${FA_KERNEL_DIR})
target_link_libraries(fat_bench PRIVATE m)

target_link_libraries(catlass_fat
  ${TORCH_LIBRARIES}
  torch_npu
  ascendc_runtime
  runtime
  ascendcl
  tiling_api
  platform
  dl)
