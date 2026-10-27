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

target_link_libraries(catlass_fat
  ${TORCH_LIBRARIES}
  torch_npu
  ascendc_runtime
  runtime
  ascendcl
  tiling_api
  platform
  dl)
