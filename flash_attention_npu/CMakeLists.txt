cmake_minimum_required(VERSION 3.16.0)
project(flash_attention_npu)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# Target Atlas 200I DK A2 SoC version
set(SOC_VERSION "Ascend310B1" CACHE STRING "system on chip type")
set(ASCEND_KERNEL_LAUNCH_ONLY ON CACHE BOOL "Only build kernel launch" FORCE)

# Locate CANN Toolkit
if(DEFINED ASCEND_CANN_PACKAGE_PATH)
elseif(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "ASCEND CANN package installation directory" FORCE)
elseif(DEFINED ENV{ASCEND_TOOLKIT_HOME})
    set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_TOOLKIT_HOME}" CACHE PATH "ASCEND CANN package installation directory" FORCE)
elseif(DEFINED ENV{USER} AND "$ENV{USER}" STREQUAL "root")
    set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest" CACHE PATH "ASCEND CANN package installation directory")
else()
    set(ASCEND_CANN_PACKAGE_PATH "$ENV{HOME}/Ascend/ascend-toolkit/latest" CACHE PATH "ASCEND CANN package installation directory")
endif()

# Locate ascendc_kernel_cmake
if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist under ${ASCEND_CANN_PACKAGE_PATH}, please check CANN package installation.")
endif()

include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

# Compile Ascend C kernel into static library libflash_attention_kernel.a
ascendc_library(flash_attention_kernel STATIC
    csrc/flash_attention_fwd_kernel.cpp
    csrc/flash_attention_bwd_dkdv_kernel.cpp
    csrc/flash_attention_bwd_dq_kernel.cpp
)

target_include_directories(flash_attention_kernel_interface INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/csrc
)

