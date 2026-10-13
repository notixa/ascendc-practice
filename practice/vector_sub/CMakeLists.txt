# CMake lowest version requirement
cmake_minimum_required(VERSION 3.16.0)

# Project information
project(Ascend_C_Vector_Sub)

# Default to Ascend310B1 for Atlas 200I DK A2
set(SOC_VERSION "Ascend310B1" CACHE STRING "system on chip type")

if(DEFINED ENV{USER} AND "$ENV{USER}" STREQUAL "root")
    set(DEFAULT_ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest" CACHE PATH "ASCEND CANN package default installation directory for root user")
else()
    set(DEFAULT_ASCEND_CANN_PACKAGE_PATH "$ENV{HOME}/Ascend/ascend-toolkit/latest" CACHE PATH "ASCEND CANN package default installation directory for other user")
endif()

if(DEFINED ASCEND_CANN_PACKAGE_PATH)
elseif(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "ASCEND CANN package installation directory" FORCE)
else()
    set(ASCEND_CANN_PACKAGE_PATH "${DEFAULT_ASCEND_CANN_PACKAGE_PATH}" CACHE PATH "ASCEND CANN package installation directory")
endif()

set(RUN_MODE "npu" CACHE STRING "run mode: npu")
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "path for install()" FORCE)

if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist, please check whether the CANN package is installed.")
endif()

include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

# ascendc_library generates device fatbin & host stub library
ascendc_library(kernels STATIC
    vector_sub.cpp
)

add_executable(main main.cpp)

target_link_libraries(main PRIVATE
    kernels
)
