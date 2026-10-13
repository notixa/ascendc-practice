#!/bin/bash

# 确保进入脚本所在目录
CURRENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
cd "$CURRENT_DIR"

echo "=========================================================="
echo ">>> [1/3] 开始初始化 Ascend C 算子工程环境..."
echo "=========================================================="

# 默认针对 Atlas 200I DK A2 芯片型号 Ascend310B1
SOC_VERSION="Ascend310B1"

# 解析命令行参数 (支持 -v 或 --soc-version 覆盖默认值)
SHORT=v:,
LONG=soc-version:,
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@" 2>/dev/null)
if [ $? -eq 0 ]; then
    eval set -- "$OPTS"
    while :; do
        case "$1" in
        -v | --soc-version)
            SOC_VERSION="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            break
            ;;
        esac
    done
fi

# 1. 确定 CANN Toolkit 根路径
if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH="$ASCEND_INSTALL_PATH"
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH="$ASCEND_HOME_PATH"
elif [ -d "/usr/local/Ascend/ascend-toolkit/latest" ]; then
    _ASCEND_INSTALL_PATH="/usr/local/Ascend/ascend-toolkit/latest"
elif [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
    _ASCEND_INSTALL_PATH="$HOME/Ascend/ascend-toolkit/latest"
else
    echo "[ERROR]: 未找到 Ascend CANN Toolkit 安装目录！"
    exit 1
fi

# 2. 极速设置环境变量（避免 bin/setenv.bash 触发全盘递归 find 的 Bug）
if command -v bisheng >/dev/null 2>&1 && [ -n "$ASCEND_TOOLKIT_HOME" ]; then
    echo "[INFO]: 当前 Shell 已包含 CANN 环境变量，直接复用。"
elif [ -f "$(dirname "$_ASCEND_INSTALL_PATH")/set_env.sh" ]; then
    echo "[INFO]: 加载官方轻量环境脚本: $(dirname "$_ASCEND_INSTALL_PATH")/set_env.sh"
    source "$(dirname "$_ASCEND_INSTALL_PATH")/set_env.sh" >/dev/null 2>&1 || true
else
    echo "[INFO]: 注入 CANN 标准环境变量..."
    export ASCEND_TOOLKIT_HOME="$_ASCEND_INSTALL_PATH"
    export ASCEND_HOME_PATH="$_ASCEND_INSTALL_PATH"
    export ASCEND_OPP_PATH="$_ASCEND_INSTALL_PATH/opp"
    export TOOLCHAIN_HOME="$_ASCEND_INSTALL_PATH/toolkit"
    export PATH="$_ASCEND_INSTALL_PATH/bin:$_ASCEND_INSTALL_PATH/compiler/ccec_compiler/bin:$PATH"
    export LD_LIBRARY_PATH="$_ASCEND_INSTALL_PATH/lib64:$_ASCEND_INSTALL_PATH/lib64/plugin/opskernel:$_ASCEND_INSTALL_PATH/lib64/plugin/nnengine:$LD_LIBRARY_PATH"
    export PYTHONPATH="$_ASCEND_INSTALL_PATH/python/site-packages:$PYTHONPATH"
fi

echo "[INFO]: 目标芯片 (SOC_VERSION) : ${SOC_VERSION}"
echo "[INFO]: CANN 安装目录           : ${_ASCEND_INSTALL_PATH}"
echo "=========================================================="
echo ">>> [2/3] 清理并构建工程 (BiSheng 设备端编译 + Host 链接)..."
echo "=========================================================="

rm -rf build
mkdir -p build

cmake -B build \
    -DSOC_VERSION=${SOC_VERSION} \
    -DASCEND_CANN_PACKAGE_PATH=${_ASCEND_INSTALL_PATH}

if [ $? -ne 0 ]; then
    echo "[ERROR]: CMake 配置失败！"
    exit 1
fi

cmake --build build -j$(nproc 2>/dev/null || echo 4)
if [ $? -ne 0 ]; then
    echo "[ERROR]: 编译失败！"
    exit 1
fi

if [ ! -f "./build/main" ]; then
    echo "[ERROR]: 未生成 ./build/main 可执行文件！"
    exit 1
fi

echo "=========================================================="
echo ">>> [3/3] 正在板端 NPU 执行算子..."
echo "=========================================================="

./build/main

echo "=========================================================="
echo "[INFO]: 运行完成！"
echo "=========================================================="
