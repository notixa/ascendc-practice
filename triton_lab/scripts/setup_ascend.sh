#!/usr/bin/env bash
# =============================================================================
# 一步装好 Triton-Ascend 练习环境（两步，第二步不能省）
#
# 为什么需要第二步：
#   `triton` 和 `triton-ascend` 两个包都会往 triton/_C/ 里写 libtriton.so，
#   uv sync 里上游 triton 通常后装 → 把 triton-ascend 自带的 **含 ascend dialect 的**
#   libtriton.so 覆盖掉。表现就是 `import triton` 报：
#       ModuleNotFoundError: No module named 'triton._C.libtriton.ascend'
#   所以必须再用 --no-deps --force-reinstall 把 triton-ascend 装一遍，让它覆盖回去。
#
# 用法：
#   bash scripts/setup_ascend.sh                 # 默认 3.6.0
#   TA_VERSION=3.2.2 bash scripts/setup_ascend.sh  # 按机器上 CANN 版本换
# =============================================================================
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

TA_VERSION="${TA_VERSION:-3.6.0}"
IDX="https://triton-ascend.osinfra.cn/pypi/simple"

echo ">>> [1/3] uv sync（拉 triton-ascend 及其依赖）"
uv sync --extra dev

echo ">>> [2/3] 覆盖 libtriton.so（关键一步，让 ascend 版原生库生效）"
uv pip install --no-deps --force-reinstall \
  --index-url "$IDX" "triton-ascend==${TA_VERSION}"

echo ">>> [3/3] 体检"
uv run python env_report.py
