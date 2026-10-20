#!/usr/bin/env bash
# 校验 clangd 是否能正确解析 CATLASS（输出 0 errors 表示配置生效）
#
# 用法：
#   bash catlass_lab/verify_clangd.sh
#   CATLASS_DIR=/path/to/catlass bash catlass_lab/verify_clangd.sh
#
# 注意：这里用的是 `clangd --check`，它会按每个文件的路径去匹配 .clangd / 用户级配置，
#       和你用 Zed 打开文件时走的是同一套配置解析逻辑。

set -u

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CATLASS_DIR="${CATLASS_DIR:-/home/notixa/codes/opensource-project/catlass}"

FILES=(
  "$LAB_DIR/clangd_probe.cpp"
  "$CATLASS_DIR/examples/00_basic_matmul/basic_matmul.cpp"
  "$CATLASS_DIR/examples/03_matmul_add/matmul_add.cpp"
  "$CATLASS_DIR/examples/27_matmul_gelu/matmul_gelu.cpp"
  "$CATLASS_DIR/examples/23_flash_attention_infer/fai_kernel.cpp"
)

fail=0
echo "CATLASS_DIR = $CATLASS_DIR"
echo "----------------------------------------"
for f in "${FILES[@]}"; do
  if [[ ! -f "$f" ]]; then
    printf '%-28s skip（文件不存在）\n' "$(basename "$f")"
    continue
  fi
  cd "$(dirname "$f")" || exit 1
  out="$(timeout 600 clangd --check="$(basename "$f")" 2>&1)"
  # clangd --check 会把 code action 失败也打成 E[...]，这里只统计真正的诊断
  n="$(printf '%s\n' "$out" | grep -E '^E\[' | grep -v 'tweak:' | wc -l | tr -d ' ')"
  if [[ "$n" == "0" ]]; then
    printf '%-28s OK\n' "$(basename "$f")"
  else
    printf '%-28s %s errors\n' "$(basename "$f")" "$n"
    printf '%s\n' "$out" | grep -E '^E\[' | grep -v 'tweak:' | head -5 | sed 's/^/    /'
    fail=1
  fi
done
echo "----------------------------------------"
if [[ "$fail" == "0" ]]; then
  echo "全部通过 ✅"
else
  echo "存在报错 ❌ —— 多半是路径变了：检查 .clangd 与 ~/.config/clangd/config.yaml 里的绝对路径"
fi
exit "$fail"
