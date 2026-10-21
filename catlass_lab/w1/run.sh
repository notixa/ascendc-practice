#!/usr/bin/env bash
# =============================================================================
# W1 受控实验 runner（无 NPU 版）
#
# 用法：
#   bash catlass_lab/w1/run.sh 1     # L1TileShape 超限
#   bash catlass_lab/w1/run.sh 2     # L0TileShape 超限
#   bash catlass_lab/w1/run.sh 3     # 只换 policy（反例，必然失败）
#   bash catlass_lab/w1/run.sh 3b    # policy + kernel 一起换（能编过）
#   bash catlass_lab/w1/run.sh 4     # LayoutB 换 ColumnMajor（能编过）
#   bash catlass_lab/w1/run.sh 5     # LayoutC 换 ColumnMajor（必报错，用来认识模板边界）
#   KEEP=1 bash catlass_lab/w1/run.sh 3   # 改完保留现场（方便在 Zed 里看类型），手动还原见末尾提示
#
# 做三件事：改一处 → 只编这一个 target → 抓静态断言 → 还原
# 依赖：CATLASS 仓可写（默认 ~/codes/opensource-project/catlass），环境变量同正常构建
# =============================================================================

set -u

CATLASS_DIR="${CATLASS_DIR:-$HOME/codes/opensource-project/catlass}"
SRC="$CATLASS_DIR/examples/00_basic_matmul/basic_matmul.cpp"
BAK="$(dirname "$0")/.basic_matmul.cpp.bak"
LOG="$(dirname "$0")/build_exp.log"

if [[ ! -f "$SRC" ]]; then
    echo "找不到样例源码：$SRC"
    echo "请用 CATLASS_DIR=/your/catlass bash $0 $*"
    exit 1
fi

# 每次开始都从仓库原始版本出发，避免上一次的改动叠加
git -C "$CATLASS_DIR" checkout -- examples/00_basic_matmul/basic_matmul.cpp 2>/dev/null || true
cp "$SRC" "$BAK"

EXP="${1:-}"
case "$EXP" in
1)
    sed -i 's|using L1TileShape = GemmShape<128, 256, 256>;|using L1TileShape = GemmShape<128, 256, 512>;|' "$SRC"
    echo "[exp1] L1TileShape 128x256x256 -> 128x256x512"
    echo "       手算：(128*512*2 + 256*512*2)*2 = 786432 B > L1 512KB  → 预期编译失败"
    ;;
2)
    sed -i 's|using L0TileShape = GemmShape<128, 256, 64>;|using L0TileShape = GemmShape<128, 256, 128>;|' "$SRC"
    echo "[exp2] L0TileShape 128x256x64 -> 128x256x128"
    echo "       手算：L0B = 128*256*2*2 = 131072 B > L0B 64KB  → 预期编译失败（报 L0B）"
    ;;
2b)
    sed -i 's|using L1TileShape = GemmShape<128, 256, 256>;|using L1TileShape = GemmShape<256, 256, 64>;|' "$SRC"
    sed -i 's|using L0TileShape = GemmShape<128, 256, 64>;|using L0TileShape = GemmShape<256, 256, 64>;|' "$SRC"
    echo "[exp2b] L1=L0=256x256x64"
    echo "        手算：L0C = 256*256*4 = 262144 B > L0C 128KB  → 预期编译失败（报 L0C）"
    ;;
3)
    # 反例：只换 policy，不换 kernel —— 必然失败，这是"契约耦合"的最好证据
    sed -i 's|using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<true>;|using DispatchPolicy = Gemm::MmadAtlasA2Preload<true>;|' "$SRC"
    echo "[exp3] 只换 DispatchPolicy：Pingpong<true> -> Preload<true>（不动 kernel）"
    echo "       预期：编译失败，no matching function for call to BlockMmad（参数个数不匹配）"
    ;;
3b)
    # 正例：policy + kernel 一起换（Preload 版 operator() 多要 next block 的地址/形状）
    sed -i 's|using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<true>;|using DispatchPolicy = Gemm::MmadAtlasA2Preload<true>;|' "$SRC"
    sed -i 's|using MatmulKernel = Gemm::Kernel::BasicMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;|using MatmulKernel = Gemm::Kernel::BasicMatmulPreload<BlockMmad, BlockEpilogue, BlockScheduler>;|' "$SRC"
    sed -i '/#include "catlass\/gemm\/kernel\/basic_matmul.hpp"/a #include "catlass/gemm/kernel/basic_matmul_preload.hpp"' "$SRC"
    echo "[exp3b] policy + kernel 一起换：Preload<true> + BasicMatmulPreload"
    echo "        参照 examples/21_basic_matmul_preload_zN；预期：能编过"
    ;;
4)
    sed -i 's|using LayoutB = layout::RowMajor;|using LayoutB = layout::ColumnMajor;|' "$SRC"
    echo "[exp4] LayoutB RowMajor -> ColumnMajor"
    echo "       预期：能编过；搬运路径换成 ColumnMajor 的 TileCopy 特化"
    ;;
5)
    sed -i 's|using LayoutC = layout::RowMajor;|using LayoutC = layout::ColumnMajor;|' "$SRC"
    echo "[exp5] LayoutC RowMajor -> ColumnMajor"
    echo "       预期：编译失败 —— \"LayoutC only support RowMajor yet!\"（模板库的边界）"
    ;;
*)
    echo "用法：bash $0 <1|2|2b|3|3b|4|5>"
    exit 1
    ;;
esac

echo "----------------------------------------------------------------"
echo ">>> 构建中（首次会重新 configure，约 1~2 分钟；之后增量 10~20s）"
(
    cd "$CATLASS_DIR" || exit 1
    bash scripts/build.sh 00_basic_matmul
) >"$LOG" 2>&1
rc=$?
echo "    build exit code = $rc  ($LOG)"

echo "----------------------------------------------------------------"
if [[ $rc -eq 0 ]]; then
    echo "结果：编译通过 ✅"
    echo "  → 若本实验预期\"失败\"，说明你的手算有误，回去核对 L1/L0 的字节数。"
else
    echo "结果：编译失败（对 exp1/2/2b/3/5 这正是预期；exp3b/4 若失败则需排查）"
    echo "  关键报错："
    grep -nE "static assertion failed|exceeding the|only support RowMajor|error:" "$LOG" \
        | grep -v "unused" | head -12 | sed 's/^/    /'
    echo
    echo "  提示：完整模板实例化栈很长，先搜 \"static assertion failed\"，再往上看第一行 required from。"
fi

if [[ "${KEEP:-0}" == "1" ]]; then
    echo "----------------------------------------------------------------"
    echo "已保留改动（KEEP=1）。还原："
    echo "  git -C $CATLASS_DIR checkout -- examples/00_basic_matmul/basic_matmul.cpp"
else
    cp "$BAK" "$SRC"
    echo "----------------------------------------------------------------"
    echo "已还原 $SRC"
fi
