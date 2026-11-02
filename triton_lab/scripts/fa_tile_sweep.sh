#!/usr/bin/env bash
# 逐个 tile 扫 FA kernel，每个 tile 一个独立进程：
# 编译失败/运行时崩溃都不会影响后面的候选。
#
# 用法: bash scripts/fa_tile_sweep.sh [seq] [causal] [dtype] [head_dim]
# 例:   bash scripts/fa_tile_sweep.sh 1000 0 fp16 128
set -u

SEQ="${1:-1000}"
CAUSAL="${2:-0}"
DTYPE="${3:-fp16}"
HEAD_DIM="${4:-128}"

while read -r bq bk; do
  [ -z "${bq:-}" ] && continue
  echo "===== BQ=$bq BK=$bk ====="
  python scripts/fa_tile_probe.py \
    --seq "$SEQ" --causal "$CAUSAL" --dtype "$DTYPE" --head-dim "$HEAD_DIM" \
    --block-q "$bq" --block-k "$bk" --no-check
  rc=$?
  case $rc in
    0) echo ">>> OK $bq $bk" ;;
    3) echo ">>> COMPILE_FAIL $bq $bk" ;;
    *) echo ">>> CRASH $bq $bk (rc=$rc)" ;;
  esac
done <<'TILES'
96 128
64 256
128 256
128 128
64 128
32 256
TILES
