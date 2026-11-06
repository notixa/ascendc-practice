#!/usr/bin/env bash
# FlashDecoding tiling sweep：逐个 (BLOCK_S, SPLIT_S) 起独立进程跑 fd_bench.py。
#
# 用法:
#   bash scripts/fd_sweep.sh                          # preset=quick, 默认 BLOCKS/SPLITS
#   bash scripts/fd_sweep.sh default                  # 跑 default 的 7 个 case
#   bash scripts/fd_sweep.sh long --baseline none     # long + 不测 eager baseline
#   BLOCKS="64 128 256" SPLITS="256 512 1024" \
#       bash scripts/fd_sweep.sh quick
#
# 每个配置独立进程：编译失败/507015 不会影响后续配置。
set -u

cd "$(dirname "$0")/.." || exit 1

PRESET="${1:-quick}"
shift || true

# 默认不扫 256：BLOCK_S=256 在 seq_len 非整除时有 507015 风险。
# 需要时显式 BLOCKS="64 128 256"。
BLOCKS="${BLOCKS:-64 128}"
SPLITS="${SPLITS:-128 256 384 512 768 1024 2048}"

ok=0
fail=0
skip=0
total=0

for bs in $BLOCKS; do
  for ss in $SPLITS; do
    total=$((total + 1))
    if [ "$ss" -lt "$bs" ] || [ $((ss % bs)) -ne 0 ]; then
      echo "===== SKIP block_s=$bs split_s=$ss (需要 split_s >= block_s 且整除) ====="
      skip=$((skip + 1))
      continue
    fi

    echo "===== block_s=$bs split_s=$ss preset=$PRESET ====="
    python scripts/fd_bench.py \
      --preset "$PRESET" \
      --split-s "$ss" \
      --block-s "$bs" \
      "$@"
    rc=$?
    if [ "$rc" -eq 0 ]; then
      ok=$((ok + 1))
    else
      fail=$((fail + 1))
      echo ">>> FAIL block_s=$bs split_s=$ss (rc=$rc)"
    fi
  done
done

echo "============================================================"
echo "sweep summary: total=$total ok=$ok fail=$fail skip=$skip"
[ "$fail" -eq 0 ]
