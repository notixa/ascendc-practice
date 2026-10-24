# W4:FA prefill 简化版(无 PagedAttention、无 GQA)

对照 `examples/23_flash_attention_infer/fai_kernel.cpp` 的教学版前向实现。

## 1. 改了什么

| 项 | 原版 | 这里 |
|---|---|---|
| KV 存放 | PagedAttention:`blockTable` 查表 | **连续存放**,`kvOffset = b*S*strideKV + h*D` |
| `getKVOffset` | `blockTableId * blockSize * strideKV + maskTailS * strideKV + ...` | `nowNIdx * KV_SPLIT_SIZE * strideKV`,不读表 |
| head 关系 | GQA/MQA,`groupSize = qHeads/kvHeads` | **`numHeads == kvHeads`**,`groupSize = 1` |
| 任务维度 | `(batch, qSBlock, qNBlock)` 二维展开 | `(batch, qBlk, head)` 三维直接解码 |
| stride | `strideQO != strideKV` | 合并成一个 `stride` |
| 序列长度 | 变长(`actualQseqlen`/`actualKvseqlen` 数组) | 定长 `S` |
| mask 边界 | `noSkipKvS` / `noMaskKvS` / `noMaskTailS` / `maskedKvS` | 只剩 `nDiag = q0/128` |
| Tail 变体必要性 | `maskTailS != 0` 时必须走 head/tail 拆分 | **恒为 0**,逻辑退化 |
| 流水深度 | `preLaunch = 2`,3 槽 ring | **0**,单槽,锁步 |
| 循环结构 | 主循环 + 尾循环,`kvSIdx` 两种步进 | 前缀循环 + 对角块,步进恒定 |

### 为什么 `maskTailS` 会消失

原版的全部对齐麻烦来自 `noMaskTailS = noMaskKvS % 128`。训练/prefill 下 `qSeqlen == kvSeqlen`,
于是 `diffS = 0`、`noSkipKvS = (qBlk+1)*128`、`noMaskKvS = qBlk*128`,**天然 128 对齐**:

```
noMaskTailS = 0        → 不需要从分页块中间起读 → 不需要 head/tail 拆分
maskedKvS   = rowNum   → 对角块宽度就是本 q block 的行数
nDiag       = qBlk     → 对角块号 == 需要跳过的全通块数
```

## 2. 结构

一个任务 =(batch, qBlk, head),对应一个核的一次迭代。

```
q0    = qBlk * 128
rowNum= min(128, S - q0)
nDiag = q0 / 128          // 全通前缀的 KV 块数

① 前缀 [0, nDiag)         每 4 块一次 QK/PV,无需 mask
② 对角块 [nDiag, nDiag+1)  1 块,QK 后 softmax 时加 128x128 上三角 mask
```

两个循环各自发 QK → softmax → PV,靠 `qkReady` / `smReady` / `pvReady` 锁步。
`O` 的累加(`go = go*dm + lo`)完全在 AIV 的 UB 里,只有对角块结束时落盘。

## 3. 约束

- `headDim = 128`(模板 `L1TileShape = <128,128,128>`)
- `qSeqlen == kvSeqlen`,且 `qSeqlen % 128` 可不为 0(最后一个 q block 不满)
- `numHeads == kvHeads`
- 只有 `half`(bf16 没实例化)
- 单 batch 内定长

## 4. 编译 / 运行

```bash
C=/home/notixa/Ascend/cann-8.5.0
REPO=/home/notixa/codes/opensource-project/catlass
B=$C/bin/bisheng

cd catlass_lab/w4_fa && mkdir -p .build

$B -DCATLASS_ARCH=2201 -D_GLIBCXX_USE_CXX11_ABI=1 \
   -I$REPO/examples/common -I$REPO/include -I$C/include/aclnn \
   -Xhost-start -fstack-protector-strong -Xhost-end \
   --npu-arch=dav-2201 \
   -mllvm -cce-aicore-record-overflow=true \
   -mllvm -cce-aicore-dcci-insert-for-scalar=false \
   -Wno-macro-redefined -Wno-ignored-attributes -Wno-unused-value \
   -fPIC -o .build/fa.o -c -x asc fa_prefill.cpp

$B -Wl,-z,relro,-z,now -Wl,-z,noexecstack -s \
   -Wno-unused-command-line-argument .build/fa.o -o .build/fa_bin \
   -L$C/lib64 -L$C/x86_64-linux/devlib \
   -lascendc_runtime -lruntime -lprofapi -lmmpa -lascend_dump -lc_sec \
   -lerror_manager -lascendcl -ltiling_api -lnnopbase -lplatform -ldl \
   -lascendalog -lunified_dlog -lascend_hal -lm
```

注意 **`-lm`**:host 侧的 golden 用了 `expf/sqrt`,原版 `matmul_leakyrelu` 没用到数学函数所以没这一项。

运行(需要 A2 硬件):

```bash
./build/fa_bin 1 256 2      # batch=1, seq=256, heads=2
```

## 5. 验证状态

| 项 | 状态 |
|---|---|
| bisheng 编过(dav-2201) | ✅ 0 error |
| 链接(`main` + 符号) | ✅ 315KB |
| clangd 解析 | 未单独跑,可直接用 `catlass_lab/verify_clangd.sh` 加进来 |
| **数值正确性** | ❌ **未验证**(手上没有 A2) |

上机后第一批要查的地方(按可能性排序):

1. **`preloadFlag`** —— QK/PV 第一次调用传 1,后续传 0。若第一个块算出来是 0,先查这里
2. **`LayoutP` 的 stride** —— 全程用 `TILE_N = 512`,对角块只有 `rowNum` 列但 stride 仍是 512
3. **对角块的 mask 行索引** —— epilogue 用 `proTokenIdx = rowOffsetCurLoop % tokenNumPerHead`,
   只有在"一个 sub-core 一次 row loop 吃完"时才对(本实现 `rowNum <= 128` 满足)
4. **`nLoop` 的语义** —— mmad 里是"总块数"而非"剩余块数",对角块因此传 `nDiag + 1`
5. **`qSBlockSize` 参数** —— 传的是 `rowNum`(最后一块可能是 49),不是常量 128

## 6. 想加回性能时按这个顺序

1. `preLaunch = 2` + 3 槽 ring(S/P/oTemp 各 3 份,`slot = stackSeqCount % 3`)
2. 拆主/尾循环,让 QK 领先 PV 两个 stack tile
3. 4 块叠加已经是默认行为(由 `UNIT_BLOCK_STACK_NUM = 4` 决定)
4. 最后才考虑 paged / GQA
