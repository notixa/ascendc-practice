# W3 学习路线：精读 CATLASS FlashAttention 样例

> 本文是 `catlass_lab/README.md` 里 W3 阶段的执行手册，配合 W4（写自己的 FA 骨架）使用。
> 所有行号基于 CATLASS master 当前版本（2026-10 记录），文件改动后行号会有漂移，以类名/函数名为准。

---

## 0. 当前位置

| 项 | 状态 |
| :-- | :-- |
| W1（能读 basic_matmul 链路） | ✅ |
| W2（能独立配一条 element-wise epilogue 链路） | ✅ `matmul_leakyrelu`：编译 0 error + 链接成功 |
| W3（精读官方 FA 样例） | ← **你现在在这里** |
| W4（写自己的 FA 骨架） | 待开始 |
| W5（910B 实机跑通 + 对精度 + 看 perf） | 等硬件 |

### W2 成果存档

| 文件 | 行数 | 说明 |
| :-- | :-- | :-- |
| `catlass_lab/matmul_leakyrelu/matmul_leakyrelu.cpp` | 147 | host + kernel 一体，`MatmulActivation` + `ElemWiseNoSource` |
| `catlass_lab/matmul_leakyrelu/tile_elemwise_leakyrelu.hpp` | 27 | 自己写的 Tile，2 参 `operator()`，`PERCENT` 做负斜率 |

验证方式（不是 clangd，是真编译）：

```bash
# 见本文第 8 节"命令附录"
```

关键配置（背下来，W4 还要复用）：

```cpp
using CType = Gemm::GemmType<float, LayoutC>;   // 累加器：fp32
using DType = Gemm::GemmType<half,  LayoutC>;   // 输出：fp16
using ComputeType = CType;                      // Tile 在 UB 里用 fp32 算
using EpilogueDispatchPolicy = Epilogue::EpilogueAtlasA2ElemWiseNoSource;
// 当前文件写的是 64 * 256；更稳的写法是让它随 tile shape 自动变：
constexpr uint32_t computeLength = (L1TileShape::M / 2) * L1TileShape::N;  // 128/2*256 = 16384
using TileElemWiseEpilogue = Epilogue::Tile::TileElemWiseLeakyRelu<ArchTag, ComputeType, computeLength, 1>;
using EpilogueTileCopy = Epilogue::Tile::TileCopy<ArchTag, CType, DType>;
using BlockEpilogue = Epilogue::Block::BlockEpilogue<
    EpilogueDispatchPolicy, CType, DType, TileElemWiseEpilogue, EpilogueTileCopy>;
using MatmulKernel = Gemm::Kernel::MatmulActivation<BlockMmad, BlockEpilogue, BlockScheduler>;
```

---

## 1. 目标与硬件前提

- **最终目标**：把 FlashAttention 迁移到 CATLASS（fwd 优先，bwd 另议）。
- **当前硬件**：Atlas 200I DK A2（310B / `dav-m200`），**CATLASS 不支持**（只支持 A2/A3=2201、950=3510）。
- **因此**：W1–W4 全部是「读 + 写 + 编译」闭环；数值正确性、性能数据留给 910B。
- **别碰**：`examples/49_*`、`104_*`、`105_*`（950 专用，TLA/RegBase/l0c2ub 会带偏）。

---

## 2. 阅读路线（5 步，每步带自检问题）

目录：`examples/23_flash_attention_infer/`

```
CMakeLists.txt   fai.cpp(319)   fai_kernel.cpp(797)   fai_tiling.cpp(163)
gen_data.py      kernel_common.hpp      README.md / README_en.md
```

### 第 1 步：host 侧 `fai.cpp`（319 行）

**要回答**：这 10 个 buffer 各自多大、谁写谁读、为什么需要中间那块？

```
q  k  v  mask  blockTables  s  p  o  oTemp  oUpdate
```

提示：`s` = QK^T 的结果（fp32），`p` = softmax 后的概率（fp16），`oTemp`/`oUpdate` 是 rescale 的暂存。
**对比点**：普通 matmul 只要 A/B/D 三个 buffer，FA 为什么要这么多？

### 第 2 步：`fai_tiling.cpp`（163 行）

**要回答**：FA 的 tiling 比普通 matmul 多算了什么？

重点抓三个数：

| 参数 | 为什么关键 |
| :-- | :-- |
| `head_dim` | 决定 QK 的 K 维、PV 的 K 维，直接影响 L0 tile 是否整除 |
| block M / N | 决定 S、P、O 三块 UB 的占用 |
| cube : vector 核数配比 | A2 固定 1 AIC : 2 AIV，FA 是否按这个切 |

### 第 3 步：AIC 段 `fai_kernel.cpp:63-385`（322 行）

**要回答**：
- QK 和 PV 两个 Gemm 是**怎么交替**的？
- S（QK 结果 fp32）落在哪、**什么时候**变成 P（fp16）？
- 哪些数据停 L0C、哪些必须落 GM？

这一段是纯 cube 侧，和普通 matmul 最像，先读它建立信心。

### 第 4 步：AIV 段 `fai_kernel.cpp:386-642`（256 行）

**要回答**：online softmax 的 4 个状态各自存在哪（UB？GM？），跨 KV block 怎么递推？

| 状态 | 含义 | 存放位置（你来填） |
| :-- | :-- | :-- |
| `m` | 每行当前最大值 | |
| `l` | 每行指数和（分母） | |
| `O` | 累加输出 | |
| `P` | 本块概率（送给 PV 的输入） | |

### 第 5 步：6 个专有组件（只读 `operator()`，不用读全文）

| 组件 | policy | 实现文件 | 行数 |
| :-- | :-- | :-- | :-- |
| QK 的 Mmad | `MmadAtlasA2FAQK` | `gemm/block/block_mmad_fa_qk.hpp` | 181 |
| PV 的 Mmad | `MmadAtlasA2FAIPV` | `gemm/block/block_mmad_fai_pv_normal.hpp` | — |
| QK 尾块 | `MmadAtlasA2FAITailQK` | `gemm/block/block_mmad_fai_qk_head_tail.hpp` | 298 |
| PV 尾块 | `MmadAtlasA2FAITailPV` | `gemm/block/block_mmad_fai_pv_head_tail.hpp` | — |
| 在线 softmax | `EpilogueAtlasA2OnlineSoftmax` | `epilogue/block/block_epilogue_online_softmax_no_mask.hpp` | **821** |
| O 重缩放 | `EpilogueAtlasA2RescaleO` | `epilogue/block/block_epilogue_rescale_o_no_split_row.hpp` | 274 |

> policy 定义位置：`include/catlass/gemm/dispatch_policy.hpp`、`include/catlass/epilogue/dispatch_policy.hpp`。
> BlockEpilogue 特化位置：`grep -rn "class BlockEpilogue<EpilogueAtlasA2OnlineSoftmax" include/catlass/epilogue/block/`

每个组件读完写**一句话**：它和普通 matmul 的对应组件差在哪。

---

## 3. 专题：为什么需要 "Tail" 变体

同样一个 QK 乘法，为什么要写两份？

```
block_mmad_fa_qk.hpp              181 行
block_mmad_fai_qk_head_tail.hpp   298 行   ← 多 117 行
```

**任务**：自己 diff 出来，回答"这 117 行在干什么"。三条线索：

1. FAQK 的 K 维是 **head_dim**（通常 128），和普通 matmul 的 K（可能几千）不同量级 —— K 循环次数极少，"pingpong 两 stage 预取"的价值完全变了；
2. head_dim 不能被 L0 tile 的 K 整除时，最后一块要特殊处理；
3. 名字里的 `head_tail` 暗示它处理 head 维度的"尾部"。

⚠️ 这个结论对 W4 很关键：**如果你把这套机制当成冗余砍掉，head_dim 不是 128 时会算错。**

---

## 4. W3 交付物

### 4.1 数据流图

一张图，标清：

```
Q, K, V (GM)
   ↓
QK^T → S (fp32, 存哪?) → online softmax → P (fp16, 存哪?)
   ↓
P · V → O 累加 → rescale → 输出 (GM)
```

必须标注：每一步在哪个存储器（GM / UB / L0A / L0B / L0C）、哪几步跨核（AIC ↔ AIV）。

### 4.2 对照表（用你的手写版映射）

你有 `ascendc/flash_attention_npu` 的手写实现，这是"用已知映射未知"的最佳素材。填写下表：

| 我手写版做的事 | 23 号样例里对应的东西 | 备注 |
| :-- | :-- | :-- |
| `QK^T` 的 mmad 循环 | | |
| 行 max 递推 | | |
| `exp(s - m_new)` | | |
| 行 sum 递推 | | |
| `O = O*exp(m_old-m_new) + P·V` | | |
| causal mask 处理 | | |
| 尾块处理（seqlen 非整除） | | |
| 跨核同步点 | | |

---

## 5. W4 预告：写自己的 FA 骨架

**裁剪顺序：先砍功能，再砍性能。**

1. 砍 paged cache（`blockTables` 那条路）→ 用连续 K/V
2. 固定 `head_dim = 128`、单 head、单 batch
3. mask 只留 causal
4. 不做 `oTemp` / `oUpdate` 双缓冲优化（先跑对）

**交付物**：编译通过 + 纸上跑通一遍流程 + 一份"我比 23 号少了什么、为什么可以少"的清单。

### ⚠️ bwd 的现实

仓库里的 backward 样例 `104_ascend950_hstu_backward`、`105_ascend950_fmha_backward` **都是 950 专用**，A2 没有参照。

所以现实路径是：**先把 fwd 做完做好，bwd 后面自建**。建议尽早和老师对齐预期，别等到 W4 才发现没模板可抄。

---

## 6. 两个硬约束

1. **200I 只能编译不能跑**：`npu-smi` 不存在、`/dev/davinci*` 不存在；`msprof op simulator` 会卡在 "Running simulation task"，别耗时间。
2. **clangd 0 error ≠ 能编**：`.clangd` 里有 CPU 仿真头路径（`tools/tikicpulib/lib/include`），clangd 能解析但 bisheng 找不到。**用第 8 节的 bisheng 命令做最终判定。**

---

## 7. 已踩过的坑（W1–W2 沉淀）

| 坑 | 现象 | 解法 |
| :-- | :-- | :-- |
| namespace 当类型 | `Unknown type name 'Epilogue'` | `Catlass::Epilogue` 是 namespace，要用里面的 `EpilogueAtlasA2Xxx` |
| 名字存在但不可见 | `no template named 'X' in namespace 'Y'` | 缺 `#include`，不是名字错 |
| CPU 仿真头 | `fatal error: 'kernel_fp16.h' file not found` | 真机链没有这个头，删掉；`fp16_t` 走 `helper.hpp` |
| 头顺序 | `no member named 'exp' in namespace 'std'` | 别单独 include `golden/matmul.hpp`，用伞形 `golden.hpp` |
| Tile 参数个数 | `no matching function for call to object of type 'TileElemWiseXxx'` | 实参个数由消费它的 BlockEpilogue 决定（NoSource=2，OneSource=3，Gemm=3 含标量） |
| C/D 类型混淆 | 结果全错但能跑 | `CType`=fp32 累加器，`DType`=输出 fp16，host buffer 和 `sizeD` 要跟着 DType |
| 跨核同步没给地址 | 挂死/超时 | AIC+AIV 的 kernel 必须 `aclrtGetHardwareSyncAddr` 并传给 `RunAdapter` |

---

## 8. 命令附录

### 8.1 真编译（最重要）

把任意 `catlass_lab/**.cpp` 当 kernel 文件编一遍，`0 errors` 才算过：

```bash
C=/home/notixa/Ascend/cann-8.5.0
REPO=/home/notixa/codes/opensource-project/catlass
B=$C/bin/bisheng

cd /home/notixa/codes/ascendc/catlass_lab/matmul_leakyrelu && mkdir -p .build

$B -DCATLASS_ARCH=2201 -D_GLIBCXX_USE_CXX11_ABI=1 \
   -I$REPO/examples/common -I$REPO/include \
   -I$C/include/aclnn \
   -Xhost-start -fstack-protector-strong -Xhost-end \
   --npu-arch=dav-2201 \
   -mllvm -cce-aicore-record-overflow=true \
   -mllvm -cce-aicore-dcci-insert-for-scalar=false \
   -Wno-macro-redefined -Wno-ignored-attributes -Wno-unused-value \
   -fPIC -o .build/x.o -c -x asc matmul_leakyrelu.cpp
```

链接（验证有 `main`、符号能解析）：

```bash
$B -Wl,-z,relro,-z,now -Wl,-z,noexecstack -s \
   -Wno-unused-command-line-argument .build/x.o -o .build/x_bin \
   -L$C/lib64 -L$C/x86_64-linux/devlib \
   -lascendc_runtime -lruntime -lprofapi -lmmpa -lascend_dump -lc_sec \
   -lerror_manager -lascendcl -ltiling_api -lnnopbase -lplatform -ldl \
   -lascendalog -lunified_dlog -lascend_hal
```

### 8.2 编辑器自检

```bash
bash catlass_lab/verify_clangd.sh
```

期望：全部 `0 errors`；`fai_kernel.cpp` 允许 2 个（`g_coreType` 在 CPU 仿真桩里不是常量表达式，clangd 与 bisheng 的已知差异）。

### 8.3 改完 `.clangd` 或 `.zed/settings.json`

Zed 里执行 `editor: restart language server`（clangd 只在启动时读配置）。

---

## 9. 参考资料索引

| 内容 | 路径 |
| :-- | :-- |
| FA 样例 | `examples/23_flash_attention_infer/` |
| FA 样例说明 | 同目录 `README.md`（中文，讲数据流和参数） |
| Mmad policy | `include/catlass/gemm/dispatch_policy.hpp` |
| Epilogue policy | `include/catlass/epilogue/dispatch_policy.hpp` |
| 普通 matmul 对照 | `include/catlass/gemm/kernel/basic_matmul.hpp` |
| activation 类 kernel | `include/catlass/gemm/kernel/matmul_activation.hpp` |
| golden 校验 | `examples/common/golden/matmul.hpp`（`ComputeMatmulElemWiseLeakyRelu` 在 :479） |
| 组件开发文档 | `docs/zh/1_Practice/03_kernel_development.md`、`07_epilogue_adaptation.md` |
| 学习总纲 | `catlass_lab/README.md` |
