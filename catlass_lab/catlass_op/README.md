# catlass_op —— FATkernel 的 PyTorch 接入

把 `catlass_lab/flash_attention_train/` 里的 `FATKernel` 包成 `torch.ops.catlass.fat`。

参考仓库里的官方脚手架：
`catlass/.agents/skills/catlass-example-to-torch-intf/`（SKILL.md + templates/）

---

## 结构

```
catlass_lab/catlass_op/
├── CMakeLists.txt
├── include/
│   └── catlass_kernel.h        # device wrapper 的声明（gcc 可编，无 __gm__）
├── kernel/
│   └── catlass_kernel.asc      # type alias + __global__ 入口 + host wrapper（bisheng 编）
├── src/
│   └── catlass_torch.cpp       # torch 胶水：校验 → tiling → workspace → launch
└── test_fat.py                 # torch.ops.catlass.fat + CPU golden
```

kernel 本体不复制过来，`.asc` 通过 `-I../flash_attention_train` 直接 `#include "fat_types.h"` ——
**保持单一真相**，改 kernel 不用同步两份。

### `flash_attention_train/` 现在只剩三个文件

| 文件 | 作用 | 谁 include |
|---|---|---|
| `fat_kernel.h` | `FATKernel` 类模板（AIC/AIV 的 `operator()`） | `fat_types.h` |
| `fat_tiling.h` | `FATParams` / `FATTiling`（POD） | `fat_kernel.h`、`catlass_torch.cpp` |
| `fat_types.h` | **编译期契约**：dtype / tile / 4 个 Block / `FatKernelType` | `.asc`、未来的 standalone |

`fat.cpp` 已删除（它同时含 kernel 别名和 host 骨架，和 `.asc` 重复且编不过）。
"不依赖 torch 的独立验证"如果以后需要，另写 `standalone/fat_main.cpp`，`#include "fat_types.h"` 即可。

---

## 已完成的两件事

1. ~~删掉 `fat_kernel.h` 末尾的 `FATEntry`~~ ✅（那里写了 `FATKernel kernel;`，类模板不能 CTAD）
2. ~~抽类型别名到 `fat_types.h`~~ ✅（`.asc` 不再写任何别名）

入口函数只存在于一处：`kernel/catlass_kernel.asc` 的 `fat_impl`。

---

## 构建

```bash
# 需要：CANN 环境 + torch + torch_npu（当前这台机器上没装 torch，所以这里编不了）
source /home/notixa/Ascend/cann-8.5.0/set_env.sh

cd catlass_lab/catlass_op
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

产物：`build/libcatlass_fat.so`

CMake 的两个可调项：

| 变量 | 默认 | 说明 |
|---|---|---|
| `CATLASS_INCLUDE_DIR` | `/home/notixa/codes/opensource-project/catlass/include` | CATLASS 本体 |
| `FA_KERNEL_DIR` | `../flash_attention_train` | `fat_kernel.h` / `fat_tiling.h` 所在 |
| `NPU_ARCH` | `2201` | 910B |

---

## 运行

```bash
cd catlass_lab/catlass_op
python3 test_fat.py
```

Python 接口：

```python
import torch, torch_npu
torch.ops.load_library("build/libcatlass_fat.so")

o = torch.ops.catlass.fat(q, k, v, mask, scale, batch, block_dim)

# q/k/v 支持两种布局（内存排布都是 token-major，返回形状与 q 一致）：
#   3-D: [batch*N_q, heads, head_dim]      → batch 必须显式传（>=1）
#   4-D: [batch, N_q, heads, head_dim]     → batch 可传 0，从 q.size(0) 推出
# 4-D 在 host 侧 reshape 成 3-D，连续张量上零拷贝。
#
# mask  : [>=128, >=128] fp16, contiguous, 1=屏蔽（triu(diagonal=1)）
# scale : float，通常 1/sqrt(head_dim)
# block_dim : int，0 = 自动（min(numTask, AIC 核数)）
```

**约束**（当前 kernel 的假设）：

- `heads` 必须是 `q`/`k`/`v` 相同（无 GQA）
- `head_dim = 128`
- `N_q == N_k`（causal、无 chunk）—— `test_fat.py` 的 case 都满足
- 3-D 时 `batch` 必填；4-D 时可以传 0 自动推断
- 只有 forward，没有 autograd
- **必须传 mask**（causal）：`[>=128, >=128]` 的 fp16 上三角表，`1` = 屏蔽，`diagonal=1`（自己看自己保留）
- mask 的坐标是"相对对角块起点"，所以一张 128x128 表服务所有 q block；列起点恒为 0，表不能被裁剪

---

## 与官方模板的 4 处偏离

| # | 模板假设 | 这里 |
|---|---|---|
| 1 | `Kernel::Arguments` + `ToUnderlyingArguments` | `FATKernel` 没有 → 直接构造 `FATParams`（9 个指针） |
| 2 | 一块 workspace（`GetWorkspaceSize`） | **三块**：`s`(fp32) / `p`(fp16) / `oTemp`(fp32)，`at::empty` 各一次 |
| 3 | 题目信息走 `Arguments` | `FATTiling` 是运行时 POD → `from_blob` + `.to(device)` |
| 4 | 不需要 hw sync addr（Rule 4） | **待验证**：FA 是 AIC+AIV，用了 `CrossCoreSetFlag` |

第 4 条是唯一的不确定项。如果一个都不同步、AIV 一直等不到 `qkReady`（表现为 hang 或 O 全 0），
就在 `src/catlass_torch.cpp` 里加：

```cpp
uint64_t hwSync = 0;
aclrtGetHardwareSyncAddr(reinterpret_cast<void**>(&hwSync));
// 传给 wrapper → __global__ 入口里 AscendC::SetSyncBaseAddr(hwSync);
```

（`fai.cpp` 就是这么做的；纯 AIC 的 matmul 不需要，所以模板说不要。）

---

## workspace 公式（三处必须一致）

```
s     : coreNum * B_q * B_k      float
p     : coreNum * B_q * B_k      half
oTemp : coreNum * B_q * headDim  float
```

对应：

1. `src/catlass_torch.cpp` 的三个 `at::empty`
2. `kernel/catlass_kernel.asc::get_fat_workspace_size`
3. kernel 里 `layoutS{B_q,B_k}` / `layoutP{B_q,B_k}` / `layoutOTmp{B_q,headDim}` 的 stride

**改任何一个都要同步另外两个。**

---

## 验证阶梯

`test_fat.py` 里的 case 顺序就是验证顺序：

| case | 验证什么 |
|---|---|
| `(1,128,128,1,128)` | 单任务、单 head、只有一个 KV 块（对角块） |
| `(1,128,128,2,128)` | 多 head（`DecodeTask` 的 h 解码 + oOffset） |
| `(1,256,256,1,128)` | 两个 q block（前缀块 + 对角块，`kRowNum` 的边界） |
| `(1,256,256,2,128)` | 组合 |
| `(2,256,256,2,128)` | 多 batch（`b` 偏移） |

任一 case 挂了，先看它比上一个多了什么维度 —— 那就是 bug 的位置。

---

## 已验证 / 未验证

| 项 | 状态 |
|---|---|
| `.asc`（kernel 侧）编译 | ✅ 76 KB `.o`，0 error（用去掉 `FATEntry` 的副本验证的） |
| `src/catlass_torch.cpp` 编译 | ❌ 本机没装 torch/torch_npu，编不了 |
| CMake 配置 | ❌ 同上 |
| 数值正确性 | ❌ 需要 910B |
