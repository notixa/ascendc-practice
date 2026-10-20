# CATLASS 学习路径（贴着现有 AscendC 经验定制）

> 你的目标：**把 `flash_attention_npu` 迁到 CATLASS**；现状：只有 200I（310B，CATLASS 不支持），**之后能拿到 910B**。
> 所以路线按"先吃透、先写出来，等机器到了再跑"来排：现在这段时间不是空等，是难得的"只跟编译器打交道"的窗口期。
> 本文所有路径/行号均在本机实测核对过（2026-10-20）。

---

## 0. 结论先行

1. **硬件边界先划清楚。** CATLASS C++ 只支持 `Atlas A2/A3`（`CATLASS_ARCH=2201`）和 `Ascend 950`（`3510`）；
   `include/catlass/arch/arch.hpp` 里只有 `AtlasA2`、`Ascend950` 两个 ArchTag。
   你手上的 **200I DK A2（310B / dav-m200）不在支持列表**：**能编译，跑不了**。
   你之后能拿到 910B，所以现在这段（200I 阶段）的定位是：**读码 + 装配 + 编译通过**，把"跑不跑得起来"这件事推迟到 910B 到手，别在仿真器上耗时间。
2. **别碰 CATLASS DSL（`python/tla_dsl`）。** 它只支持 Ascend 950，且要求 `CANN >= 9.1`；你本机是 CANN 8.5.0。C++ 模板这条路才是老师说的 CATLASS。
3. **学习顺序要拿你写过的东西当锚点**：
   `matmul_leakyrelu_npu`（AscendC Matmul API + UB 上 LeakyRelu）→ `examples/03_matmul_add`、`27_matmul_gelu`；
   `flash_attention_npu`（手写 TQue/SetFlag 的 QK、PV、online softmax）→ `examples/23_flash_attention_infer`。
   这两条线正好是 CATLASS 最有代表性的两个用法：**Epilogue 融合** 和 **多 BlockMmad 拼装**。

---

## 1. 环境现状（我已核实）

| 项 | 现状 |
| :-- | :-- |
| 源码 | `/home/notixa/codes/opensource-project/catlass`（master，含 v2.1.0 内容，75MB） |
| 已有构建目录 | `/home/notixa/catlass-build`（`CATLASS_ARCH=2201`，18:52 今天成功编出 `examples/00_basic_matmul/00_basic_matmul`） |
| CANN | 8.5.0（`/home/notixa/Ascend/cann-8.5.0`，CATLASS C++ 要求 ≥8.5.0 ✅） |
| 编译器 | 系统 gcc 16.2.1（**超出 CATLASS 要求 `<13.0`**），但 `g++-15` 在，且当前 shell 的 `CPLUS_INCLUDE_PATH` 已指向 gcc-15 头文件 → 能编过 |
| Python | 3.14（仓库脚本建议 `<3.12`，`gen_data.py` 之类可能踩坑，C++ 主流程不受影响） |
| 运行 | **本机无 NPU**（无 `npu-smi`、无 `/dev/davinci*`） |
| 仿真 | `msprof op simulator` 库在（`tools/simulator/Ascend910B4/lib`），实测卡在 `Running simulation task`，暂不可用（沙箱只允许写工作区，plog 落盘被拒可能是原因之一） |

**能做什么**：改代码 + 编译（CATLASS 的编译期反馈极强，静态断言就是你的"单测"）+ 读源码。
**不能做什么**：跑精度对比、看 profiling。这两件事必须等 A2/A3 机器。

---

## 1.5 你的情况 → 5 周计划（目标：FA 迁移）

只有 200I 的阶段（第 1~4 周）**验收标准全靠"编译过 + 讲得清"**，不要试图跑；910B 到手后（第 5 周）才进入"跑数、对精度、看 perf"。

| 周 | 主题 | 具体动作 | 验收（200I 上就能做） |
| :-: | :-- | :-- | :-- |
| **W1** | 会用模板"装"一个 Matmul | 读 `02_host_example_assembly.md`；编译 `00_basic_matmul`；做 S1 的 4 个受控实验 | 从零（空文件）默写出 2.3 那 8 行装配并编译通过；能解释 4 条静态断言各是哪个 buffer 超了 |
| **W2** | Epilogue 流水（FA 的地基） | 读 `03_matmul_add` → `27_matmul_gelu` → `07_epilogue_adaptation.md`；把 `matmul_leakyrelu_npu` 用 CATLASS 重写 | 编译通过；能画出 `L0C →(FixPipe)→ UB →(Vector)→ GM` 的流水，并说明 online softmax 为什么天然适合做成 epilogue |
| **W3** | 精读官方 FA 样例 | 读 `examples/23_flash_attention_infer/`（`fai_kernel.cpp`、`fai_tiling.cpp`、`gen_data.py`） | 交一张**数据流图 + 对照表**：你的 `flash_attention_fwd_kernel.cpp` 每段 ↔ 23 号样例的哪个类；标注 `MmadAtlasA2FAQK/FAPV`、`EpilogueOnlineSoftmax/EpilogueRescaleO`、Tail 变体各自解决什么 |
| **W4** | 写出"我自己的 FA 骨架" | 照 23 的结构搭一个精简版（先砍掉 paged cache / 先固定 head_dim、mask 只留 causal），host 侧 + tiling 一起写 | 编译通过、能在纸上跑通一遍流程；列出"我比 23 号少了什么、为什么可以少" |
| **W5**（910B） | 跑通 + 对精度 + 看 perf | 先跑 23 号样例 `Compare success`，再跑自己的；用 `msprof` 对比你手写 FA 与 CATLASS 版 | 精度对齐 + 一份性能对比（含 bubble/流水对比） |

**一个必须先跟老师说清的现实**：
- `examples/23_flash_attention_infer` 是**推理 FA**（paged KV cache、causal mask、`qSeqlen ≤ kvSeqlen`），和你的训练式 fwd（带 softmax stats、写回 O）结构接近但不完全一样 → **fwd 有参照，能迁**。
- **bwd 没有 A2 参照**：仓库里的 backward 样例（`104_ascend950_hstu_backward`、`105_ascend950_fmha_backward`）都是 **950 专用**。所以"FA 迁移"这个目标，现实的做法是 **先 fwd，bwd 后面自建**（或只做 fwd 的工程化）。这一点建议提前和老师对齐预期，别等到第 4 周才发现 bwd 没模板可抄。

---

## 2. 心智模型：CATLASS 到底在解决什么问题

### 2.1 一句话

> AscendC 的 `Matmul` API 是**黑盒**（你填 tiling，它内部自己切 L1/L0、自己 ping-pong）；
> CATLASS 把这个黑盒**拆成白盒零件**：调度策略、分块大小、搬运、Cube 计算、后处理，全部由你用模板参数显式装配。
> 它不替你思考，它替你把"已经想好的流水"写干净。

### 2.2 三套范式对照（你的代码 → CATLASS）

| 语义 | 你写的（`matmul_leakyrelu_npu/csrc`） | CATLASS 里是谁 |
| :-- | :-- | :-- |
| 选硬件 | `__CCE_AICORE__` / soc 编译宏 | `Arch::AtlasA2`（`arch.hpp:18`，带 UB=192K、L1=512K、L0A=64K、L0C=128K 常量） |
| 输入输出描述 | `MatmulType<TPosition::GM, CubeFormat::ND, half, false>` | `Gemm::GemmType<Element, Layout>`（`gemm/gemm_type.hpp`）+ `layout::RowMajor/ColumnMajor` |
| 分块粒度 | `MatmulLeakyReluTilingData`（host 算 tiling） | `GemmShape<M,N,K>` 两级：`L1TileShape` / `L0TileShape`（**编译期常量**） |
| 流水策略 | `Matmul` 引擎内部默认 ping-pong | `Gemm::MmadAtlasA2Pingpong<true>`、`MmadAtlasA2Preload<>`、`MmadAtlasA2SingleCoreSplitk<L1A,L1B,L0C>`（`gemm/dispatch_policy.hpp`） |
| 核间任务分配 | `GetBlockIdx()` 手算 m/n 块 | `GemmIdentityBlockSwizzle<3,0>`（BlockScheduler） |
| Cube 计算 + 搬运 | `mm_.Iterate()` | `Gemm::Block::BlockMmad<...>`（`gemm/block/block_mmad.hpp`） |
| cube 结果 → UB → LeakyRelu | `C_TYPE = VECIN` + `TQue` + `LeakyRelu()` | `BlockEpilogue`（`epilogue/`，例：`27_matmul_gelu`） |
| 启动 kernel | `<<<blockDim, nullptr, stream>>>` | `Gemm::Device::DeviceGemm<Kernel>`（管 workspace + 下发） |

**你最该带走的一句**：`matmul_leakyrelu` 里"`C_TYPE=VECIN` 把结果从 L0C 搬到 UB 再做向量"这件事，在 CATLASS 里叫 **BlockEpilogue**，而且它是可插拔的独立流水线 —— 这正是你要从"会用 API"升级到"会搭流水"的台阶。

### 2.3 装配链（`examples/00_basic_matmul/basic_matmul.cpp:95-113` 真实代码）

```cpp
using ArchTag        = Arch::AtlasA2;
using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<true>;
using L1TileShape    = GemmShape<128, 256, 256>;
using L0TileShape    = GemmShape<128, 256, 64>;
using AType = Gemm::GemmType<half, layout::RowMajor>;
...
using BlockMmad      = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
using BlockEpilogue  = void;                                  // 不融合后处理
using BlockScheduler = Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
using MatmulKernel   = Gemm::Kernel::BasicMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;
using MatmulAdapter  = Gemm::Device::DeviceGemm<MatmulKernel>;
```

注意两件事（也是你旧路线图里写错的地方）：
- `BasicMatmul` 的模板参数是 `<BlockMmad, BlockEpilogue, BlockScheduler>`，**没有 ArchTag**（ArchTag 从 BlockMmad 里 typedef 出来）。
- 分块是**编译期**的：`L1TileShape` 写错不会运行报错，而是编译期 `static_assert` 直接拦。

---

## 3. 分层地图（照着目录读，不要从 `include/` 乱翻）

```
include/catlass/
├── arch/          ArchTag + 片上容量常量 + LocalTensorBuffer（谁在开 L1/L0/UB 的 buffer）
├── gemm/
│   ├── kernel/    算子级组装（basic_matmul.hpp 就是 C = A*B 的完整骨架）
│   ├── block/     BlockMmad（Cube 流水）、BlockScheduler/Swizzle（核间任务）
│   ├── device/    DeviceGemm（workspace、下发、aclrt 交互）
│   ├── tile/      最小的搬运/计算积木（tile_copy / tile_mmad / ascend950 特化）
│   ├── dispatch_policy.hpp   ★ 所有"策略"都在这里，先读这个文件
│   └── gemm_type.hpp
├── epilogue/      ★ 后处理流水线（你写 LeakyReLU/GELU 融合的地方）
└── layout/        RowMajor / ColumnMajor / 分形相关
```

配套文档（**按这个顺序读，别跳**）：
1. `docs/zh/2_Design/00_project_overview.md` —— 先用 10 分钟建立全图
2. `docs/zh/1_Practice/02_host_example_assembly.md` —— 怎么把上面那 8 行 using 填出来
3. `docs/zh/1_Practice/03_kernel_development.md` → `04_block_mmad_development.md` → `06_tile_development.md`
4. `docs/zh/1_Practice/07_epilogue_adaptation.md` —— 写自定义融合时再读
5. `docs/zh/2_Design/01_kernel_design/04_matmul_summary.md` —— 调优阶段读（性能模型/优化点清单）

---

## 4. 分阶段路线（每阶段都有"验收"，别只看不写）

### S0 建立地图（半天）
- **读**：`docs/zh/2_Design/00_project_overview.md` + `docs/zh/2_Design/01_kernel_design/03_dispatch_policies.md`（110 行，讲清每种策略的适用场景）+ 打开 `include/catlass/gemm/dispatch_policy.hpp` 搜索 `struct Mmad` 把十几个 policy 的 `STAGES` 扫一遍
- **写**：纸上默画 2.3 的装配链，标出每个模板参数"决定了什么硬件行为"
- **验收**：能不看代码说出：`L1TileShape` 变了会影响谁、`STAGES=2` 是什么意思、`BlockEpilogue=void` 代表什么

### S1 让"编译期"变成你的老师（1 天）
在 `00_basic_matmul` 上做 4 个受控实验（只改参数，不改结构），记录报错原文：
1. `L1TileShape` 128×256×256 → 256×256×256：**A2 的 L1 只有 512KB**，看静态断言怎么骂你
2. `L0TileShape` 128×256×64 → 128×256×128：L0C 128KB 超限
3. `DispatchPolicy` 换 `MmadAtlasA2Preload<>`：能编过说明策略可替换；对比两者的 `STAGES`/`ENABLE_SHUFFLE_K`
4. `LayoutB` 换 `ColumnMajor`：理解 B 矩阵的 K/N 谁连续、对搬运 tile 的影响
- **验收**：4 条报错你都能解释"是哪块内存、哪个维度超了"，并写进第 6 节的坑记录

### S2 第一个自己写的 Epilogue（2~3 天，**FA 的地基，必做**）
目标：把你已经写过的 `Matmul + Bias + LeakyReLU` 用 CATLASS 重写一遍 —— 练的是"L0C 出来的结果怎么在片上被 Vector 处理"，
这正是 FA 里 online softmax 的形态（只不过 softmax 还要跨 tile 维护 running max/sum）。
- **读**：`examples/03_matmul_add`（最简 epilogue）、`examples/27_matmul_gelu`（vector 融合）、`epilogue/` 下的 `block_epilogue_*`
- **改**：照抄 03 的结构，把 Add 换成 LeakyReLU（`negative_slope` 当 kernel 参数传进去）
- **验收**：
  - 编译通过（910B 到手后补 `Compare success`）
  - 对比代码量：你的 `matmul_leakyrelu_kernel.cpp` vs CATLASS 版（后者应该只剩"装配 + 一个 epilogue 类"）
  - 能说清 bias 是从 GM 读进 UB 还是在 epilogue 里 broadcast

### S3 调度与并行策略（2 天）
- **读**：`examples/06_optimized_matmul`、`09_splitk_matmul`、`34_single_core_splitk_matmul`、`05_block_scheduler_development.md`、`docs/zh/2_Design/01_kernel_design/02_swizzle.md`
- **重点**：`GemmIdentityBlockSwizzle<offset, direction>` 改的是**核间遍历顺序**（DRAM bank 冲突 / L2 命中）；SplitK 是"核数填不满"时的补救
- **验收**：能画出 `M=4096,N=4096` 时 20 个核的取块顺序，并解释为什么 offset=3 能减少同时读同一块 DRAM

### S4 目标：把你的 FA 搬到 CATLASS（1 周，这是给老师看的产出）
- **读**：`examples/23_flash_attention_infer/`
  - `fai_kernel.cpp:26` 的 `FAInferKernel<BlockMmadQK, BlockMmadPV, BlockMmadQKTail, BlockMmadPVTail, EpilogueOnlineSoftmax, EpilogueRescaleO, PAGED_CACHE_FLAG>`
  - 关键点：**FA 在 CATLASS 里 = 两个 BlockMmad + 两个自定义 Epilogue**，QK 用 `MmadAtlasA2FAQK`、PV 用 `MmadAtlasA2FAPV`（`dispatch_policy.hpp:73-86`），Tail 变体处理非对齐尾块
  - `L1TileShape = GemmShape<128,128,128>`，`L0TileShape = L1TileShape`
- **对照你自己的 `flash_attention_fwd_kernel.cpp`**：QK matmul / softmax / PV matmul 各对应哪段，你的 `TQue`+`SetFlag` 被哪个类吃掉了
- **产出**：先跑通 23 的 `gen_data.py` + 精度对比；再把你 fwd 的 mask / scale / 尾块逻辑换成 CATLASS 组件
- **验收**：同一 shape 下你的实现与 23 号样例结果对齐；能讲清"CATLASS 版比我手写版少了哪些同步、多了哪些约束"

### S5 工程化（有余力再碰）
- `examples/advanced/basic_matmul_aclnn` —— op_host/op_kernel 两段式 + aclnn 接口（工业交付形态）
- `examples/102_dynamic_optimized_matmul` —— Matmul 泛化工程（一套代码覆盖多种 shape/模板）
- `examples/python_extension` —— pybind/torch 扩展，**跟你现在 `torch.utils.cpp_extension` 的用法能无缝接上**
- `tests/optest/` —— 官方算子级测试框架，比手写 main.cpp 更省事

---

## 5. 日常开发循环

```bash
# 0) 一次性：环境变量（当前 shell 已配好，新 shell 记得 source）
source /home/notixa/Ascend/cann-8.5.0/set_env.sh

# 1) 编译单个样例（默认 CATLASS_ARCH=2201）
cd /home/notixa/codes/opensource-project/catlass
bash scripts/build.sh 00_basic_matmul
# 已有构建目录可直接增量编（更快）：
# cmake --build /home/notixa/catlass-build --target 00_basic_matmul -j

# 2) 运行（需 A2/A3 真机）
cd output/bin && ./00_basic_matmul 256 512 1024 0    # 期望输出 Compare success.

# 3) 调试手段
bash scripts/build.sh --debug     00_basic_matmul    # -O0，模板报错更好读
bash scripts/build.sh --enable_print 00_basic_matmul # 设备侧 cce::printf
bash scripts/build.sh --msdebug   00_basic_matmul    # mssanitizer/msdebug
```

**读模板代码的正确姿势**（不然你会被 `using` 链绕死）：
1. 先看例子里的 8 行装配 → 知道**实例化的是谁**
2. 从 `kernel/basic_matmul.hpp` 的 `operator()` 看**主循环骨架**（scheduler → BlockMmad → epilogue）
3. 再进 `block/block_mmad.hpp` 看对应 `DispatchPolicy` 的特化（`template <> class BlockMmad<MmadAtlasA2Pingpong<...>>`）
4. 千万别从 `tile/` 开始读，那是最后才需要的细节

**让 clangd 能跳转**（✅ 已配好，见第 8 节）：
CATLASS 是纯模板代码，没有 go-to-definition 基本没法读。配置已落地在
`catlass_lab/.clangd` + `catlass_lab/clangd_stub.h` + `~/.config/clangd/config.yaml`，
改完配置后用 `bash catlass_lab/verify_clangd.sh` 一键自检。

---

## 6. 坑清单（会浪费你时间的那些）

| # | 坑 | 处理 |
| :-- | :-- | :-- |
| 1 | **gcc ≥13 直接编不过**：bisheng 用系统 gcc16 的 libstdc++ 头，`__normal_iterator` 报 incomplete type | 当前 shell 的 `CPLUS_INCLUDE_PATH` 已指向 gcc-15 头（`/usr/lib/gcc/x86_64-linux-gnu/15.3.0/include/c++`）。换 shell/重启后先确认这个变量还在，否则把 `g++-15` 设为 `CXX` 或把该路径写进 fish 配置 |
| 2 | 310B/200I 上跑 = 白折腾 | 200I 阶段只做"编译 + 读码"；跑数留到 910B |
| 3 | `CATLASS_ARCH` 默认 2201，950 样例要 `-DCATLASS_ARCH=3510` | 编 950 样例（名字带 `ascend950`）时别忘 |
| 4 | TileShape 不 16 对齐 | Cube 是 16×16×16，所有 M/N/K 分块取 16 的倍数，否则命中低效 padding |
| 5 | `L0TileShape` 的 K 不等于 `L1TileShape` 的 K 是**正常**的（如 64 vs 256） | 这是 L1 内再切分给 L0 的两级流水，别以为是笔误 |
| 6 | Python 3.14 vs 仓库要求 <3.12 | 只有 `gen_data.py` / DSL 工具链受影响；必要时用 `.venv` 里的 3.10 |
| 7 | 旧文档 `CATLASS_AND_TRITON_ASCEND_LEARNING_ROADMAP.md` 里的 `BasicMatmul<ArchTag, ...>`、`examples/27_matmul_gelu` 的 `TileElemWise` 说法 | 以仓库真实代码为准（`BasicMatmul<BlockMmad, BlockEpilogue, BlockScheduler>`），别照抄旧文档 |
| 8 | 一上手就通读 `fai_kernel.cpp`（700+ 行） | 先读 `main.cpp` 的 8 行装配 → 再读 `FAInferKernel` 的 `operator()` 主循环 → 最后才进两个 Epilogue 的实现 |

---

## 7. 已确认 / 待确认

**已确认**（2026-10-20 沟通）：
- 硬件：现在只有 200I DK A2，之后能拿到 910B → 按 1.5 的 5 周计划走
- 目标：自己定的 = 把 FlashAttention 迁到 CATLASS

**待确认**：
1. **910B 大概什么时候到？** 如果 ≤2 周，W1~W4 的"只编译不运行"完全可接受；如果更久，建议中间插一个"用 23 号样例的 CPU golden 造数据"的准备动作。
2. **老师只要 fwd 还是要 bwd？** 仓库里 backward 只有 950 专用样例（`104`/`105`），A2 上没有可抄的 bwd 模板 —— 这个预期要提前说清。
3. **有没有时间盒？** 一周 demo 就砍掉 S3/S5，只做 S0+S1+S2+精读 23；一个月就按 5 周计划 + S3。
4. **最终交付形态**：单独的可执行样例（像 `examples/` 那样）还是要挂进你现有的 `flash_attention_npu` torch 扩展？后者要额外看 `examples/python_extension` 的 torch 绑定写法。

---

## 8. 编辑器 / LSP 配置（已完成，2026-10-20 实测通过）

CATLASS 全是模板，**没有跳转就等于不能读**。这套配置的目标是：在 Zed 里对 CATLASS 的类型/模板能正常 `跳转定义`、`hover`、`补全`，且不报一屏红线。

### 8.1 改了哪些文件

| 文件 | 作用 |
| :-- | :-- |
| `catlass_lab/.clangd` | 只管 `catlass_lab/` 及其子目录：CATLASS 头路径 + 关键宏 + 替身头文件 |
| `catlass_lab/clangd_stub.h` | **给 clangd 用的 AscendC 最小替身**，补 bisheng 内建符号（`half`/`TPosition`/`HardEvent`/`TBuf`/`GetTPipePtr`…） |
| `~/.config/clangd/config.yaml` | 追加了一段 `PathMatch: .*/opensource-project/catlass/.*`，**让 CATLASS 仓库本身的文件**（examples、include）也享受同一套配置 |
| `.zed/settings.json` | clangd 参数调优 + inlay hints + 索引排除 |
| `catlass_lab/.clang-format` | 与 CATLASS 官方仓同款风格（Google/120 列/大括号换行），避免保存即格式化把代码改成 LLVM 默认风格 |
| `catlass_lab/verify_clangd.sh` | 自检脚本，跑一遍就知道配置还活着没 |

### 8.2 三个"必须这么写"的点

1. **`-DCATLASS_ARCH=2201`**
   CATLASS 全仓 137 处 `#if CATLASS_ARCH == 2201` 条件编译。不定义它，`BlockMmad`/`TileCopy` 的 A2 特化根本不参与解析 —— 表现就是"跳进去是空的主模板 + static_assert"。

2. **绝对不能直接 `-include kernel_operator.h`**
   它依赖 bisheng 编译器内建类型 `pipe_t` / `event_t` / `PIPE_*`（CANN 头里根本没有定义，是编译器给的）。
   clang 不认识 → 一个 TU 直接爆 **11400 个 error**，clangd 的 AST 当场崩。
   所以用 `clangd_stub.h` 提供同名替身，并且**预先占住 `ASCENDC_MODULE_OPERATOR_H` 这个 include guard**，让 `#include "kernel_operator.h"` 在 clangd 下变成空操作，避免替身与真实头重复声明。

3. **`-ferror-limit=0`**
   深层 tile 头里可能残留个别解析不了的 AscendC 参数类型。把上限抬掉，保证 clangd 不会"报够 20 个错就放弃建索引"。

### 8.3 `.zed/settings.json` 调了什么（相对你原来的）

| 变更 | 原因 |
| :-- | :-- |
| `--clang-tidy` → `--clang-tidy=false` | 模板库里 clang-tidy 既慢又刷屏，看代码时纯噪声 |
| `--header-insertion=iwyu` → `never` | 自动补 `#include` 在 CATLASS 里几乎必错（它靠模板参数隐式带类型） |
| 新增 `--pch-storage=memory` | 大模板头反复解析，PCH 放内存快很多 |
| 新增 `inlay_hints`（显示类型） | 模板代码里全是 `auto` / `typename X::Y`，类型提示能省一半猜的时间 |
| 新增 `file_scan_exclusions` | 不索引 `.git`/`build`/`output`/`.venv`/`*.o`/`*.so`，项目搜索才干净 |

### 8.4 验证 & 失效排查

```bash
# 一键自检：5 个文件，全 OK 即配置生效
bash catlass_lab/verify_clangd.sh
```

当前结果（2026-10-20 20:19）：

```
clangd_probe.cpp             OK
basic_matmul.cpp             OK
matmul_add.cpp               OK
matmul_gelu.cpp              OK
fai_kernel.cpp               OK
全部通过 ✅
```

⚠️ **一个正常现象**：单独打开某个"实现型"头文件（如 `gemm/block/block_mmad.hpp`、`block_mmad_pingpong.hpp`）时，
会看到几条 `explicit specialization of undeclared template struct 'BlockMmad'` 之类的红线 ——
因为这些头文件不是自包含的（它的主模板声明在别的头里）。**不影响跳转**：
从 `examples/*.cpp` 或 `kernel/basic_matmul.hpp` 里点进去看，上下文是完整的。
另外 `kernel/basic_matmul.hpp` 单独打开是 0 报错的，正好是 W1 要精读的那个文件。

失效时按顺序排查：
1. **路径变了**（仓库搬家/改名）→ 改 `catlass_lab/.clangd` 和 `~/.config/clangd/config.yaml` 里的绝对路径；
2. **改了配置没生效** → Zed 里执行 `editor: restart language server`（clangd 只在启动时读 `.clangd`）；
3. **索引残留** → `rm -rf ~/.cache/clangd/index` 后重启语言服务器（第一次重建会慢几分钟，模板量大）；
4. **新增了 CATLASS 头文件、出现新的红线** → 大概率是某个 AscendC 符号还没进替身，去 `clangd_stub.h` 补一个同名 `struct`/`using`/variadic 函数桩即可（补完跑一次自检脚本）。

> 说明：`clangd_stub.h` 被 `#if defined(__clang__) && !defined(__CCE__)` 包着，**只在 clangd 里生效**；
> bisheng 真机编译永远不会看到它，所以不用担心污染真实构建。
