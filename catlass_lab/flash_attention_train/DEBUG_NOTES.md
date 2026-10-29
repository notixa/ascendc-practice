# CATLASS FA 卡死问题 —— 排查结论

> 状态：**已解决**（根因见第 0 节）。M1 已跑通：`max_err ≈ 1e-3`，causal / 非 causal 均正确。
> 后面各节是当时"未解决"阶段的排查记录，保留作为过程史料。


---

## 0. 真正根因（最终定位）

**AIV 侧的 coreIdx 用错了。**

```cpp
// ✗ 错：AIV 有 2×coreNum 个核，直接拿 GetBlockIdx() 当 task 索引
for (uint32_t taskIdx = AscendC::GetBlockIdx(); taskIdx < numTask; taskIdx += GetBlockNum())

// ✓ 对：两个 subcore（2k、2k+1）同属 AIC k，必须折算
const uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
for (uint32_t taskIdx = coreIdx; taskIdx < numTask; taskIdx += GetBlockNum())
```

官方对应位置：`examples/23_flash_attention_infer/fai_kernel.cpp:453`
（AIC 分支在 `:130`，直接用 `GetBlockIdx()`）。

### 为什么这会造成"只有 AIV→AIC 不通"的假象

`numTask = 1`、`GetBlockNum() = 8` 时，AIV blockIdx 0 进入循环，**1～15 全部跳过**。
而 `__mix__(1,2)` 下 AIC k 的跨核 wait 是**汇聚**语义 —— 要它的两个 AIV subcore 都置位才放行。

| 方向 | 语义 | 结果 |
|---|---|---|
| AIC → AIV | 1 对 2 **广播** | 置一次两个都能收到 → **通** |
| AIV → AIC | 2 对 1 **汇聚** | 只来了 subcore 0 的置位 → **永久死锁** |

于是所有"flag id / pipe / mode / dcci"的扫描全部无效 —— 根本不是同步机制的问题，
而是**有一半的 AIV 根本没在干活**。

### 定位它用的一次实验

把打点改成不过滤、且带 `[blockIdx/blockNum]`，一次就暴露：

```
[0/8] AIV enter ... [15/8] AIV enter     ← 16 个都起来了
[0/8] s   [0/8] m                        ← 但只有 subcore 0 走到 softmax/置位
（[1/8] s / [1/8] m 永远缺席）
```

### 打点本身也踩过的三个坑（同样重要）

1. **一次塞 19 条 PRINTF** → device 侧 print buffer 打满，后面的输出被静默丢弃，
   误判成"卡在第 19 条"。单次打点应 ≤ 6 条。
2. **去掉 `\n`** → `AscendC::PRINTF` 不换行就不 flush，输出全空，又误判成"一条都没执行"。
3. **用模糊锚点（"到下一个注释分隔符"）做代码替换** → 把相邻函数头和 `B_Q/B_K` 常量一起删掉，
   连续两次编译失败。改代码要用**整函数/整段**作原子单位，并 `git diff` 核对。

### 另一个同源教训：benchmark 自身也要先验证

`fat_bench` 的 `F32ToF16Exact` 查表从 `a=5` 起整体错位（`1.25` 编成 `4.0`、`2.0` 编成 `32.0`），
导致实际喂进 kernel 的数据比预期大 16 倍、softmax 溢出成 one-hot；
golden 的输出量化也用同一张坏表。表现是 `max_err = 8224`，看起来像 kernel 错，
实际是**测量工具错**。同时 mask 无条件按 causal 生成，`causal=0` 时与 golden 口径不一致。

**在 kernel 通过之前，先独立验证 benchmark（让 golden 与 numpy/torch 对一遍）。**

---

## 1. 现象

`mmadPV`（`BlockMmad<MmadAtlasA2FAPV>`）内部的 `Arch::CrossCoreWaitFlag(softmaxReady)` **永远等不到**，
而 AIV 侧确实执行了 `Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(smReady)`。

最小复现：`N=128, H=1, D=128, blockDim=8`（单 task、单 KV 块）。

```
[AIC] enter
[AIC] qk set
[AIC] before PV
[PV] 1 wait MTE1_MTE2 ok
[PV] 2 copy V->L1 ok
[PV] 3 copy V L1->L0 ok
[AIV] enter
[AIV] sm set            ← AIV 已置位
（卡死，没有 [PV] 4 got smReady）
```

把那一处 wait 跳过（`FAT_PV_NO_WAIT`）后，**整条链跑完**：

```
[PV] 1..9 全过 → [AIC] after PV → [AIC] pv set → [AIV] got pv
```

⇒ **除这一个 flag 外，kernel 的算法、tile、flag 收支、mmadPV/softmax/rescale 全部正确。**

---

## 2. 已验证的事实（都有实测证据）

| # | 事实 | 证据 |
|---|---|---|
| 1 | **环境正常**：官方 `examples/23_flash_attention_infer` 能跑通 | 输出 `Compare Success` |
| 2 | **AIC→AIV 方向完全正常** | AIV 收到了 `qkReady`（softmax 内部的 wait 通过）和 `pvReady`（`[AIV] got pv`）|
| 3 | **AIV→AIC 方向从未生效** | 见下面的扫描表，全部 `exit=124`（超时）|
| 4 | 两个核**是并发的**，且共享 GM | AIV 写 GM 的值 host 侧能读到；AIV 的"入口探针"曾被 AIC 读到 |
| 5 | **AIC 的标量 GM 读会被永久缓存**，`DataCacheCleanAndInvalid` 无效 | AIV 写入后 host 读到 `[0]=1`，而 AIC 轮询 5×10⁶ 次始终读到 `0` |
| 6 | 不是 `blockDim` 的问题 | `blockDim = 1 / 8 / 24` 都卡 |
| 7 | 不是 torch 嵌入的问题 | 独立可执行（纯 ACL 初始化，照官方 main）**同样卡** |
| 8 | 不是同步基地址的问题 | `aclrtGetHardwareSyncAddr` 在 torch 和独立进程里都返回 `0x3fffffb9000` |
| 9 | 不是 `__mix__` 标注的问题 | 加了 `extern "C" __global__ __aicore__ void __mix__(1,2) fat_impl(...)`，编译 warning 消失，**行为不变** |
| 10 | 不是流水深度的问题 | 实现深度-2 软件流水（`PIPE_DEPTH=2`，3 槽 pingpong，照官方 `preLaunch=2`）后**仍然卡在第一块** |
| 11 | 不是 token 计数的问题 | 只让 AIV subblock 0 参与握手，仍然卡 |

### AIV→AIC flag 的完整扫描（全部失败）

| 维度 | 试过的值 | 结果 |
|---|---|---|
| flag id | 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 | 全卡 |
| 置位 pipe | `PIPE_MTE3` / `PIPE_V` / `PIPE_MTE2` | 全卡；`PIPE_S` 直接崩（`aclError 507015`，非法 pipe）|
| 置位 mode | 0,1,2 | 全卡 |
| 置位 mode = 3 | — | 不卡，但 AIC 的 wait **立即返回**（假同步，`max_err` 和"完全没同步"一模一样）|
| 等待 mode | 0,1,2,3 | 无影响 |
| 等待 pipe | `PIPE_S` / `PIPE_V` / `PIPE_MTE1` / `PIPE_MTE2` / `PIPE_MTE3` / `PIPE_FIX` | 全卡 |
| subblock | 两个都置 / 只 subblock 0 置 | 都卡 |

### GM 哨兵方案（绕过 flag）也失败

AIV 往 GM 写哨兵、AIC 轮询：

- ✅ host 侧回读确认 **AIV 写成功了**（`[host] flags after kernel: [0]=1`）
- ❌ **AIC 读不到**：轮询 5×10⁶ 次始终是 0

> 关键细节：`dcci` 的方向必须配对 —— **写侧 `DcciDst::CACHELINE_OUT`（写回）**，
> **读侧 `DcciDst::CACHELINE_ALL`（失效）**。用反了会导致"IIC 读到旧值"或"脏行被丢弃"。
> 即便如此，**AIC 的标量 GM 读仍然拿不到新值**（Cube 核的 DCache 似乎无法被 invalidate）。

---

## 3. 我们的代码 vs 官方 examples/23（已逐项比对，无差异）

| 项 | 官方 | 我们 |
|---|---|---|
| flag id | `QK=1, SM=2, PV=3` | 同 |
| 置位 | `CrossCoreSetFlag<0x2, PIPE_FIX>(qk/pv)`、`<0x2, PIPE_MTE3>(sm)` | 同 |
| 等待 | `CrossCoreWaitFlag(flag)`（默认 mode 0 / PIPE_S） | 同 |
| `SetSyncBaseAddr` | kernel 入口第一句 | 同 |
| 启动 | `<<<blockDim, nullptr, stream>>>`，`blockDim = aicCoreNum` | 同 |
| 核类型标注 | `CATLASS_GLOBAL`（= `__global__ __aicore__`） | 同 + `__mix__(1,2)` |
| `resource` 成员 | `Arch::Resource<ArchTag> resource;` | 同 |
| 流水深度 | `preLaunch = 2`，workspace `preLaunch+1` 槽 | 已实现，仍卡 |

**唯一的实质差异是循环结构**（官方每 task 处理多个 KV 块 + `preLaunch` 流水；
我们每 task 一个 KV 块）—— 但深度-2 流水版本已经补上了这个差异，**仍然卡**。

---

## 4. 我的判断

**AIV→AIC 的 FFTS 跨核 flag 在我们的 kernel 里不工作，而 AIC→AIV 正常**，
且已排除所有软件层面的参数（id / pipe / mode / 时序 / 计数 / 核类型 / 启动方式 / 环境）。

**最可能**：官方 `examples/23` 里某个我们还没注意到的前提条件（例如某个
`AscendC` 初始化调用、或 AIV 侧的 pipe 预置 set），或者是 `MmadAtlasA2FAPV`
这条路径在 CANN 9.0.0 + 910B3 上与官方用的 block 有细微差异。

**建议的下一步（按性价比排序）**：

1. **拿官方的 `FAInferKernel` 直接跑我们的数据**：把 `fai_kernel.cpp` 抄成 `fat_kernel_v2.h`，
   只把 tiling 参数换成我们的（token-major、`headNum*headDim` stride），先跑 `N=128,H=1`。
   它已经被证明能跑，**这是最快"跑通"的路径**。
2. 在官方例程里加 PRINTF，确认它的 `softmaxReady` 握手**确实**在发生（而不是被优化掉/走了别的路径）。
3. 用 `msprof` 抓官方和我们的 kernel，对比 AIC/AIV 的 flag 事件。

---

## 5. 留下的有用资产

| 文件 | 用途 |
|---|---|
| `catlass_op/standalone/fat_main.cpp` | **独立可执行基准**：纯 ACL 初始化、自带 CPU golden、秒级迭代、**零 Python launcher 开销**（正好和 Triton 的 device-only 口径对齐）。用 `./fat_bench N H causal blockDim` |
| `catlass_op/CMakeLists.txt` | `fat_bench` 目标 + `FAT_ENABLE_PRINT`（`AscendC::PRINTF` 需要 `--cce-enable-print`；**CANN 9.0.0 上不加也能打印**，CANN 8.5.0 上会报 `not support in current compilation mode`）|
| `catlass_op/kernel/catlass_kernel.asc` | `__mix__(1,2)` 标注（1 AIC : 2 AIV），编译 warning 消失 |
| Triton 侧 | 已优化的 FA：tile 96×128、`__mix__` 无关、device-only 30 TFLOPS（vs 厂商 122 TFLOPS）|

---

## 6. 给以后写 A2 AIC+AIV kernel 的提醒

1. **`AscendC::PRINTF` 需要 `--cce-enable-print`**（CANN 8.5.0 不支持该模式；9.0.0 上不加也能打）。
   不加的话 PRINTF 被编译成空操作 —— 会让你误判"卡在入口"。
2. **`AIC` 和 `AIV` 的标量 GM 访问不可靠**：AIC 侧的 `GetValue` 会被 DCache 永久缓存，
   `DataCacheCleanAndInvalid` 未必能失效。跨核通信**不要**用 GM 轮询 + 标量读。
3. **`dcci` 方向**：写侧 `CACHELINE_OUT`，读侧 `CACHELINE_ALL`。反了会"静默丢数据"。
4. **`blockDim` 传 `aicCoreNum`**（不是 2×），AIV 会自动是 2 倍。
5. **`__mix__(1, 2)`** 显式标注核类型，否则编译期会警告 `kernel type ... auto derivate may be failed`。
6. 调试顺序建议：**先跑官方同类例子**，确认环境；再 PRINTF 定位；最后才怀疑编译器/平台。
   本次排查最大的教训是：**早期一个设计有缺陷的 pingpong 测试**（只有 AIV0 置位却让 7 个 AIC 等待）
   导致了"跨核 flag 全都不通"的错误结论，白扫了 id/pipe/mode 几十组组合。
