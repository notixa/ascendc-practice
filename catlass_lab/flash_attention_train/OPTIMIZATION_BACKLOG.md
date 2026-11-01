# CATLASS FA 性能优化 —— 待办清单

> 建立于 2026-11-01。每项都附**实测依据**，避免以后重新推导。
> 当前基线：`N=4096 H=32 causal` = **1.29 ms / 106.3 TFLOPS**（20 核，preLaunch=2）。

---

## P0 · `LOAB_BLOCK` 1 → 2（或 4）：削减 AIC 的标量开销

### 依据（msprof `PipeUtilization`，中位数 / 32 次 kernel）

```
                    A(无 preLaunch)   B(preLaunch=2)    变化
总时长                 184.2 us        105.1 us       -43%
AIC busy               145.0 us         84.5 us       -42%
  └ mac (Cube)          30.7 us         30.7 us        -0%   ← 一字不差
  └ scalar              75.7 us         79.6 us        +5%   ← ★ 占 AIC busy 的 94%
  └ mte2                39.3 us         58.5 us       +49%
  └ fixpipe             29.0 us         38.3 us       +32%
AIV busy               146.3 us         85.7 us       -41%
  └ vec                 46.8 us         46.8 us        -0%   ← 一字不差
  └ scalar              42.6 us         44.0 us        +3%
```

**两个结论**：

1. **preLaunch 的作用已被证实**：`mac` / `vec` 绝对时间**完全不变**，总时长却砍掉 43%
   ⇒ 只发生"重叠"，没有减少任何工作。
2. **流水已完美平衡**：`AIC busy 84.5` vs `AIV busy 85.7`，差异仅 1.4%
   ⇒ 再提升必须**同时**压两边。而两边的绝对大头都是 **`scalar`**：
   - AIC: `scalar 79.6 us = 94%` 的 AIC 时间，`mac` 只占 36%
   - AIV: `scalar 44.0 us = 51%`

### 假设

`LOAB_BLOCK = 1`（`include/catlass/gemm/block/block_mmad_fai_{qk,pv}_*.hpp`）意味着
**每个 KV 块都要跑一遍完整的标量地址计算**（`getBlockShape` / `getKVOffset` /
`CopyGmToL1` 的 layout 推导 / 循环控制 / 跨核 flag）。把 `LOAB_BLOCK` 提到 2/4，
标量循环次数直接减半 / 减到四分之一。

### 改动面与风险

| 项 | 说明 |
|---|---|
| 改动点 | `block_mmad_fai_qk_normal.hpp` / `_head_tail.hpp` / `fai_pv_*` 里的 `LOAB_BLOCK` 常量 |
| 直接受影响的逻辑 | `nkBlockLoop` / `nkBlockNextIdx` 里那些标 `// gap` 的换算；`blockStackIdx += LOAB_BLOCK`；`LayoutS/LayoutP` 的 tile 划分 |
| 已知约束 | `KV_BASE_BLOCK = 512` 是 S/P 的 GM 行宽 = `STACK_NUM × B_k`；`LOAB_BLOCK` 变大后 L1 里要同时放更多块，注意 L1 容量（QK 用 0~96KB、PV 用 96KB 之后）|
| 风险等级 | **中高**。这是库内模板常量，不只是我们调用侧，改完要重跑全部 15 个正确性 case |
| 先验证什么 | 改完先看 `msprof` 的 `aic_scalar_time` 是否真的下降（不看时间看 pipe）|

### 预期

按比例粗估：`AIC scalar 79.6 → ~45 us` ⇒ `AIC busy 84.5 → ~55 us`。
但 `AIV busy 85.7` 会变成新的瓶颈 ⇒ **净收益取决于 AIV 能否同时改善**。
乐观 ~1.3×，悲观 ~1.05×。**必须先测再改**。

### 备选（若 `LOAB_BLOCK` 风险太大）

- 把 `getKVOffset` / `getBlockShape` 里能提到循环外的地址计算上提
- 减少每块一次的 `CrossCoreSetFlag/WaitFlag`（合并相邻 stack 的 flag）
- 注意：这两项都要改库内模板，风险类似

---

## P1 · `B_q` 128 → 256：摊薄每 task 固定开销

### 依据（实测反推的性能模型，误差 ≤15%）

```
T ≈ numTask × ( 353 ns + nStacks × 113 ns )
     F = 353 ns/task  ← loadQGM(32KB Q 进 L1) + flag 协议 + 首块冷拷贝
     s = 113 ns/stack ← 隐含 296 TFLOPS，≈ 芯片 fp16 峰值
```

| shape | numTask | nStacks | 预测 | 实测 | 固定开销占比 |
|---|---|---|---|---|---|
| `16,512,8` | 512 | 1 | 0.2387 | 0.2387 | **76%** |
| `4,2048,8` | 512 | 4 | 0.4126 | 0.4126 | 35% |
| `1,4096,32` | 1024 | 8 | 1.2889 | 1.2855 | 28% |

**这是 `B=16, N=512, H=8` 落后 vendor 1.39× 的唯一根因**（不是 launcher，不是流水）。

### 改动面

- `fat_main.cpp` 的 `B_q` / `B_k` 常量（现在是 `const uint32_t B_q = 128, B_k = 128;`）
- `LayoutMask{B_q, B_k, B_k}` 的 mask 表尺寸（现在是 128×128 硬编码）
- epilogue 的 `qNBlockSize` / subblock 行切分（`qSBlockSize = qRowNum`，要按 256 重算）
- workspace 的 `ElemsS/ElemsP/ElemsOt`（跟着 `B_q` 翻倍）
- `fat_bench` 的 golden 与实际 shape 对齐

### 预期

`numTask` 减半 ⇒ 固定开销摊薄，`N=512` 场景 **~1.5×**；`N≥2048` 几乎无收益（固定开销本来就只占 28%）。

---

## P2 · 跨 task 流水：消除 task 之间的排水

每个 task 结束时要把 `PL` 个 stack 的 PV/rescale 排空，下一个 task 的第 0 拍又要重新填充。
`numTask` 大时这部分浪费显著（`N=512` 时每 task 只有 1~2 个 stack）。

- 需要 persistent kernel 风格的"task 间不断流"，或把 task 索引也纳入流水
- 难度大，且与 P1（`B_q=256`）的效果部分重叠 —— **先做 P1 看收益再决定**

---

## P3 · split-KV：解锁 decode 场景

当前并行度 = `numTask = B × ⌈N_q/128⌉ × H`。decode 时 `N_q=1` ⇒ `numTask = B×H`，
远小于 20 核（`B=1,H=8` 时只有 8 核在干活，60% 芯片空闲）。

按 KV 维切分 + 二次归约（online softmax 可结合）⇒ 并行度 ×S。

- 参考：`fai_kernel.cpp` 的 `block_epilogue_online_softmax_FD.hpp` / `rescale_o_FD.hpp`（FD = FlashDecoding）就是官方实现
- 代价：workspace 存 `numTask × S × 128×128 fp32` 部分和 + 一个归约 pass
- **只在 `numTask < coreNum` 时才划算**，不要无条件开

---

## P4 · causal 的变长 `preLaunchStackNum`（低优先级）

官方 causal 路径的延迟量是**变长**的：

```cpp
// fai_kernel.cpp:284-288
noMaskTailInteStackNum = (noMaskKvS / pagedBlockSize) % blockStackNum;
preLaunchStackNum = (maskType != 0) ? ((preLaunch - 1) * blockStackNum + noMaskTailInteStackNum)
                                    : (preLaunch * blockStackNum);
```

我们现在统一用 `PL * STACK_NUM`。**数值已验证正确**（15/15 case ≤1e-3），
只影响少数"块号映射在边界不整"的情形 ⇒ 收益增量预计很小，**除非发现特定 shape 的 causal 明显偏慢再动**。

---

## 附：测量方法（避免重复踩坑）

| 坑 | 教训 |
|---|---|
| `coreNum` 给了 8 | 910B3 有 20 AIC。给 8 会得出"CATLASS 比 Triton 慢"的**反向结论** |
| 只做 `-c` 编译验证 | 漏掉链接期 undefined symbol（`PlatformAscendCManager`）⇒ `make` 失败却没发现 |
| torch 路径漏传 stream | kernel 落在 ACL 默认流，`torch.npu.Event` 测不到 ⇒ app 时间恒定 0.23 ms，与 shape 无关 |
| 裸 op 调用当部署性能 | 有 ~0.19 ms 恒定 host 开销 ⇒ 小 shape 全被淹掉。要用 `--graph` 图捕获 |
| 用 `max_err` 判断因果性 bug | `visibleEnd` vs `N_k` 在非 causal 下恒等 ⇒ 错误被掩盖；**causal 必须单独跑，且从最小 case 起** |
| `--aic-mode=task-based` 想画波形图 | A2 的 msprof **不落** kernel 内部的逐周期 pipe 数据（`TASK_PMU_INFO` 是 task 级聚合）⇒ 只有柱状图，没有泳道波形。要波形得上 CANN Simulator |

**采集命令（A/B 对比时两版参数必须完全一致）**：

```bash
FAT_SKIP_GOLDEN=1 msprof --output=./prof \
  --ai-core=on --aic-metrics=PipeUtilization \
  --task-time=on --ascendcl=on --aic-mode=task-based \
  ./fat_bench 4096 1 0 20 0 1
```

看这几个数（`mindstudio_profiler_output/op_summary_*.csv`）：
`aicore_time` / `aic_mac_time` / `aic_scalar_time` / `aic_mte2_time` / `aic_fixpipe_time` /
`aiv_time` / `aiv_vec_time` / `aiv_scalar_time` / 各自的 `_ratio`。
