# Triton-Ascend FA 优化计划

基线数据见 [`2026-10-27-triton-baseline/RESULTS.md`](../2026-10-27-triton-baseline/RESULTS.md)。
本文档只列**下一步要做的实验**，按预期收益排序，每条都注明判别依据与失效边界。

---

## 0. 先把瓶颈算清楚：3.5 µs/迭代 = 8 个 BAR

基线实测（N=4096, full）：总迭代 16384，20 核 → 819 次/核，2.860 ms → **3.5 µs / 每次 KV 迭代**。

KB 卡 `flash_attention_optimization.md` 给出的逐指令数据是 **`BAR`(pipe_barrier) ≈ 712 cycle/条**：

```
0.4375 µs/BAR × 1.63 GHz ≈ 712 cycle   ✓ 与 KB 卡实测一致
3.5 µs / 0.4375 µs ≈ 8 个 BAR / 迭代   ✓
```

而当前循环体的**向量依赖链正好 8 步**：

| # | 循环体内语句 | 产生 1 个 BAR |
|---|---|---|
| 1 | `s = tl.dot(q, tl.trans(k))` | ✓ |
| 2 | `s = tl.where(...)`（causal mask） | ✓ |
| 3 | `m = tl.maximum(m, tl.max(s,-1))` | ✓ |
| 4 | `s = tl.exp(s - m[:,None])` | ✓ |
| 5 | `l = factor*l + tl.sum(s,-1)` | ✓ |
| 6 | `o = factor[:,None] * o` | ✓ |
| 7 | `s = s.to(fp16)` | ✓ |
| 8 | `o = tl.dot(s, v, o)` | ✓ |

**⇒ 两个正交的杠杆**：

```
代价 = 迭代数 × 每迭代 BAR 数 × BAR 代价
       ↑A            ↑B            (硬件常数, 712 cycle)
```

- **杠杆 A：减迭代数** → 放大 BLOCK_Q/BLOCK_K（当前被 UB 卡住）
- **杠杆 B：减每迭代 BAR 数** → 缩短向量依赖链（当前 8 → 6 有可能）

---

## A. 杠杆 B：缩短依赖链（预期收益最大且可行）

### A1. 静态上界 softmax —— 去掉 `factor` 相关 2 个 BAR  【预期 ~25%】

**思路**：`factor = exp(m_prev - m)` 存在的唯一理由是 `m` 会变。如果我们用**一个先验上界**当 `m` 且永不更新，则 `factor ≡ 1`：

```
去掉：o = factor[:,None] * o        （第 6 步）
去掉：factor * l                    （第 5 步退化成 l += sum(s)）
⇒ 8 BAR → 6 BAR
```

**上界怎么取**（精确且便宜）：由 Cauchy-Schwarz，`|q_i·k_j| ≤ ||q_i||₂ · ||k_j||₂`，所以

```
m_static = (max_i ||q_i||₂) × (max_j ||k_j||₂) / sqrt(D)
```

`max_i ||q_i||₂` 需要扫一遍 q（`N×D`），KL 复杂度只有 `O(N·D)`，相对 `O(N²·D)` 的主循环可忽略。

**判别依据**：改完 `lse` 应与原来一致（差异只来自 exp 下溢边界）。
**失效边界**：若 `m_static` 远大于真实 `max(s)`（例如 q/k 范数分布极不均匀），`exp(s - m)` 大面积下溢 → 精度掉。
**必须先验**：`m_static - max(s)` 的分布（打印 p99，> 30 就要小心，fp32 exp 在 -80 以下才彻底下溢）。

### A2. causal 拆成「前缀段 + 对角段」两个循环  【预期 ~10%（仅 causal）】

当前 mask 在每个迭代都做（第 2 步）。但 causal 下**只有最后 1 个 KV 块需要 mask**，前面全是全通：

```python
# 伪代码
for j in range(j_diag):          # 无 mask，7 个 BAR
    ...
# 对角块（1 次）
s = where(mask, s, -inf)         # 8 个 BAR
```

N=4096 时前缀段占 31/32 → `(31×7 + 8) / (32×8) = 0.878` → **约 12% 提升**。

⚠️ 注意：这和 KB 卡 §3.2「掩码链合并 → 0%」不矛盾。那条减的是**指令条数**，这条减的是**依赖链长度（BAR 个数）**。

### A3. `propagate_nan=ALL` 分档  【预期 2~5%，零风险】

```python
if MULTI_BLOCK:
    m = tl.maximum(m_prev, tl.max(s, axis=-1), propagate_nan=tl.PropagateNan.ALL)
else:
    m = tl.maximum(m_prev, tl.max(s, axis=-1))
```

默认 `NONE` 会被展开成 7 条向量指令（2×isnan + vmax + 2×vsel），`ALL` 对应单条 `vmax`。

**语义等价前提**（写进注释）：`s` 永不含 NaN —— 输入有限、`m` 用有限值初始化（已是 `-1e38`）、mask 用 `-inf` 而非 NaN。
**必须分档**：KB 实测全局开启 **−9%**，单块 case 退化 25~45%。

---

## B. 杠杆 A：减迭代数（当前被 UB 卡住）

### B1. 探 BLOCK 上限：`64/128` → `128/64` → `128/128`

当前 `64/128` 与 `128/64` 的迭代数**相同**（`ceil(N/64)×ceil(N/128) = ceil(N/128)×ceil(N/64)`），所以换形状**本身不省迭代**。
真正省一半的是 **`128/128`**（迭代数 ÷2）。

**但 D=128 fp16 的 UB 估算**：

| 缓冲 | 128×128 时 |
|---|---|
| q (fp16) | 32 KB |
| k (fp16) | 32 KB |
| v (fp16) | 32 KB |
| s (fp32) | **64 KB** |
| o (fp32) | **64 KB** |
| 合计 | **≈ 224 KB > 192 KB(UB)** ✗ |

与 KB 卡记录的「D=128 fp16 在 128×128 报 `PlanMemory Failed`、D=160 反而编得过」一致。

**判别**：直接试，看编译报错 `ub overflow` 里的数字。
**若失败，解锁路径**（按风险排序）：

| 手段 | 省多少 | 风险 |
|---|---|---|
| `o` 累加器用 fp16 | 32 KB | 精度（长序列累加） |
| `s` 用 fp16 存（softmax 在 fp16 做） | 32 KB | 精度（exp 范围小） |
| `128×96`（BLOCK_K=96） | 8 KB | 迭代数只省 25% |

⚠️ **每一项都必须重跑精度闸门**，且这些改动会同时影响 A1/A2 的收益（结构性改动后要重扫 tile，见 KB §2.5）。

### B2. `kv_lo` 不对齐  → **对本 kernel 不适用**

`kv_lo` 永远等于 0（causal 无滑动窗口），没有「对齐浪费」可省。跳过。

### B3. causal 区间收缩 → **已完成**

已实现，实测 `causal/full = 0.542`，与离线暴力验证的理论块数比 **50.8%** 吻合。无需再动。

---

## C. 分核与调度

### C1. LPT 任务排序（重活优先）  【预期 2~4%】

causal 下 `i` 越大活越重（`i=0` 只 1 个 KV 块，`i=63` 有 64 个）。grid-stride 顺序分配会让重活集中在编号末尾。

```python
i0 = (task % (head * blocks_q)) // head
i  = blocks_q - 1 - i0        # ★ 反序；少写 -1 会导致 i 越界（已踩过）
```

⚠️ **必须加 host 侧断言**（5 行，纯 Python）：

```python
for task in range(batch * head * blocks_q):
    i = blocks_q - 1 - ((task % (head * blocks_q)) // head)
    assert 0 <= i < blocks_q, f"i={i} 越界（task={task}）"
```

### C2. 小 case 的计时口径

N ≤ 1024 时实测被 Triton 的 Python launcher 支配（~0.44 ms 地板）。要测小 case 的真实 kernel 时间：

```bash
export TRITON_BENCH_METHOD=npu        # autotune 内部用 profiler
msprof --application="python3 ..."    # 或直接采 device-only 时间
```

**这是"测量方法"优化，不是 kernel 优化**，但决定小 case 的结论是否可用。

---

## D. 不要做的（KB 卡 §3，省时间）

| 方向 | 实测 |
|---|---|
| 14 个编译开关逐个单测 | 命中率 **0**，全在 ±2% 噪声内 |
| `multibuffer=True` | **−10%** |
| `enable_mixed_cv=True` | **−9%**（只在 910_95 生效） |
| 「减少向量指令条数」整类（掩码链 12→4 等） | **0%**（三次证伪）—— 注意与 A2 的区别：A2 减的是 BAR 数 |
| q/k/v `.contiguous()` 布局连续化 | 4 case 一个都没变快，另一算子 15/17 变慢 |
| `V.to(fp32)` 再 dot | 要在 UB 物化 64KB fp32 副本，负收益 |
| num_warps / num_stages | Ascend 不支持 |

---

## 实验顺序（一次只改一个变量）

| # | 实验 | 预期 | 前置 |
|---|---|---|---|
| 1 | 量 `m_static - max(s)` 的分布 | — | A1 的前提 |
| 2 | **A3** `propagate_nan` 分档 | 2~5% | 无 |
| 3 | **A2** causal 拆两段循环 | ~12%（causal） | 无 |
| 4 | **A1** 静态上界 m | **~25%** | 实验 1 通过 |
| 5 | **C1** LPT 排序 | 2~4% | 加断言 |
| 6 | **B1** 试 128×128，看 UB | 可能 2× | 无 |
| 7 | B1 失败 → 降 UB（fp16 o/s） | 解锁 6 | 精度闸门 |

每次记录到 `bench/experiments/YYYY-MM-DD-<假设>/`：

```
## 实验 N — <一句话假设>
- 基线：<上次的 median latency>
- 改动：<只写一处>
- 测量方法：warmup/iters/repeats、取哪个统计量
- 结果：<数值 + 波动>
- 结论：<支持/推翻>，因为 <...>
- 下一步：
```

---

## 附：TFLOPS 与峰值的口径

基线 24 TFLOPS。910B 系列 fp16 峰值需查 datasheet（`[待确认]`）。
按 ~300 TFLOPS 量级估算 ≈ **8%**，与 KB 卡「CUBE 仅占 9.7%」互相印证。
**报性能时不要只给 TFLOPS，要给「迭代数 × BAR 数」的分解** —— 那个才是可优化的量。
