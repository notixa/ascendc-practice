# Triton-Ascend FA **device-only 基准**（2026-10-27）

> **这是后续所有优化的参考表。**
> 之前 `2026-10-27-triton-baseline/RESULTS.md` 里的数字含 Triton 的 Python launcher 开销
> （~0.42 ms），小 case 的 TFLOPS 全是污染值，**以本表为准**。

## 口径

`python3 bench/bench_fa.py --only triton --mode both --graph`

`--graph` 默认用 **pure replay** 计时 ⇒ 数字里没有：
- Triton 的 Python launcher（绑参、grid、`torch.zeros*`、`get_device_properties`）
- 输入 `copy_`

⇒ **这是纯粹的 device 执行时间**（= 若用 `msprof` 采 kernel time 应该看到的值）。

环境：Ascend910B3 / CANN 9.0.0 / triton 3.2.0 / torch 2.7.1
tile：`BLOCK_Q=96, BLOCK_K=128`（实验 2 的结果）
测量：warmup=10 / iters=50 / repeats=5，中位数；波动 < 1%

## 结果

| case | mode | **device (ms)** | TFLOPS | max_err |
|---|---|---|---|---|
| B1 H8 N512 | full | **0.080** | 13.5 | 3.0e-4 |
| B1 H8 N512 | causal | **0.068** | 7.9 | 9.1e-4 |
| B1 H8 N1024 | full | **0.194** | 22.2 | 2.0e-4 |
| B1 H8 N1024 | causal | **0.136** | 15.8 | 9.9e-4 |
| B1 H8 N2048 | full | **0.611** | 28.1 | 1.1e-4 |
| B1 H8 N2048 | causal | **0.386** | 22.3 | 1.1e-3 |
| B1 H8 N4096 | full | **2.290** | **30.0** | 1.0e-4 |
| B1 H8 N4096 | causal | **1.272** | 27.0 | 9.7e-4 |
| B4 H8 N2048 | full | **2.366** | 29.0 | 1.8e-4 |
| B4 H8 N2048 | causal | **1.402** | 24.5 | 1.0e-3 |

## 观察 1：TFLOPS 在 N ≥ 2048 时饱和于 **~30 TFLOPS**

| N | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|
| full TFLOPS | 13.5 | 22.2 | 28.1 | **30.0** |

小 N 效率低不是因为 kernel 差，而是**固定开销占比大**（见观察 3）。

## 观察 2：causal/full 收敛到 **0.556**（理论块数比 0.508）

| N | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|
| causal/full | 0.85 | 0.70 | 0.63 | **0.556** |

单调收敛，N=4096 时 0.556 vs 理论 0.508 → **还有 ~9% 的块粒度浪费**
（`BQ=96` 下对角块的部分行是浪费的，`BQ=BK=64` 时浪费最小）。

## 观察 3：小 N 有 ~40~60 µs 的**固定 device 开销**

用实验 2 的成本模型 `t = 2.31 µs × iters/核 + W` 对照：

| N | mode | 预测 | 实测 device | 残差 |
|---|---|---|---|---|
| 512 | full | 38 | 80 | **42** |
| 512 | causal | 24 | 68 | **44** |
| 1024 | full | 144 | 194 | 50 |
| 2048 | full | 576 | 611 | 35 |
| 4096 | full | 2273 | 2290 | 17 |

**⇒ N=512 的 80 µs 里，约 42 µs（52%）是固定开销（kernel launch / 收尾）**，
真正可优化的只有 ~38 µs。

**所以 N=512 的 kernel 侧天花板 ≈ 1.5×**（80 → 56 µs），再想快只能减少 launch 次数（融合其它算子）。

## 结论

1. **大 N（≥2048）的基准是 28~30 TFLOPS**，后续 kernel 优化的对比都用这张表。
2. **小 N 已被固定开销支配**，收益空间有限；那里的收益应该来自**算子融合**而非单算子调优。
3. `max_err` 全部在 1e-4~1.1e-3 ✓（之前 causal 报 nan 是 golden 的 `triu` 漏了 `diagonal=1`，非 kernel 问题）。

## 下一步（kernel 侧，按预期收益）

| # | 实验 | 预期 | 备注 |
|---|---|---|---|
| 1 | `propagate_nan` 开关 A/B | ±3% | 代码已有，从未测过差异 |
| 2 | `enable_ubuf_saving` / `ops_reorder` | 1~4% | KB 30 参数扫里唯二正收益 |
| 3 | 128×128 + `o` fp16 | **1.17×** | UB 224→192KB，精度有风险 |
| 4 | 手工 CV 软件流水（`dot(p,v)` 提前发射） | ? | ~20 行 |

对 CATLASS 的对比：**必须两边都用 device-only 口径**（CATLASS 侧用 `aclrtGraph` 或 `msprof`）。
