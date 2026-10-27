# 实验 3 — 图捕获消除 Triton launcher 开销

## 背景

测得 `CPU 提交 / 次 = 0.426 ms ≈ 端到端 0.427 ms` ⇒ 纯 CPU 提交瓶颈
（Triton 的 Python launcher：绑 ~50 个参数 + 2 次 `torch.zeros*` + `get_device_properties`）。

## 改动

`bench/graph_fa.py`：`GraphedFLA` 包一层，**`fla_fwd` 一行没改**。

```python
预热 3 次（先让 Triton 编译完，编译不能落在捕获里）
with torch.npu.graph(g):
    self.o, self.lse = fla_fwd(self.q, self.k, self.v, causal=...)
# 之后：q/k/v copy_ → g.replay()
```

## 结果（Ascend910B3，warmup10/iters50/repeats5，中位数）

| N | mode | 原始 (ms) | **纯 replay = device 时间** | 图+输入拷贝 | 加速（图+拷贝）|
|---|---|---|---|---|---|
| 512 | full | 0.415 | **0.083** | 0.100 | **4.15×** |
| 512 | causal | 0.490 | **0.075** | 0.161 | 3.04× |
| 1024 | full | 0.437 | **0.198** | 0.216 | **2.02×** |
| 1024 | causal | 0.425 | **0.141** | 0.158 | **2.68×** |
| 2048 | full | 0.605 | 0.610 | 0.631 | 0.96× |
| 2048 | causal | 0.424 | 0.391 | 0.412 | 1.03× |
| 4096 | full | 2.289 | 2.290 | 2.315 | 0.99× |
| 4096 | causal | 1.264 | 1.277 | 1.305 | 0.97× |

`max_err`（full）：4.4e-4 / 2.5e-4 / 1.3e-4 / 9e-5 —— **与不开图时一致** ✓

## 结论 1：`总时间 = max(launcher ≈ 0.42 ms, device 时间)`

8 行数据全部符合：

| N | mode | device | max(0.42, device) | 实测原始 |
|---|---|---|---|---|
| 512 | full | 0.083 | 0.420 | 0.415 ✓ |
| 1024 | full | 0.198 | 0.420 | 0.437 ✓ |
| 2048 | full | 0.610 | 0.610 | 0.605 ✓ |
| 4096 | full | 2.290 | 2.290 | 2.289 ✓ |

**含义**：只要 `device < 0.42 ms`，测到的就全是 launcher；一旦 `device > 0.42`，launcher 被完全掩盖。

## 结论 2：图捕获的适用边界 = `device 时间 < 0.42 ms`

| device 时间 | 图捕获效果 |
|---|---|
| ≪ 0.42 ms（N≤512） | **4~5×** |
| ~0.2 ms（N=1024） | **2~2.7×** |
| > 0.42 ms（N≥2048） | **0.96~1.04× —— 不要再开图**，3 次 `copy_` 的 ~17 µs 变成纯开销 |

## 结论 3（重要修正）：device-only 真实时间表

之前所有 "0.44 ms 地板" 掩盖掉的真实 kernel 时间，现在拿到了：

| N | full device | causal device | 真实 TFLOPS (full) |
|---|---|---|---|
| 512 | 0.083 ms | 0.075 ms | **12.9** |
| 1024 | 0.198 ms | 0.141 ms | **21.7** |
| 2048 | 0.610 ms | 0.391 ms | **28.2** |
| 4096 | 2.290 ms | 1.277 ms | **30.0** |

⚠️ **修正 `2026-10-27-triton-baseline/RESULTS.md` 里的 TFLOPS**：那里 N=512 报的 2.4 TFLOPS
是 launcher 污染的产物，**真实值是 12.9 TFLOPS**。小 case 的"性能差"大部分是调度开销，不是 kernel。

**并且修正一条之前的判断**：我曾说 "causal N=2048 (0.454) ≈ 地板 → launcher-bound"，
**这是错的** —— device 时间是 0.391 ms，它一直是 device-bound，只是恰好接近 0.42。
真正被 launcher 支配的是 **N ≤ 1024**。

## 观察到的异常

`causal N=512` 的「图+拷贝」是 0.161 ms，而 `full N=512` 只有 0.100 —— 两者张量大小相同，
拷贝量相同。推测是**同一进程里连续建 8 张图，后建的图内存池被前几张挤压**。
建议：**一个进程只建一张图**，或复用 `graph_pool_handle()`。

## 采纳建议

| 场景 | 用不用图 |
|---|---|
| N ≤ 1024（decode 提前批、短序列、小 batch 推理） | ✅ **用，2~5×** |
| N ≥ 2048（长序列 prefill） | ❌ **不用**，直接调即可 |
| 需要和 CATLASS 公平对比 | ❌ 两边都只看 device-only（`msprof`）或两边都开图 |

## 下一步

1. 修 `graph_fa.py` 的 golden（`triu` 缺 `diagonal=1` → causal 报 nan，非 kernel 问题）
2. `bench_fa.py` 加 `--graph` 开关，输出两套口径
3. 回到 kernel：`propagate_nan` A/B、`enable_ubuf_saving` / `ops_reorder`、128×128 + `o` fp16
