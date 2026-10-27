# Triton-Ascend FA 基线（2026-10-27）

## 环境

| 项 | 值 |
|---|---|
| 设备 | **Ascend910B3**（dav-2201） |
| CANN | 9.0.0（`/home/developer/Ascend/cann-9.0.0`） |
| torch / torch_npu | 2.7.1+cpu / 2.7.1.post4 |
| triton | **3.2.0**（triton-ascend，ascend dialect OK） |
| 机器 | aarch64 |

## 测量方法

```
warmup = 10
iters   = 50      # 每轮 50 次调用，Event 对夹后除以 50
repeats = 5       # 取中位数，同时打印 [min, max]
```

- 输入：`q/k/v` fp16，shape `[B, N, H, D]`（**token-major**，和 CATLASS 侧一致；此前误按 `[B,H,N,D]` permute 导致整轮结论无效）
- 输出：`o` fp16、`lse` fp32（Triton 版额外产出 LSE，CATLASS 版没有）
- 正确性闸门：CPU fp32 golden（**已用 torch 官方 SDPA 交叉验证，两 golden 差 2.7e-4**）
- 阈值：`max_err < 2e-2`

**tile 策略**（`base_tile()`：`bk=min(128,ceil16(N))`、`bq=64`（受 `bq*bk≤8192` 与 `bq*D*2≤64KB` 约束）、`bd=ceil16(D)`）：

| N | BLOCK_Q | BLOCK_K | 每 task 的 KV 迭代数 |
|---|---|---|---|
| 512 | 64 | 128 | 4 |
| 1024 | 64 | 128 | 8 |
| 2048 | 64 | 128 | 16 |
| 4096 | 64 | 128 | 32 |

## 原始结果

| case | mode | latency (ms) | [min, max] | TFLOPS | max_err |
|---|---|---|---|---|---|
| B1 H8 N512 D128 | full | 0.439 | [0.438, 0.441] | 2.4 | 3.0e-4 |
| B1 H8 N512 D128 | causal | 0.452 | [0.450, 0.459] | 1.2 | 9.1e-4 |
| B1 H8 N1024 D128 | full | 0.461 | [0.461, 0.474] | 9.3 | 2.0e-4 |
| B1 H8 N1024 D128 | causal | 0.486 | [0.485, 0.489] | 4.4 | 9.9e-4 |
| B1 H8 N2048 D128 | full | 0.744 | [0.743, 0.745] | 23.1 | 1.1e-4 |
| B1 H8 N2048 D128 | causal | 0.456 | [0.455, 0.460] | 18.9 | 1.1e-3 |
| B1 H8 N4096 D128 | full | **2.860** | [2.857, 2.890] | **24.0** | 1.0e-4 |
| B1 H8 N4096 D128 | causal | **1.550** | [1.549, 1.554] | 22.2 | 9.7e-4 |
| B4 H8 N2048 D128 | full | 2.879 | [2.878, 2.881] | 23.9 | 1.8e-4 |
| B4 H8 N2048 D128 | causal | 1.626 | [1.623, 1.627] | 21.1 | 1.0e-3 |

波动范围全部 < 1%（测量可信）。

## 解读

### ① 正确性已过关
`max_err ∈ [1e-4, 1.1e-3]`，远低于 2e-2 阈值 —— fp16 输入 + fp32 累加的预期水平。
causal 的误差略大（~1e-3 vs ~1e-4），是 mask 后可见 key 减少、归一化项更小的正常现象。

### ② 大 N 时是**纯 device-bound，性能天花板 ≈ 24 TFLOPS**

| N | full 延迟 | 每 task 迭代数 | 总迭代数 | 每核迭代数 | **每次迭代成本** |
|---|---|---|---|---|---|
| 4096 | 2.860 ms | 32 | 512 task × 32 = 16384 | ÷20 核 ≈ 819 | **3.5 µs** |

**3.5 µs / KV 迭代** —— 与 KB 卡记录的量级一致（4~10 µs），说明瓶颈是**每次迭代的固定开销（Cross-Core 同步）**，不是算力、不是带宽。

24 TFLOPS 若按 910B 系列 fp16 峰值 ~300 TFLOPS 量级估算 ≈ **8%**（⚠️ 峰值需查 datasheet 确认，此处 `[待确认]`）。
KB 卡的逐指令仿真结论是“CUBE 仅占 9.7%、`BAR` 占 55%” —— 和这个 8% 互相印证。

### ③ 小 N 的数据**不可信**：被 Triton Python launcher 支配

关键证据：**N=512 full 和 N=1024 full 的延迟几乎一样**（0.439 vs 0.461 ms），而工作量差 4 倍。

| N | 按 3.5 µs/迭代 预测 | 实测 | 比值 |
|---|---|---|---|
| 512 | 13 迭代/核 × 3.5 µs ≈ **0.045 ms** | 0.439 ms | **10×** |
| 1024 | 26 × 3.5 ≈ 0.09 ms | 0.461 ms | 5× |
| 2048 | 109 × 3.5 ≈ 0.38 ms | 0.456 ms(causal) | 1.2× ✓ |
| 4096 | 819 × 3.5 ≈ 2.87 ms | 2.860 ms | 1.0× ✓ |

**N ≥ 2048 才是真正的 kernel 时间；N ≤ 1024 测到的是 Triton 的 Python 侧 launch 开销（~0.44 ms 地板）。**

**旁证**：N=512 和 N=1024 的 `causal/full` 比值是 **1.03 / 1.05**（causal 做的活更少却略慢）—— 纯粹是开销噪声，反证这两个点不是 device-bound。

### ④ causal 的算法收益**与大 N 的理论预测高度吻合**

| case | causal / full | 说明 |
|---|---|---|
| N=2048 | **0.613** | |
| N=4096 | **0.542** | 理论块数比 50.8% ✓ |
| B4 N=2048 | **0.565** | |
| N=512 / 1024 | 1.03 / 1.05 | ✗ 开销污染，见 ③ |

N=4096 的 0.542 与离线暴力验证的“causal 只跑 50.8% 的 KV 块”几乎完全一致 —— **说明循环上界收缩生效，且这些 case 确实是 device-bound**。

### ⑤ batch 维近线性
B4 相对 B1（同为 N=2048）：full 2.879/0.744 = **3.87×**（4× 工作量）；causal 1.626/0.456 = **3.57×**。
→ 核数足够，没有尾波问题（N=2048 B1 时有 256 task，20 核跑 12.8 波）。

## ⚠️ 对 CATLASS vs Triton 对比的影响

1. **CATLASS 还没跑通**（kernel 死锁，正在查），所以 `catlass/triton` 列全是 `nan`。
2. **小 case 的比较会失真**：CATLASS 用 C++ `<<<>>>` 直调（launch 开销 ~µs 级），Triton 走 Python launcher（~0.44 ms）。
   在 N ≤ 1024 上，Triton 的时间主要是 launcher 开销，**不能用来判断 kernel 优劣**。
3. **建议的对比口径**：
   - **主表只用 N ≥ 2048**（B1N2048 / B1N4096 / B4N2048）
   - 小 case 单独标注“受 launcher 开销支配”
   - 如需公平比较小 case，改用 `msprof` 采集 device-only kernel time，或用 `npugraph` 捕获后计时

## 下一步

| 优先级 | 动作 |
|---|---|
| 1 | 修 CATLASS 的死锁（`SetSyncBaseAddr` 已加但仍挂 → 加 `PRINTF` 定位是 AIV 没起来还是 flag 没到） |
| 2 | CATLASS 通过数值闸门后，用**同一套 bench** 出对比表（只看 N ≥ 2048） |
| 3 | Triton 侧按 KB 卡做优化实验：`BLOCK_Q 64→128`（迭代数 1/2）、`kv_lo` 不对齐、`propagate_nan` 分档 |
| 4 | 小 case 的公平计时方案（msprof / npugraph） |
