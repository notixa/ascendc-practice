# bench —— CATLASS FA vs Triton-Ascend FA

对比两个自己写的 FlashAttention forward：

| 后端 | 位置 | 布局 | 语义 |
|---|---|---|---|
| `catlass` | `catlass_lab/catlass_op` | `[B, N, H, D]` 或 `[B*N, H, D]` | causal（靠 mask 表） |
| `triton` | `triton_lab/kernels/flash_attention/fa_fwd.py` | `[B, H, N, D]` | **非 causal**（KV 循环跑满） |

---

## 一、先对齐语义：这不是小事

`fla_fwd` 的 KV 循环是 `for j in range(T_k)` —— **跑满全部 KV 块，没有因果跳过**。而 CATLASS 版是 causal（只在 `j == nDiag` 的对角块上 mask）。

直接比 = 拿"算了 N² 的一半"和"算了全部 N²"比，结论没有意义。

**对齐办法**：给 CATLASS 传一张**全 0 mask**。

```python
mask = torch.zeros(128, 128, dtype=torch.float16).npu()
```

全 0 = 不屏蔽 = full attention —— 因为 `S + 0 × (−3e38) = S`。这样两边算的 FLOPs 完全相同，比的是**实现效率**。

真实 causal 差距作为第二次实验单独测（CATLASS 少算约一半，理论上应快 ~2 倍）。

---

## 二、Roofline：这个 workload 是 compute-bound

每 head 的 FLOPs 与理想访存（Q/K/V/O 各读一次）：

| N | FLOPs/head（full） | 理想访存/head | 算术强度 |
|---|---|---|---|
| 512 | 0.13 G | 512 KB | **256** FLOP/B |
| 1024 | 0.54 G | 1 MB | **512** FLOP/B |
| 2048 | 2.15 G | 2 MB | **1024** FLOP/B |
| 4096 | 8.59 G | 4 MB | **2048** FLOP/B |

AI = `N/2`（causal）或 `N`（full）。910B 的算力/带宽比大约在 `300 TFLOPS / 1.6 TB/s ≈ 190 FLOP/B` 量级 ——

> **N ≥ 512 就已经跨过 roof 的拐点，是纯 compute-bound。**
> 所以唯一有意义的性能指标是 **达到的 TFLOPS / 峰值**，而不是带宽利用率。

填 `--peak-tflops <datasheet值>` 后脚本会直接输出 `%peak`。

---

## 三、怎么跑

```bash
# 前提：catlass_op 已经编出 libcatlass_fat.so
source ~/Ascend/cann-8.5.0/set_env.sh
cd ~/codes/ascendc
python3 bench/bench_fa.py --peak-tflops <你的峰值>
```

其它开关：

| 开关 | 用途 |
|---|---|
| `--only catlass` / `--only triton` | 只测一个（另一个还没跑通时） |
| `--skip-correctness` | 闸门已过，只测性能 |
| `--atol 2e-2` | 正确性阈值（fp16 + softmax，2e-2 是常用值） |
| `--so <path>` | `libcatlass_fat.so` 的位置 |

---

## 四、测量方法

- `warmup = 10`，`iters = 50`，`repeats = 5`
- 每轮用一个 `torch.npu.Event` 对夹住 50 次调用，除以 50 得单次延迟
- 5 轮取**中位数**，同时打印 `[min, max]`
- **波动 > 10% 会打警告** —— 波动比改进幅度还大时，先修测量，别下结论

### 三个必须排除的干扰

1. **layout 转换不算在计时内**：Triton 要 `[B,H,N,D]`、CATLASS 要 `[B,N,H,D]`，脚本在两个后端各自内部做 `permute().contiguous()`。
   ⚠ 严格说这是**适配成本**，真实集成时其中一方必然要付这笔转置。如果要用它，就把计时包在外面单独测一次。
2. **workspace 分配不算在计时内**：CATLASS 版的三块 S/P/oTemp 在 host wrapper 里 `at::empty`（PyTorch 有缓存分配器，多次调用后开销很小），Triton 版只有 `O` 和 `L` 两个输出。
3. **golden 只在闸门里算**（CPU fp32），不进计时。

---

## 五、正确性闸门（先过再谈性能）

脚本对**两个后端都按 full attention 对齐**同一份 CPU fp32 golden。任一后端 `max_err >= atol` 就跳过计时。

> 这是硬性前置：CATLASS 版目前**还没在真机上验证过数值**。它跑得快但算错，不是"性能好"，是"没结果"。

---

## 六、解读结果

汇总表最后一列 `catlass/triton`：

- `> 1` → CATLASS 更慢
- `< 1` → CATLASS 更快
- 差距 < 20% 基本可以认为"同一档"，因为 fp16 下的测量噪声和 shape 敏感性常有这个量级

**如果 CATLASS 更慢，先查这三件事**（按概率排序）：

1. 它现在是**零流水**（`preLaunch=0`、单槽）：AIC 每块都要等 AIV 消费完才继续 → 理论上有 2× 左右的损失
2. 它**没做 stack**（4 块一组）：每块 2 次跨核同步 → 同步开销 ×4
3. 它的 tile 是 128×128×128，而 Triton 在 D=128 时用 `BLOCK_Q=64, BLOCK_K=64` —— **Cube 的利用率可能差一档**

**如果 CATLASS 更快**，大概率是因为：
- 它用 Cube 做 `S = QKᵀ` 和 `PV`（Triton 在 Ascend 上也是 `tl.dot`，但 tile 更小）
- 它的 softmax 有专门的 UB 分区和 pingpong（`Ls/Lp/Lo/Go` 那套手工布局）

---

## 七、后续实验（一次只改一个变量）

按 perf-loop 的规矩，每次记录：假设 → 改动 → 测量 → 结论。

| # | 假设 | 改动 |
|---|---|---|
| 1 | CATLASS 的零流水是主要瓶颈 | 开 `preLaunch=2` + 3 槽 ring |
| 2 | 跨核同步次数是瓶颈 | 4 块一组（stack），同步 ÷4 |
| 3 | tile 太小导致 Cube 利用不足 | 试 `BLOCK_Q/BLOCK_K = 128`（Triton 侧）|
| 4 | causal 的算法优势有多大 | CATLASS 真实 mask vs Triton full，报"每 FLOP 效率"看是否同档 |
| 5 | 多 head 并行度不够 | 加大 `B*H`，看是否饱和 |

实验记录放在 `bench/experiments/YYYY-MM-DD-<假设>/`。
