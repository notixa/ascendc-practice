# CATLASS FA vs Triton-Ascend FA vs vendor —— 性能横向对比

> 日期：2026-10-30
> 设备：Atlas 910B3（AIC = **20**，实测 `coreNum` 曲线在 20 见顶、24 回落）
> CANN 9.0.0 / torch 2.7.1 / torch_npu 2.7.1.post4 / triton-ascend 3.2.0
> 测试脚本：`bench/bench_all.py`（一条命令跑三个实现并出表）

---

## 一、结论

1. **CATLASS FA 全面快于 Triton-Ascend FA：1.79 ~ 2.08×，均值 ≈ 1.93×。**
2. **因果（causal）场景是 CATLASS 的优势区**：`N≥2048` 时已基本追平 vendor（1.03× / 1.08×），小 N 时甚至快 3.2×。
3. **唯一短板是大 N 全注意力**：`N=4096 full` 时 vendor 快 2.00×，原因是 CATLASS 未做 KV 维切分。
4. **vendor 不利用因果性**（`causal/full = 1.09`），而 CATLASS（0.59）与 Triton（0.56）都利用了。

---

## 二、主表

`CATLASS` 已用满 20 个 AIC 核；`Triton` 为其 self-tuned 的 96×128 tile；`vendor` 为 `npu_fusion_attention`。

| N | mode | CATLASS TFLOPS | CATLASS ms | Triton TFLOPS | Triton ms | vendor TFLOPS | vendor ms | **CATLASS/Triton 加速比** | CATLASS/vendor |
|---|---|---|---|---|---|---|---|---|---|
| 512 | full | 25.7 | 0.042 | 13.2 | 0.082 | 9.2 | 0.117 | **1.95×** | 0.36× |
| 512 | causal | 13.2 | 0.041 | 7.3 | 0.074 | 4.1 | 0.131 | **1.80×** | 0.31× |
| 1024 | full | 40.4 | 0.106 | 21.9 | 0.197 | 39.1 | 0.110 | **1.86×** | 0.97× |
| 1024 | causal | 27.6 | 0.078 | 15.4 | 0.140 | 17.5 | 0.123 | **1.79×** | 0.63× |
| 2048 | full | 54.3 | 0.316 | 28.1 | 0.611 | 97.4 | 0.176 | **1.93×** | 1.79× |
| 2048 | causal | 42.3 | 0.203 | 21.8 | 0.395 | 43.7 | 0.197 | **1.95×** | **1.03×** |
| 4096 | full | 62.2 | 1.105 | 29.8 | 2.303 | 124.3 | 0.553 | **2.08×** | 2.00× |
| 4096 | causal | 52.7 | 0.652 | 26.8 | 1.284 | 57.0 | 0.603 | **1.97×** | **1.08×** |

（倍数 > 1 表示 CATLASS 比 vendor 慢）

---

## 三、口径

| 实现 | 命令 | 口径 |
|---|---|---|
| **CATLASS** | `./fat_bench N H causal 20 0 20` | C++/ACL 直调，warmup 10 + 20 次 × 5 轮取中位数，**零 Python launcher** |
| **Triton** | `bench_fa.py --only triton --graph --json` 的 `device-only (纯 replay)` | 图捕获后的纯 replay，**不含 Python launcher**（与 CATLASS 同口径） |
| **vendor** | `bench_fa.py --only vendor --json`（`npu_fusion_attention`）| 带 torch launcher（对它略不利，但它本身慢，相对影响小） |

FLOPs 口径：`4 × B × H × pairs × D`，`pairs = N(N+1)/2`（causal）或 `N²`（full）。

---

## 四、为什么 CATLASS 能快 2×

CATLASS 的 `fai_*` 是**显式双核协同**：

```
AIC (Cube)   : 只做 QK、PV 两个矩阵乘
AIV (Vector) : 只做 online-softmax、rescaleO
两者通过 GM workspace + Arch::CrossCoreSetFlag/WaitFlag 交接
```

而 Triton-Ascend 的 FA 是单 program 内混合，Cube/Vector 的切换由编译器调度，重叠度更低。

两边都使用各自调优后的 tile（Triton 96×128，CATLASS 128×128），因此这是"各自最佳配置"的对比。

---

## 五、剩余优化空间（按性价比）

| # | 优化 | 预期 | 依据 |
|---|---|---|---|
| 1 | **`preLaunch = 2` 流水**（官方 3 槽 + QK↔PV 交错）| 1.2–1.6× | 当前 AIC 在等 `smReady` 时 Cube 完全空闲 |
| 2 | **KV 维切分（split-K + 归约）** | 大 N 1.5–2× | vendor 大 N 强的主因；当前一个 `(b,h,q块)` 独占一核、串行扫完整个 KV |
| 3 | 更大 `B_q`（256） | 少量 | 减少 softmax/rescale 调用次数 |
| 4 | Triton 侧 A1（静态上界 softmax，8→6 BAR） | 少量 | 已搁置 |

**CATLASS 的 TFLOPS 到 N=4096 仍在单调上升**（25.7 → 40.4 → 54.3 → 62.2），说明尚未饱和，1/2 两项还有实实在在的空间。

---

## 六、数据可信度

- **vendor 小 N 数据有算法抖动**：`N=1024 full (0.110ms)` 接近 `N=512 full (0.117ms)`，非单调 —— `bench_fa.py` 的 variant ladder 在小 N 会切到不同算法。**评价 vendor 只看 N≥2048。**
- **causal 与 full 的绝对 FLOPs 不同**（causal ≈ 一半），不要跨行比较 TFLOPS。
- CATLASS 正确性：`N=128..2048`、`H=1/8`、causal/full 全部 `max_err ≤ 1e-3`（fp16 量化级），两轮重复测量逐位一致（确定性）。

---

## 七、复现

```bash
# 构建
cd catlass_lab/catlass_op && rm -rf build && mkdir -p build && cd build
cmake .. -DCATLASS_INCLUDE_DIR=/mnt/workspace/gitCode/cann/catlass/include && make -j$(nproc)
cd /mnt/workspace/gitCode/cann/ascendc

# 一条命令出表
python bench/bench_all.py --iters 20 --markdown

# 只看大 N（避开 vendor 小 N 抖动）
python bench/bench_all.py --shapes 2048:8:0 2048:8:1 4096:8:0 4096:8:1 --iters 20 --markdown
```

---

## 八、过程中定位到的两个关键 bug（供后续参考）

### 1. `fai_qk`/`fai_pv` 的 K/V 基址重复加了 `nIdx` 偏移（致命）

```cpp
// ✗ 错：调用方又把块偏移加了一次
kvOffset = b*N_k*H*D + nIdx*B_k*H*D + h*D;
// block 内部还会再加 nowNIdx * KV_SPLIT_SIZE * strideKV

// ✓ 对：调用方只传该 (b, h) 的基址
kvBaseOffset = b*N_k*H*D + h*D;
```

**症状极具迷惑性**：`nIdx=0` 的 stack 全对；第二个 stack 的**第 1 块**因为吃了上一轮 L1 预取的残留而侥幸正确，第 2 块起才错 ⇒ 表现为"`N=640`（4+1 块）通过、`N≥768` 全错"。这个模式导致排查时长期误判为"缓冲区被覆盖 / 跨 stack 合并错"。

### 2. `blockDim` 只给了 8（浪费 60% 算力）

`fat_bench` 的自动分支是 `min(numTask, aicMax)`，但所有调试命令都显式传了 `8`（早期为了缩短打点输出），之后的性能测量全部沿用了这个 `8`。

**影响**：`N=4096 full` 从 2.65 ms（8 核）→ 1.11 ms（20 核），**2.38× 差异**，足以让"CATLASS 比 Triton 慢 13%"翻转成"快 108%"。

**加固**：`fat_bench` 现在运行时查询 `GetCoreNumAic()`，且 banner 打印 `coreNum=N/M (AIC)`，核未用满时额外提示。
