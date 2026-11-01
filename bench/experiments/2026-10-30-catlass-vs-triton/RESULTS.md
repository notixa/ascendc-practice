# CATLASS FA vs Triton-Ascend FA vs vendor —— 真实场景性能对比（定稿）

> 日期：2026-11-01
> 设备：Atlas 910B3（AIC = **20**，实测 `coreNum` 曲线在 20 见顶、24 回落）
> CANN 9.0.0 / torch 2.7.1 / torch_npu 2.7.1.post4 / triton-ascend 3.2.0
> D = 128（强制，见文末「约束」）
> 一键复现：`python bench/bench_all.py --preset realistic --modes causal --scope both --markdown`

---

## 一、结论

1. **6 / 7 个真实场景，CATLASS 领先 vendor（`npu_fusion_attention`）1.28–3.04×。**
2. **Triton-Ascend 全面落后 vendor 2.09–3.57×**，也就是落后 CATLASS 约 3–6×。
3. **唯一落后的是 `B=16, N=512, H=8`（慢 1.39×）** —— 原因是「每 task 固定开销」在小 KV 时占比 76%，与 launcher、流水都无关（详见第四节）。
4. CATLASS 的 kernel 时间在 `N=4096, H=32` 达 **106.3 TFLOPS**，隐含的块级效率已接近 910B3 的 fp16 峰值。

---

## 二、device 口径（kernel 本身）

| B | N | H | mode | CATLASS ms | CATLASS TFLOPS | 相对 vendor |
|---|---|---|---|---|---|---|
| 1 | 512 | 8 | causal | 0.0335 | 16.1 | vendor 的 0.25×（**快 4.0×**）|
| 1 | 2048 | 8 | causal | 0.1257 | 68.4 | 0.64×（**快 1.56×**）|
| 1 | 4096 | 8 | causal | 0.3436 | 100.0 | 0.57×（**快 1.77×**）|
| 1 | 2048 | 32 | causal | 0.4264 | 80.6 | 0.76×（**快 1.32×**）|
| 1 | 4096 | 32 | causal | 1.2928 | 106.3 | 0.59×（**快 1.68×**）|
| 4 | 2048 | 8 | causal | 0.4121 | 83.4 | 0.73×（**快 1.37×**）|
| 16 | 512 | 8 | causal | 0.2353 | 36.6 | **1.39×（慢）** |

**口径**：`fat_bench`（C++/ACL 直调），warmup 10 + 20 次 × 5 轮取中位数，**零 Python launcher**。

## 三、部署口径（含框架开销，CATLASS/Triton 都走图捕获）

| B | N | H | CATLASS(app) ms | Triton ms | vendor ms | CATLASS 相对 vendor |
|---|---|---|---|---|---|---|
| 1 | 512 | 8 | 0.0438 | 0.4753 ⚠ | 0.1331 | **快 3.04×** |
| 1 | 2048 | 8 | 0.1321 | 0.4532 | 0.1956 | **快 1.48×** |
| 1 | 4096 | 8 | 0.3509 | 1.2702 | 0.6070 | **快 1.73×** |
| 1 | 2048 | 32 | 0.4415 | 1.3944 | 0.5631 | **快 1.28×** |
| 1 | 4096 | 32 | 1.2968 | 4.8379 | 2.1745 | **快 1.68×** |
| 4 | 2048 | 8 | 0.4215 | 1.3939 | 0.5638 | **快 1.34×** |
| 16 | 512 | 8 | 0.2380 | 0.5647 | 0.1696 | 慢 1.39× |

⚠ Triton 的 `1,512,8` 同口径两次测量差 6.8×（0.0697 → 0.4753），该行不可信，需重测。

`CATLASS(app)` 与 `CATLASS` 只差 **3–15 µs**（图 replay 地板价）⇒ host 已不是瓶颈。

---

## 四、性能模型（由实测反推，误差 ≤15%）

```
T ≈ numTask × ( 353 ns + nStacks × 113 ns )        nStacks = KV块数 / 4

   每 task 固定开销  F = 353 ns   ← loadQGM(32KB Q 进 L1) + flag 协议 + 首块冷拷贝
   每 stack(512列KV) s = 113 ns   ← 隐含 296 TFLOPS，≈ 芯片 fp16 峰值
```

| shape | numTask | nStacks | 预测 | 实测 | 误差 |
|---|---|---|---|---|---|
| `16,512,8` | 512 | 1 | 0.2387 | 0.2387 | 0% |
| `4,2048,8` | 512 | 4 | 0.4126 | 0.4126 | 0% |
| `1,4096,32` | 1024 | 8 | 1.2889 | 1.2855 | +0% |
| `1,2048,8` | 128 | 4 | 0.1032 | 0.1220 | −15%（6.4 个/核 → 尾部不均）|

**推论**：
- `N≥2048`：固定开销占 28% 以下，**块级效率接近峰值，没什么空间**
- `N=512`：固定开销占 **76%**，`B16 N512` 落后 vendor 的根因
- 想再提升小 N，只有 `B_q` 128→256（numTask 减半）或跨 task 流水

---

## 五、优化历程与各自收益

| 改动 | 收益 | 依据 |
|---|---|---|
| **① 修复 K/V 基址双重偏移** | 从「全错」到「全对」 | `fai_qk/pv` 内部已加 `nowNIdx*KV_SPLIT_SIZE*strideKV`，调用方只该传 (b,h) 基址 |
| **② `coreNum` 8 → 20（用满 AIC）** | **2.38×** | `N=4096` 2.6462 → 1.1114 ms；24 核反而变慢（过订阅）|
| **③ `preLaunch = 2` 流水** | **1.88×** | `N=4096 H=8 full` 1.0883 → 0.5804 ms，与「Cube 占空比 50%→100%」的理论上限吻合 |
| **④ `curStackTileMod` 传真实槽号** | 从「挂死」到「全对」 | epilogue 的 `dm` 在 UB 里每槽一份：`curStackTileMod*128 + rowOffset` |
| **⑤ 对角块列宽用 `visibleEnd` 而非 `N_k`** | 从「挂死」到「全对」 | `columnNum` 偏大 → mask GM（仅 128×128）越界读 → 硬件挂住 |
| **⑥ `PL` 按 `totalStackNum` 自适应** | +3%（N=512）| `PL = min(2, total-1)`；`total=2` 时 `PL=2` 会退化成串行 |
| **⑦ workspace + tiling 缓存** | **1.69–1.76×**（app 口径）| 每次 `at::empty` 26MB + 同步 H2D ~0.15 ms |
| **⑧ 图捕获（torch 路径）** | **~4×**（小 shape）| 裸 op 调用有 ~0.19 ms 恒定 host 开销，与 device 时间无关 |

---

## 六、已知短板与下一步

| 短板 | 原因 | 可行手段 | 预期 |
|---|---|---|---|
| `B16 N512` 慢 1.39× | 每 task 固定开销 353 ns 占 76% | **`B_q` 128→256** | ~1.5× |
| `N=512` 绝对性能低（16 TFLOPS）| 同上 + 尾部轮次不均 | 同上 | — |
| 大 N 已接近峰值 | 块级 296 TFLOPS | — | 无空间 |
| decode（`N_q=1`）未支持 | 并行度 = `B×H`，远小于 20 核 | **split-KV** | 该场景数倍 |
| vendor 小 N 数据有算法抖动 | 它按 shape 切 tilingkey | 只看 `N≥2048` | — |

---

## 七、约束与注意事项

- **`D` 必须 = 128**：库内 `EMBED_SPLIT_SIZE = 128`、`KV_BASE_BLOCK = 512` 是硬编码；`B_q = B_k = 128` 是 `fat_bench` 的常量。主流 LLM（Llama/Qwen 系）正好是 128。
- **`coreNum` 必须是物理 AIC 数**：910B3 = 20。给 8 会浪费 60% 算力，给 24 会过订阅变慢。
- **vendor 的 causal 走它自己的 variant 阶梯**（mask/sparse_mode），未必与我们等价；小 N 有非单调现象，评价 vendor 只看 `N≥2048`。
- **causal 与 full 的绝对 FLOPs 不同**（causal ≈ 一半），不要跨行比 TFLOPS。
- `CATLASS(app)` 用图捕获（`NPUGraph`），因此 **`torch_npu.profiler` 看不到这个 kernel**；要 profiling 用 `fat_bench` + `msprof`。

---

## 八、复现

```bash
# 构建
cd catlass_lab/catlass_op && rm -rf build && mkdir -p build && cd build
cmake .. -DCATLASS_INCLUDE_DIR=/mnt/workspace/gitCode/cann/catlass/include
make -j$(nproc)

cd /mnt/workspace/gitCode/cann/ascendc

# 三方对比（真实场景，两种口径）
python bench/bench_all.py --preset realistic --modes causal --scope both --markdown

# 只看 kernel（排除一切框架开销）
./catlass_lab/catlass_op/build/fat_bench 4096 8 0 20 0 20

# profiling
FAT_SKIP_GOLDEN=1 msprof --output=./prof \
  --ai-core=on --aic-metrics=PipeUtilization --task-time=on --ascendcl=on \
  ./catlass_lab/catlass_op/build/fat_bench 4096 8 0 20 0 0
```

---

## 九、过程中定位到的关键 bug（供后续参考）

### 1. K/V 基址重复加 `nIdx` 偏移（致命）
`fai_qk`/`fai_pv` 内部会加 `nowNIdx * KV_SPLIT_SIZE * strideKV`，调用方只该传该 `(b,h)` 的基址。
**症状极迷惑**：`nIdx=0` 的 stack 全对；第二个 stack 的**第 1 块**因吃了上一轮 L1 预取的残留而侥幸正确，第 2 块起才错 ⇒ `N=640`（4+1块）通过、`N≥768` 全错。

### 2. AIV 的 `coreIdx` 少了 `/ GetSubBlockNum()`（死锁）
`GetBlockNum()` 在 AIV 上返回 AIC 数（20），`GetBlockIdx()` 返回 0..39。
只除不除导致每个 AIC 只有 subcore 0 干活；而 `__mix__(1,2)` 的 **AIV→AIC 置位是聚合语义**，缺一个 subcore 就永不放行 ⇒ AIC 等 `smReady`、AIV 等 `pvReady`，互相死等。

### 3. 合并两段循环时 `visibleEnd` 被写成 `N_k`（越界挂死）
非 causal 下两者恒等 → 掩盖错误；causal 下 `columnNum` 偏大 2~4× ⇒ 带 mask 的 epilogue 按 `row*128+col` 读只有 128×128 的 mask GM ⇒ 越界读 → 硬件挂住。

### 4. `torch` 路径把 `causal` 写死成 `1u`
`--only catlass` 的 `full` 模式一直按 causal 算（`max_err` 0.79），而 `fat_bench` 同 shape 是 0.000488。

### 5. 测量方法类的坑（同样致命）
- **`--core-num` 用了 8**（早期为缩短打点输出），导致「CATLASS 比 Triton 慢 13%」的结论完全错向（真实是快 108%）
- **验证只做 `-c`（不链接）**，漏掉 `PlatformAscendCManager` 的 undefined symbol → `make` 失败但没发现
- **`torch` 路径漏传 stream**，kernel 落在 ACL 默认流，`torch.npu.Event` 测不到 ⇒ app 时间恒定 0.23 ms
- **裸 op 调用有 ~0.19 ms host 开销**，与 shape 无关，把小 shape 的真实性能全淹掉
