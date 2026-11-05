# Triton-Ascend FlashAttention 调优记录（910B3）

> 姊妹文档：`bench/README.md` 讲的是 **CATLASS vs Triton 的语义对齐与实验方法**；本文只记
> **Triton 侧调优的结论与排除清单**。

平台：**Ascend 910B3（dav-2201，A2）**，20 AIC : 40 AIV，UB 192 KB（1572864 bits），L0C 128 KB，**910B 无 L0C→UB**。
工具链：**triton-ascend 3.6.0.post2 + bishengir 2.0.0**（CANN 9.1.0）。
基准形状：`B=1, N_q=N_k=4096, H=32, HEAD_DIM=128, causal`，FLOPs = `4·B·H·N(N+1)/2·D`。

---

## 1. 当前结果

| 版本 | 时间 | TFLOPS | 相对 |
|---|---|---|---|
| 升级前（3.2.1 / bishengir 1.1.0）| 3.790 ms | 35.4 | 0.61× |
| `(64,256)` + 3.6.0/2.0.0 | 3.257 ms | 42.2 | 0.86× |
| **`(64,512)` + `n_k>512` 守卫（当前最优）** | **2.310 ms** | **59.5** | **1.00×** |
| CATLASS（Ascend C）参考 | 1.293 ms | 106.3 | 1.79× |

正确性：`bench/triton_check.py` 6/6 通过，`o_err` 3.7e-4 ~ 5.7e-4
（口径 `max|o-ref| / max|ref|`，阈值 2e-2）。

---

## 2. 真正的杠杆：`BLOCK_K` 256 → 512（+41%）

**这是整轮唯一实质有效的优化，也是一个被 `_TILES` 表掩盖很久的低垂果实。**

| tile | 时间 | TFLOPS | UB 需求 |
|---|---|---|---|
| `(32,512)` | 3.747 ms | 36.7 | — |
| **`(64,512)`** | **2.310 ms** | **59.5** | 装得下 |
| `(96,512)` | — | — | 1572864 bits（正好等于上限，仍失败）|
| `(128,512)` | — | — | 2097152 bits（256 KB，超 1.33×）|
| `(64,1024)` | — | — | 4325376 bits（4.1 MB）|

- **`BLOCK_Q` 下限 = 64**：32 时每核重复读 `N/32` 次 K/V，`mte2` 直接翻倍。
- **`BLOCK_K` 上限 = 512**：由 UB 决定（`s`/`p` 各 `BQ×BK×2 B`）。
- **没有剩余空间**：`128x512` 缺 64 KB，`64x1024` 缺 4 MB。

### ⚠️ 必须保留的守卫

```python
if n_k % 512 == 0 and n_k > 512:
    return 64, 512
```

`(64,512)` 在 **`N_k == BLOCK_K == 512` 且 `causal=False`** 时触发 aicore **507015**
（单次迭代 + 循环末尾 `tl.advance` 越过 `block_ptr` 边界）：

| 配置 | 结果 |
|---|---|
| `(64,512)` `N_k=512` non-causal | ❌ 507015 |
| `(64,512)` `N_k=1024 / 2048 / 4096` non-causal | ✅ |
| `(64,512)` `N_k=512` causal | ✅ |
| `(64,256)` `N_k=512` non-causal | ✅ |

---

## 3. 排除清单（20 项，全部有实测证据）

### 3.1 kernel 结构类

| # | 尝试 | 结果 | 原因 / 证据 |
|---|---|---|---|
| 1 | 4 路分组（一次算 4 个 q 块）| 无效 | 被放大 `BLOCK_K` 严格支配；TTIR 与 HIVM 双层确认 UB 峰值按放大的 `BLOCK_K` 算 |
| 2 | spill + STACK（`s` 溢到 GM 换更大 tile）| 净负 | `s` 过 GM 使段数 2→3 |
| 3 | `extract_slice` 分片（旧假设：UB 有余、指令是瓶颈）| 方向相反 | 当时 UB 紧、指令多 |
| 4 | `tl.trans(k)` → 转置指针 | 免费 | 已折进 `nd2nz` |
| 5 | `q_idx` 提到循环外 | 0.15% | 编译器 LICM 已做 |
| 6 | `tl.sum` 用 fp16 | 慢 0.65% | 且精度变差 |
| 7 | `tl.range` vs 裸 `range` | 0.3% | 同会话 |
| 8 | `loop_unroll_factor` | 无效 | — |
| 9 | **显式 GM workspace（S/P/OTmp + `sub_vec` 切分）** | **最好 0.81×** | 见 §5 |

### 3.2 编译选项类

| # | 尝试 | 结果 | 原因 / 证据 |
|---|---|---|---|
| 10 | `multibuffer=False` | 无效 | 只映射成布尔 `--enable-auto-multi-buffer` |
| 11 | **`num_warps`** | **完全忽略** | 只在 `is_pure_simt=True` 时传给 bishengir（`compiler.py:1486-1491`）；我们有 `tl.dot` ⇒ 走 HIVM mix 路径 ⇒ **二进制相同** |
| 12 | **`num_stages`** | **只决定布尔** | `compiler.py:751-761`：`==1` 关多缓冲，其余值全开；**默认已是 2** ⇒ 传 2/3 完全等价 |
| 13 | `tile_mix_vector_loop` / `tile_mix_cube_loop` | ±0.1% | — |
| 14 | `enable_mixed_cv` | −0.03% | — |
| 15 | `set_workspace_multibuffer=4` + `limit_auto_multi_buffer_only_for_local_buffer=False`（官方 `dsa_prefill.py:490` 组合）| 0.3% | 只作用于编译器自己插入的 GM 临时区，对用户传入指针无效 |
| 16 | `multibuffer_mode=[(gm,N)]` | 编译失败 | bishengir 拒该格式；且与 `num_stages` 互斥（`compiler.py:1387` 会 raise）|
| 17 | **`enable_preload=True`**（`--enable-preload`，skew/PreLaunch）| **0.998×** | 无效 |
| 18 | **`enable_dynamic_cv_pipeline=True`** | 快 2×，**算错** | pass 被跳过（`only supported on 91095`），只剩副作用 `disable_auto_inject_block_sync=True` ⇒ 丢跨核同步 |
| 19 | `disable_auto_inject_block_sync=True` | 2.162 ms，**结果错** | 与 `base` 输出逐张量对比：`rel=2.56`（N=4096）、`rel=11.9`（N=512 full）、N=320 出 NaN |
| 23 | **核内 tiling + `tl.static_range`** | **负收益** | `acc` 是 loop-carried ⇒ 展开成 N 份独立 SSA 值 ⇒ **UB 线性增长**（1/4/8 chunk → 3153/5620/10385 K）；chunk 每翻倍慢 20~70% |
| 24 | **核内 tiling + 普通 `range`** | UB 解耦成功、**但慢 76~127%** | UB 确实只由 `CHUNK` 决定（`BK=512/1024/2048` 都是 1,593,344）⇒ **`CHUNK` 思路在 UB 侧成立**；但每 chunk 一次 `factor*acc` ⇒ 一次 `acc` 的 L0C→GM 往返 |
| 25 | **两趟法（消除 rescale）+ `CHUNK`** | **慢 2.2~3.2×** | `s` 算两遍 + dot 缩成 `CHUNK` 尺寸 ⇒ **`BLOCK_K` 失去"少同步"的作用**（`BK` 只成了外层计数壳）|
| 26 | **`--set-cv-pipeline-mode=skew`（= PreLaunch）** | **死锁 / NaN** | 单独用**死锁**；配 `--disable-auto-inject-block-sync=true` 变 **NaN**。对比 `unroll`（无变化）、`off`（编译报错）⇒ **skew 确实改变 codegen，但要求的同步模型与 A2 的 `injectBlockSync` 不兼容** |
| 20 | **UB 类 4 个 flag**（`--enable-lazy-loading` / `--disable-multi-buffer-on-ub` / `--plan-memory-strategy=largest-first` / `multibuffer-mode=[(ub,1)]`）| **全部无效** | 判据是"能否编过更大 tile"：四者的最大可行 tile 都停在 `64x512`，无一救活 `128x512` |

### 3.3 明确不可用的路

| 尝试 | 结论 |
|---|---|
| `al.scope` + 手写 `sync_block_set/wait` | **死锁**：flag 插在 `bind-sub-block` pass 之前，两段仍按程序顺序 ⇒ 双向等待。`ttadapter` IR 的 `func.func × 1` 证实 |
| 官方 `dsa_prefill` 的 `sub_vec` 切分 | 方法可行（`NVEC=2` 结果正确），但**必须过 GM**；GM 路径本身慢 2×，收益被吃掉 |

---

## 4. 关键机制（省得再查）

### 4.1 硬件
- 20 AIC : 40 AIV = **1:2** ⇒ 每个 `block_id` 有 1 cube + 2 vector 子核
  （`sub_block_id` 形如 `cube0` ×20、`vector0/vector1` ×40）。
- UB **192 KB** / 核；L0C 128 KB；**910B 上 L0C→UB 不存在**，跨核数据必须过 GM。
- UB→L1 直连是 A5 专属（`fixpipe`、`copy_from_ub_to_l1` 都带 `is_910_95()` 守卫）。

### 4.2 `al.sub_vec_id()` 可用，但编译器**不会**自动切分 vector 工作

微基准（纯 vector：mask + max + exp + sum，`64x256`，`REPS=200`，20 核）：

```
NVEC=1（每核做全量）: 0.387 ms
NVEC=2（每核做半量）: 0.240 ms      ⇒ 1.61×
```

⇒ 官方显式切分有直接性能理由，不只是省 UB。
但切分**必须过 GM**（Triton 张量不支持运行时切片，`tl.split` 只能沿最后一维），
而 GM 路径慢 2× ⇒ 净收益为负。`al.extract_slice` 是 SSA 路径上的唯一出路（见 §8）。

### 4.3 同步开销的真相
`base` 里约 **56%** 是跨核等待（`FA_NOSYNC` 侧证实有 2.118 ms 属于纯等待）。
但**没有安全的消除手段** —— `disable_auto_inject_block_sync` 能拿回 1.5~2×，却让结果错（#19）。
simulator 佐证：`BAR 37.1% + WAIT_FLAG 23.5% + SET_FLAG 13.2% + MOV 17.3% = 91.1%`。

### 4.4 `_pick_tile` 必须手工（不能用 autotune）
Ascend autotuner 在批量 benchmark 时**会静默丢掉需要更多 UB 的 config**，且一个 config 失败会把
整批判成 `No valid triton configs. NoneType: None`。
实测 `(64,256)` 直调 3.257 ms，放进 autotune 后只剩 `(96,128)` 4.908 ms —— **白丢 34%**。

---

## 5. 显式 GM workspace 的完整负面结论

照官方 `dsa_prefill.py` 的结构（cube 写 GM → 每个 AIV 子核读写自己的半份）重写后：

| 版本 | 时间 | 正确性 |
|---|---|---|
| `fa_fwd.py`（纯 SSA，`64x256`）| 3.257 ms | ✅ |
| 显式 GM，`32x256`，`NVEC=1` | 6.645 ms | ✅ |
| 显式 GM，`64x256`，`NVEC=1` | 3.940 ms | ✅ |
| 显式 GM，`64x256`，`NVEC=2` | 4.052 ms | ✅ |
| 显式 GM，`64x512` | ~2.9 ms | — |

**三条硬性写法约束（都踩过坑）：**

1. **GM workspace 必须用普通指针 + 在调用点内联二维广播**：
   `tl.load(p_base + rows[:, None] * stride_q + cols[None, :] * stride_k)`。
   预计算成 `off = ...` 张量会让别名分析失效，且物化 `[BQ, BK]` 的 int 索引张量
   （6 个就 393 KB）直接吃爆 UB。
2. **workspace 不能用 `tl.make_block_ptr`**：block_ptr 是不透明描述符，别名分析看不到地址 ⇒ 丢同步
   （实测 `o_err ≈ 0.9`）。
3. **不用 `al.scope` / `al.sync_block_*`**（见 §3.3）。

**慢 2× 的根因**：单槽 workspace ⇒ 每轮覆盖写同一块 ⇒ store 无法与 compute 重叠。
`set_workspace_multibuffer` / `multibuffer_mode` 都救不了（#15/#16）。

---

## 6. 方法论教训（最重要的部分）

1. **先扫朴素的 tile，再碰编译器 pass。** 20 项排除里 19 项无效，唯一有效的是 `BLOCK_K`
   256→512。`_TILES` 表只到 256、`_pick_tile` 只判断 `% 256` —— 这个盲区让后面所有
   "高级"尝试都建立在错误基线上。
2. **A/B 必须同会话。** 跨会话漂移 ~3%，同会话极差 0.2~0.4%。
3. **`FA_BH` 这类注入的环境变量不在缓存 key 里** ⇒ 每个配置必须换 `TRITON_CACHE_DIR`，
   否则读到旧二进制。
4. **对比要逐张量比 `base` 的输出**，不能只看 `o_err`：有一次验证脚本
   `_LAUNCH.clear()` 后忘了设 flag，白测出"6/6 通过"的假象。
5. **`N=4096` 不能算 golden**（`[1,32,4096,4096]` fp32 = 2 TB）⇒ 该 shape 只验有限值和耗时，
   精度靠 `N ≤ 2048` 覆盖。
6. **grep 会吞掉错误。** 过滤 `UserWarning` 时容易连 traceback 一起滤掉，导致"没有任何输出"。

---

## 7. 怎么跑

```bash
source /home/developer/Ascend/cann-9.1.0/set_env.sh
REPO=/mnt/workspace/gitCode/cann/ascendc
VENV=$REPO/triton_lab/.venv/bin/python

$VENV $REPO/bench/triton_check.py                 # 精度（6 case）
$VENV $REPO/bench/triton_check.py --perf          # 精度 + 性能
$VENV $REPO/bench/ab_launch.py --tiles 64x512     # launch 级 A/B
FAT_TILES=64x512 $VENV $REPO/bench/triton_check.py --perf
FA_NVEC=2 FAT_TILES=128x512 $VENV $REPO/bench/triton_check.py --perf
```

环境变量：

| 变量 | 作用 | 默认 |
|---|---|---|
| `FAT_TILES` | 强制 tile，如 `64x512` | 按 `_pick_tile` |
| `FA_NVEC` | AIV 子核切分（1=关，2=开）| `1` |
| `FA_PRELOAD` | `--enable-preload`（实测无效）| 未设 |
| `FA_DCP` | `enable_dynamic_cv_pipeline`（会算错）| 未设 |
| `FA_BH` | 任意 bishengir flag（需换 `TRITON_CACHE_DIR`）| 未设 |

---

## 8. 下一步：`extract_slice` 子核切分解锁 `128x512`

`128x512` 目前需 2097152 bits（256 KB），超 192 KB 的 1.33×。切分后 `s`/`p`/`acc` 各减半：

| 项 | `128x512` 全量 | `NVEC=2` 后 |
|---|---|---|
| `q` fp16 `[128,128]` | 32 KB | 32 KB（cube 需全量）|
| `s` fp16 `[128,512]` | 128 KB | 64 KB |
| `p` fp16 `[128,512]` | 128 KB | 64 KB |
| `acc` fp32 `[128,128]` | 64 KB | 32 KB |
| **合计** | **352 KB** ✗ | **192 KB**（刚好）|

`al.extract_slice(ful, offsets, sizes, strides)` 的 `offsets` **支持运行时 tensor**，
且是零拷贝视图（只生成一条 `hivm.hir.extract_slice`）。

已实现（`FA_NVEC`，默认 1 = 不改变现有行为）：

```python
BLOCK_SUB_Q: tl.constexpr = BLOCK_Q // NVEC
row_lo = 0 if NVEC == 1 else (al.sub_vec_id() * BLOCK_SUB_Q).to(tl.int32)
...
s_full = tl.dot(q, tl.trans(k)).to(tl.float16)
s = al.extract_slice(s_full, (row_lo, 0), (BLOCK_SUB_Q, BLOCK_K), (1, 1))
```

若 `128x512 + NVEC=2` 能编过并提速，`BLOCK_Q` 64→128 会把 KV 重复读再减半，
预期 `aic_mte2` 占比从 30% 降到 15%。

---

## 10. 三条根本限制（不是调优没做好，是能力边界）

### 10.1 Triton 不支持张量切片 ⇒ Cube 与 Vector 的 tile 尺寸无法解耦

想实现「Cube 一次算 `[BQ, BLOCK_K]`，Vector 按 `CHUNK` 分片处理」，必须能从全量张量上切出子块。实测：

| 写法 | 结果 |
|---|---|
| `s[:, 0:CH]`（常量切片）| ❌ `unsupported tensor index: slice object` |
| `tl.static_range` 内 `s[:, jc:jc+CH]` | ❌ 同上 |
| `tl.split(s)` | ❌ `CompilationError`（要求最后一维恰为 2）|
| `tl.reshape(s,[BQ,BK//2,2])` + `tl.split` | ✅ 但只得到**偶/奇交错**的两半，且 `s_full` 仍整体存在 ⇒ **UB 一点不省** |
| `tl.reshape + tl.permute + tl.split`（想拿连续前后半）| ❌ 编译失败 |

**⇒ 想拿到 `[BQ, CHUNK]` 的 vector 输入，只能让 `dot` 就产出 `[BQ, CHUNK]` ⇒ Cube 也必须按 `CHUNK` 切 ⇒ `BLOCK_K` 失去减少同步的作用。**

**⇒ 唯一能解耦的介质是 GM，代价是 1.71×（`fa_fwd_manual.py` = 3.923 ms）。**

### 10.2 L0A/L0B（各 64 KB）封住 `CHUNK`

两个 dot 的操作数都要进 L0A/L0B：

| dot | L0B | 约束 |
|---|---|---|
| QK：`trans(k_c)` = `[D, CHUNK]` f16 | `CHUNK×128×2 ≤ 64 KB` | **CHUNK ≤ 256** |
| PV：`v_c` = `[CHUNK, D]` f16 | `CHUNK×128×2 ≤ 64 KB` | **CHUNK ≤ 256** |

**⇒ `CHUNK=512` 必然 `Cc`/`Cbuf` overflow（实测确认）。**

### 10.3 `PreLaunch` 在 A2 的 Triton 上没有可用对应物

见清单第 26 项。CATLASS 的 `preLaunch=2 + 3 槽 ring` 依赖 Ascend C 手工管理流水与同步；Triton 侧只有 `--enable-preload`（空转）和 `--set-cv-pipeline-mode=skew`（死锁/NaN）。

---

## 11. 三方 benchmark 结果（`bench/bench_all.py --preset realistic --modes causal`）

口径：CATLASS = `fat_bench` C++ 直调；Triton = `--graph` device-only；vendor = `npu_fusion_attention`（含 launcher）。

| B | N | H | CATLASS ms | Triton ms | vendor ms | Triton/vendor | Triton/CATLASS |
|---|---|---|---|---|---|---|---|
| 1 | 4096 | 8 | 0.3436 | 0.6054 | 0.6099 | **0.99×** | 1.76× |
| 1 | 4096 | 32 | 1.2929 | 2.3211 | 2.1736 | **1.07×** | 1.80× |
| 1 | 2048 | 32 | 0.4176 | 0.7552 | 0.5661 | 1.33× | 1.81× |
| 4 | 2048 | 8 | 0.4149 | 0.7542 | 0.5658 | 1.33× | 1.82× |
| 1 | 2048 | 8 | 0.1234 | 0.3015 | 0.1973 | 1.53× | 2.44× |
| 16 | 512 | 8 | 0.2318 | 0.4346 | 0.1685 | 2.58× | 1.88× |
| 1 | 512 | 8 | 0.0340 | 0.2929 | 0.1230 | **2.38×** | **8.61×** |

**读法：**
- **N ≥ 2048：Triton 稳定落后 CATLASS 1.76~1.82×**，这是 §10 那三条限制的直接结果
- **N = 4096 时 Triton 与 vendor 基本持平（0.99× / 1.07×）** —— 大 N 上我们的 kernel 效率已与 CANN 内置算子同档
- **N = 512 时崩到 2.38~2.58×** —— 守卫强制 `(64,256)`（避开 507015）⇒ 迭代翻倍；且任务数少（`B=1,H=8` 只有 64 个任务是 3.2 波）
- **注意口径对 Triton 有利**（Triton 是纯 device，vendor 含 launcher）。用 CATLASS 的 `app` 列（含 launcher）比 Triton 的 `device` 列，仍是 1.74~7.06× ⇒ **1.8× 的差距不是计时口径造成的**

---

## 12. 后续可做的两件事（都在 `_pick_tile` 层面，低风险）

| # | 改动 | 预期 |
|---|---|---|
| 1 | **修掉 507015**：把外层 `k/v` 的 `tl.advance` 换成每轮 `tl.make_block_ptr(offsets=(j,0))` | 让 `(64,512)` 在 `N_k == BLOCK_K` 时可用 ⇒ `N=512` 迭代减半 ⇒ ~25% |
| 2 | `N ≤ 1024` 时用 `BLOCK_Q=32` | 任务数翻倍 ⇒ 波次利用率 80%→91% |
