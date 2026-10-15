# FlashAttention on Ascend NPU (Atlas 200I DK A2) - 评测与优化日志

本文档用于持续跟踪与记录自定义 **Ascend C FlashAttention 算子** 在华为昇腾 **Atlas 200I DK A2** 边缘开发板上的每一次迭代测试、精度标定、性能跑分与调优记录。

---

## 硬件与软件环境规格

| 项目 | 规格参数 |
| :--- | :--- |
| **测试开发板** | 华为 Atlas 200I DK A2 |
| **NPU 芯片架构** | 昇腾 Ascend 310B4 (`__CCE_AICORE__ == 200`, `dav-m200`)，统一 AI Core 架构 |
| **CPU 架构** | 4-Core AArch64 (ARMv8.2) |
| **内存配置** | 3.5 GB LPDDR4X 共享内存 + Swap |
| **操作系统** | Ubuntu 22.04 LTS (aarch64) |
| **CANN 驱动与工具链** | CANN 8.0.RC3 (`/usr/local/Ascend/ascend-toolkit/latest`) |
| **深度学习框架** | Python 3.9 + PyTorch 2.1.0 + torch_npu 2.1.0.post8 |

---

## 测试记录索引

- [Record 2026-10-16 #02: 多规模权威基准横向评测与瓶颈诊断](#record-2026-10-16-02-多规模权威基准横向评测与瓶颈诊断)
- [Record 2026-10-16 #01: 流水线冲突消除、dK 修复与多 Tile 精度闭环](#record-2026-10-16-01-流水线冲突消除-dk-修复与多-tile-精度闭环)

---

## Record 2026-10-16 #02: 多规模权威基准横向评测与瓶颈诊断

### 1. 测试背景与目标
- 目标：建立自动化基准套件（`benchmark.py`），对比业界权威实现，摸清当前算子在不同序列长度下的性能表现。
- 测试脚本：`benchmark.py`（10 轮 Warmup，30 轮计时取中位数）。
- 对照基准：
  1. 华为官方生产级算子：`torch_npu.npu_fusion_attention`（底层为 `aclnnFlashAttentionScore`）。
  2. 框架标准实现：PyTorch 原生 NPU 注意力（FP16，Matmul + Softmax + Autograd）。

### 2. 关键发现（硬件事实）
- **官方算子缺失**：调用官方 `torch_npu.npu_fusion_attention` 时，板端抛出错误：
  ```text
  EZ9999: Op FlashAttentionScore does not has any binary.
  launch failed for FlashAttentionScore, errno:561000.
  ```
  **结论**：华为官方 CANN 仅在 910 系列数据中心卡提供 FlashAttention 二进制，**并未给 310B 边缘芯片适配 FlashAttention 官方算子**。本工程实现是该芯片上极少数真正可用的端到端 FlashAttention。基线自动对齐为 PyTorch 原生标准实现。

### 3. 实测数据矩阵

| 形状 $(N_q, N_k, d)$ | Forward 耗时 (Our / Base) | Fwd 加速比 | Backward 耗时 (Our / Base) | Bwd 加速比 | 吞吐表现 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$(64, 64, 64)$** | **0.27 ms** / 0.38 ms | **1.41x** 🚀 | **0.45 ms** / 1.38 ms | **3.03x** 🚀 | 反向提速超 3 倍 |
| **$(64, 128, 64)$** | **0.33 ms** / 0.37 ms | **1.14x** 🚀 | **0.60 ms** / 1.70 ms | **2.83x** 🚀 | 反向提速近 3 倍 |
| **$(128, 64, 64)$** | **0.35 ms** / 0.38 ms | **1.08x** 🚀 | **0.63 ms** / 1.37 ms | **2.20x** 🚀 | 反向提速超 2 倍 |
| **$(128, 128, 64)$** | 0.51 ms / **0.37 ms** | 0.72x | **0.81 ms** / 1.41 ms | **1.74x** 🚀 | 反向持续领先 |
| **$(256, 256, 64)$** | 1.47 ms / **0.41 ms** | 0.28x | 1.98 ms / **1.41 ms** | 0.72x | 进入长序列多 Tile 区间 |
| **$(512, 512, 64)$** | 5.27 ms / **0.63 ms** | 0.12x | 6.52 ms / **1.49 ms** | 0.23x | 访存与循环开销主导 |

### 4. 性能特征剖析与下阶段优化计划
1. **中短序列优势明显**：在 $N \le 128$ 区间，反向因为重计算与片上吸收避免了 $S, P$ 中间矩阵频繁换入换出 Global Memory，速度达到原生的 **1.7x ~ 3.0x**。
2. **长序列瓶颈定位**：
   - **单核限制**：当前 `targetCoreNum = 1`，整张算子运行在单个 AI Core 上，未利用 310B 的多核心算力。
   - **缺少双缓冲（Double Buffering）**：外层 Tile 循环迭代时，MTE2 搬运数据与 Vector 计算串行等待。
   - **粗粒度同步**：内层多处使用 `PipeBarrier<PIPE_ALL>()`，存在微秒级流水线停顿累积。
3. **下阶段目标**：
   - 目标 1：开放多核切分调度（`targetCoreNum > 1`），让多个 Block 并行分担 $T_q$。
   - 目标 2：在搬运流水线中引入 Double Buffering，实现 MTE2 与 Vector 并发重叠。

---

## Record 2026-10-16 #01: 流水线冲突消除、dK 修复与多 Tile 精度闭环

### 1. 修改要点与问题修复
1. **硬件 Fixpipe 槽位冲突崩溃（`subErrType: 4`）**：
   - 根因：310B 上只有一个物理 Cube/Fixpipe 单元。在 `bwd_dkdv` 中定义了 4 个 `Matmul` 实例，并在内层循环调用 `Init()/End()`，触发硬件上下文槽位耗尽异常。
   - 修复：删除第 4 个 Matmul 实例，重用 `mmPdO_` 计算 $dS^T @ Q \rightarrow dK$；将 `Init()/End()` 移出循环，固定在 Kernel 生命周期两头。
2. **$dK$ 尾部行数据错乱修复（45~63 行脏读）**：
   - 根因：写回 GM 前使用了 `PipeBarrier<PIPE_V>()`，但搬出指令由 `PIPE_MTE3` 执行。`PIPE_V` 无法约束 MTE3，导致 MTE3 在 Vector 还没写完时提前抢跑。
   - 修复：写回 GM 前替换为 `PipeBarrier<PIPE_ALL>()`。
3. **多 Tile 跨流水线屏障补齐**：
   - 在 `flash_attention_fwd_kernel.cpp` 与 `flash_attention_bwd_dq_kernel.cpp` 的 `GetTensorC` 与下一次迭代之间插入 `PipeBarrier<PIPE_ALL>()`，彻底消除 $N_k=128$ 时的硬件挂起。

### 2. 精度验证矩阵（与 PyTorch CPU FP32 Golden 对比）
- 容差标准：$atol \le 0.02, rtol \le 0.02$。

| 形状 $(N_q, N_k, d)$ | 切分状态 | Forward 最大误差 ($O$ / $L$) | Backward 独立误差 ($dQ / dK / dV$) | Backward 链式误差 ($dQ / dK / dV$) | 坏行数 | 判定 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$(64, 64, 64)$** | $T_q=1, T_k=1$ | $0.00113$ / $0.00319$ | $0.00185$ / $0.00183$ / $0.00138$ | $0.00185$ / $0.00223$ / $0.00181$ | 0 / 64 | **PASS** |
| **$(64, 128, 64)$** | $T_q=1, T_k=2$ | $0.00079$ / $0.00335$ | $0.00204$ / $0.00227$ / $0.00084$ | $0.00215$ / $0.00227$ / $0.00184$ | 0 / 128 | **PASS** |
| **$(128, 64, 64)$** | $T_q=2, T_k=1$ | $0.00110$ / $0.00290$ | $0.00176$ / $0.00240$ / $0.00185$ | $0.00202$ / $0.00257$ / $0.00294$ | 0 / 64 | **PASS** |
| **$(128, 128, 64)$** | $T_q=2, T_k=2$ | $0.00161$ / $0.00369$ | $0.00267$ / $0.00299$ / $0.00185$ | $0.00210$ / $0.00201$ / $0.00210$ | 0 / 128 | **PASS** |

**结果**：所有形状最大误差控制在 $\le 0.00369$，全用例通过！
