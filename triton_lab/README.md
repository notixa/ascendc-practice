# Triton-Ascend 练习区

> 目标：把 CS336 assignment2 的 FlashAttention（fwd + bwd）迁到昇腾，并和 `catlass_lab` 的 C++ 版做对比。
> 与 `catlass_lab` 的分工：那边是 C++ 模板（CATLASS），这边是 Python DSL（Triton-Ascend），**同一个 FA 写两遍**。

---

## 0. 先回答两个硬件问题

**910B 支持 triton-ascend。** 依据（都是本机能查到的）：
- `~/.cannbot/cann-bench/docker/dev/Dockerfile:14-20`：`DEVICE` 合法取值是 `910 / 910b / 950 / a3 / 310p`，示例里就是 `--build-arg CANN_VERSION=9.1.0 --build-arg DEVICE=910b`；
- `~/.cannbot/cann-bench/docs/guide/triton_ascend_quick_start.md`：明确写了“910B 环境把 `DEVICE=950` 改成 `DEVICE=910b`”；
- triton-ascend 官方索引 `https://triton-ascend.osinfra.cn/pypi/simple/triton-ascend/` 上，cp310~cp313 都有 `x86_64` 和 `aarch64` 的 wheel。

**但 200I（310B）不在支持列表里** —— 列表里是 `310p`，没有 `310b`。所以：
> 和 CATLASS 一样，Triton 线的“跑起来”也要等 910B。等卡期间这边只能做**改写 + 静态审查 + 环境体检**。

**版本配对（重要）**：本地文档与镜像配套的是 **CANN 9.0.1 / 9.1.0 ↔ triton-ascend 3.2.x / 3.6.0**。
本机 CANN 是 **8.5.0**，大概率不匹配——到 910B 机器上先 `cat /usr/local/Ascend/cann/*/ascend_toolkit_install.info` 确认 CANN 版本，再决定装 3.2.2 还是 3.6.0。

---

## 1. 环境

```bash
cd triton_lab
bash scripts/setup_ascend.sh     # ★ 用这个，等价于下面两步
#   uv sync --extra dev
#   uv pip install --no-deps --force-reinstall --index-url <官方索引> triton-ascend==3.6.0

uv run python env_report.py      # 体检：triton 版本 / 是否有 ascend 后端 / target / CANN
uv run pytest                    # 有卡跑 vector add，无卡自动 skip
```

### 1.1 已知坑：`uv sync` 之后 `import triton` 会挂（必读）

```
ModuleNotFoundError: No module named 'triton._C.libtriton.ascend';
'triton._C.libtriton' is not a package
```

**原因**：`triton` 和 `triton-ascend` 两个包**都会往 `triton/_C/` 写 `libtriton.so`**。
上游 triton 的是 415MB 的通用版，triton-ascend 的是 180MB 的**带 ascend dialect** 的版本。
`uv sync` 里上游通常后装，把 ascend 版覆盖掉 → ascend 后端 `import` 就崩。

**解法**：`uv sync` 之后再强制装一次 triton-ascend，让它覆盖回去（`scripts/setup_ascend.sh` 的第 2 步）。
**注意**：以后每次重跑 `uv sync`，这一步都要重做（因为 uv 会重新拉上游 triton）。
判断有没有修好：`ls -l .venv/lib/python3.10/site-packages/triton/_C/libtriton.so` 应该是 **~180MB**；415MB 就是被覆盖了。

### 1.2 本机实测状态（2026-10-20，x86 无 NPU）

| 项 | 结果 |
| :-- | :-- |
| triton | 3.6.0（ascend 版） |
| backends | `amd` / `ascend` / `nvidia` → **ascend 在列** ✅ |
| current target | 取不到（因为没装 torch_npu，正常） |
| `pytest` | `1 passed, 1 skipped`（无卡那项自动 skip）✅ |
| CANN | 本机 8.5.0；**triton-ascend 3.2.x/3.6.0 对应 CANN 9.x，到 910B 机器上要确认/升级** |

### 1.3 torch / torch_npu 还没装

triton-ascend 不强制带 torch。到 910B 机器上补：

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu   # 先 CPU 版
uv pip install torch_npu==<与 CANN 配套的版本>                          # 再 NPU 插件
# 配套表：https://gitcode.com/Ascend/pytorch/blob/master/COMPATIBILITY.md
```

> 注意：`triton-ascend` 的 wheel 自带顶层 `triton` 包，会和上游 triton（你 `.venv` 里那个 3.7.0）抢名字，
> 所以这个项目**必须独立 venv**，别和 catlass 那边混。

## 2. 目录

```
triton_lab/
├── pyproject.toml        # uv 工程 + triton-ascend 索引配置
├── .python-version       # 3.10
├── env_report.py         # 环境体检（第一步就跑它）
├── kernels/
│   ├── vector_add.py     # 最小可跑 kernel，用来验证“工具链通了”
│   └── fa_fwd.py         # 你的 FA（从 CS336 改写），见第 3 节
└── tests/
    └── test_smoke.py     # 无 NPU 自动 skip
```

## 3. 从 CS336 迁过来的 6 条硬改（详细版见 `../CATLASS_AND_TRITON_ASCEND_LEARNING_ROADMAP.md` 0.3）

1. 删 `num_warps` / `num_stages` / `num_ctas`（GPU 独有，NPU 不生效）；
2. 并行度别用核数当 kernel 参数，核数用 `driver.active.utils.get_device_properties()` 读；
3. `BLOCK_M/N/K` 按 UB（A2 192KB）重算，别照抄 GPU 的 128×128；
4. `tl.dot` 的 `trans_*` 参数在 Ascend 上受限 → 改显式 `tl.trans`；
5. **bwd 的 dQ 原子累加 → 改确定性累加**（split 到 workspace 再 reduce，和 CATLASS 的 SplitK 同思路）；
6. 保持两版可 diff：先只做“删参数 + 改 BLOCK”，**不要顺手改算法**，否则定位不了“是环境还是代码”。

## 4. 迁移顺序（别跳）

```
vector add 跑通  →  FA forward（单 head、causal、无 paged cache）  →  对上 torch golden
                 →  FA backward（先 non-atomic 版本，慢没关系）    →  对齐数值
                 →  再谈 autotune / BLOCK 搜索 / msprof
```
