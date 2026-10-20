# W1 受控实验（无 NPU 版）

> 目的：用 `00_basic_matmul` 这一份代码，把“模板参数不是填空题，而是硬件预算”这件事变成手感。
> **受控** = 一次只改一个变量、其余原样、先手算预期、再让编译器验证。

## 一、为什么没有 NPU 也能做

`examples/CMakeLists.txt:20` 里有一句关键配置：

```cmake
add_compile_options($<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-${CATLASS_ARCH}>)
```

意思是：编译 ASC 代码时是**交叉编译**到 `dav-2201`（Atlas A2），只需要 CANN 里的 bisheng，**不需要插着卡**。
更关键的是：`L1TileShape` / `L0TileShape` 是**编译期常量**，超限由 `static_assert` 拦截（`block_mmad_pingpong.hpp:122-130`），这是纯粹的编译期行为。

所以四个实验里有 **两个天然是编译期实验**，另外两个只能做“降级版”。

| 实验 | 改什么 | 无 NPU 能验证什么 | 需要真机才能验证什么 |
| :-- | :-- | :-- | :-- |
| 1 | `L1TileShape` K 256→512 | ✅ 完整：编译失败 + 报错原文 | —— |
| 2 | `L0TileShape` K 64→128 | ✅ 完整：编译失败 + 报错原文 | —— |
| 2b（附加） | L1=L0=256×256×64 | ✅ 完整：触发 **L0C** 那条断言 | —— |
| 3 | `DispatchPolicy` Pingpong→Preload | ⚠️ 只能验“能编过”+ 看类型差异 | 流水气泡、耗时 |
| 4 | `LayoutB` RowMajor→ColumnMajor | ⚠️ 只能验“能编过”+ 看 TileCopy 特化 | 搬运效率、精度（`Compare success`） |
| 5（附加） | `LayoutC` RowMajor→ColumnMajor | ✅ 完整：`"LayoutC only support RowMajor yet!"` | —— |

**降级版的验收标准**：不是“跑得快”，而是“**我能说清它换成了哪套特化，以及为什么这里不该有性能收益/损失**”。这比跑一遍更有价值。

## 二、先手算，再编译（这一步别省）

基线（fp16 输入、fp32 累加、`STAGES = 2`）：

```
L1 : (M_l1*K*2 + N_l1*K*2) * 2 ≤ 512KB   # A 和 B 各 ping-pong 两份
L0A: M_l0*K_l0*2 * 2          ≤ 64KB
L0B: K_l0*N_l0*2 * 2          ≤ 64KB
L0C: M_l0*N_l0*4              ≤ 128KB    # 注意：与 K 无关
```

基线 `(128,256,256)` / `(128,256,64)` 代进去：

```
L1 : (65536 + 131072) * 2 = 393216 B  ≤ 524288  ✓
L0A: 128*64*2*2  = 32768 B            ≤ 65536   ✓
L0B: 64*256*2*2  = 65536 B            ≤ 65536   ✓（正好顶格）
L0C: 128*256*4   = 131072 B           ≤ 131072  ✓（正好顶格）
```

**两个“正好顶格”** 值得记住：这就是为什么基线的 L0 分块不能再往上加，也解释了实验 2 会先撞 L0B。

## 三、怎么跑

```bash
# 一条命令 = 改一处 + 只编这一个 target + 抓断言 + 还原
bash catlass_lab/w1/run.sh 1
bash catlass_lab/w1/run.sh 2
bash catlass_lab/w1/run.sh 2b
bash catlass_lab/w1/run.sh 3
bash catlass_lab/w1/run.sh 4
bash catlass_lab/w1/run.sh 5

# 想保留现场（在 Zed 里 hover 看类型）：
KEEP=1 bash catlass_lab/w1/run.sh 3
# 还原：
git -C ~/codes/opensource-project/catlass checkout -- examples/00_basic_matmul/basic_matmul.cpp
```

首次运行会在 `~/codes/opensource-project/catlass/build` 重新 configure（1~2 分钟），之后增量 10~20 秒。
完整日志落在 `catlass_lab/w1/build_exp.log`。

> 注意：脚本开头会 `git checkout` 一次该文件，保证每次从原始状态出发——
> 如果你手改过 `basic_matmul.cpp` 想保留，先自己备份。

## 四、预期结果（先别看，先手算）

| exp | 预期报错串 | 触发位置 |
| :-- | :-- | :-- |
| 1 | `L1TileShape exceeding the L1 space!` | `block_mmad_pingpong.hpp:122` |
| 2 | `L0TileShape exceeding the L0B space!` | `:129` |
| 2b | `L0TileShape exceeding the L0C space!` | `:130` |
| 3 | （无报错） | —— |
| 4 | （无报错） | —— |
| 5 | `LayoutC only support RowMajor yet!` | `:119` |

## 五、实验 3 / 4 在无 NPU 下怎么“看得见”

**手法：用编译错误当 `printf`。** 临时在 `basic_matmul.cpp` 里加两行，让编译器把解析后的类型打印出来：

```cpp
template <class T> struct ShowType;                  // 故意只声明不定义
ShowType<typename BlockMmad::CopyGmToL1B> probe;     // 报错里会打出完整的实例化类型
```

- exp4 做完，你会看到 `CopyGmToL1B` 从 `RowMajor` 那套变成 `ColumnMajor` 那套——这就是“换 layout 换了搬运路径”的实证；
- exp3 做同样的事看 `BlockMmad` 的成员差异，再打开 `block_mmad_preload.hpp:28` 对比它的特化写法。

顺便：这两行代码在 Zed 里也能直接用 **hover** 看（前提是上一轮的 clangd 配置在），比编译更快。

## 六、每个实验要记下来的东西（写进 LOG）

只记三样，别写作文：

1. **手算值**（编译前写）；
2. **编译器原话**（第一条 `static assertion failed` 的断言字符串 + 触发文件:行）；
3. **一句话结论**（例：L0C 只吃 M×N，和 K 无关，所以大 K 场景可以放心加 K）。

模板见 `LOG.md`。
