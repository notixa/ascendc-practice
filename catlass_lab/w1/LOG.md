# W1 实验记录

> 规矩：先手算 → 再编译 → 只记三样（手算值 / 编译器原话 / 一句话结论）。
> 手算公式见 `README.md` 第二节。

| exp | 改动 | 手算（字节） | 实际结果（报错串 / 文件:行 或 “编译通过”） | 一句话结论 |
| :-- | :-- | :-- | :-- | :-- |
| 1 | L1 K: 256→512 | L1 = ____ B vs 524288 | | |
| 2 | L0 K: 64→128 | L0A = ____ / L0B = ____ | | |
| 2b | L1=L0=256×256×64 | L0C = ____ vs 131072 | | |
| 3 | 只换 policy | —— | | |
| 3b | policy + kernel 一起换 | —— | | |
| 4 | LayoutB → ColumnMajor | —— | | |
| 5 | LayoutC → ColumnMajor | —— | | |

## 备注区（每次踩到的坑、看错的数字、想不明白的地方）

- 
- 

- exp3 结论：只换 policy 会 `no matching function`（Preload 版 operator() 要 12 个参数）；policy 与 kernel 必须成对换（exp3b），参照 examples/21_basic_matmul_preload_zN。
