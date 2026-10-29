"""
bench_fa.py —— CATLASS FA (catlass_op) vs Triton-Ascend FA (triton_lab) 的性能对比

设计原则（perf-loop）：
  1. 先过正确性闸门，再谈性能 —— 算错的 kernel 跑得快没有意义
  2. 同一份输入、同一 dtype、同一 layout 归一化方式
  3. warmup + 多次重复取中位数，并打印波动范围（波动 > 10% 说明测不准）
  4. 只改一个变量：backend 是唯一被比较的变量，其余全部相同

两种语义都测，分开报告：
  - full   : CATLASS 传【全 0 mask】（等价于不屏蔽）、Triton causal=False
  - causal : 两边都 causal（Triton 需先实现 causal，见 README）

用法（在 910B 机器上）：
    python3 bench/bench_fa.py
    python3 bench/bench_fa.py --only catlass
    python3 bench/bench_fa.py --skip-correctness
    python3 bench/bench_fa.py --peak-tflops 320
"""

import argparse
import math
import os
import statistics
import sys

import torch
import torch_npu

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

#   (batch, heads, N, head_dim)   —— CATLASS 要求 head_dim=128 且 N % 128 == 0
CASES = [
    (1, 8, 512, 128),
    (1, 8, 1024, 128),
    (1, 8, 2048, 128),
    (1, 8, 4096, 128),
    (4, 8, 2048, 128),
]

WARMUP = 10
ITERS = 50
REPEATS = 5

_CATLASS_MASK = {}


def catlass_mask(device, causal):
    """causal=True → 严格上三角（1=屏蔽）；False → 全 0（不屏蔽 = full attention）"""
    key = (str(device), causal)
    if key not in _CATLASS_MASK:
        base = torch.ones(128, 128, dtype=torch.float16)
        m = torch.triu(base, diagonal=1) if causal else torch.zeros_like(base)
        _CATLASS_MASK[key] = m.to(device)
    return _CATLASS_MASK[key]


# ---------------------------------------------------------------------------
# 后端封装：统一接收 [B, N, H, D]（fp16, contiguous），返回同样 shape
# ---------------------------------------------------------------------------
def make_catlass(so_path):
    torch.ops.load_library(so_path)

    def run(q_bnhd, k_bnhd, v_bnhd, causal, block_dim=0):
        return torch.ops.catlass.fat(
            q_bnhd,
            k_bnhd,
            v_bnhd,
            catlass_mask(q_bnhd.device, causal),
            1.0 / math.sqrt(q_bnhd.size(-1)),
            0,
            block_dim,
        )

    return run


def make_vendor():
    """torch_npu 的厂商融合 FA —— 产品级参照。

    这是判断「30 TFLOPS 算好还是算差」的唯一参照：
      - 若厂商也在 ~30 TFLOPS  → 30 就是平台现实，我们的 kernel 已达产品级
      - 若厂商 100+ TFLOPS     → triton-ascend 有 3x 编译器损失，转 CATLASS 方向正确

    ⚠ 接口在不同 torch_npu 版本上有差异（layout 名 / atten_mask 形状 / sparse_mode 语义），
    所以这里做了一个小阶梯：按顺序试，第一个不抛异常的就用它。数值对不对由闸门负责。
    """
    fn = getattr(torch_npu, "npu_fusion_attention", None)
    if fn is None:
        raise RuntimeError("torch_npu 里没有 npu_fusion_attention，无法做厂商对照")

    _mask = {}
    _picked = {}

    def _get_mask(n, device):
        key = (n, str(device))
        if key not in _mask:
            # atten_mask 语义：True = 被屏蔽（不参与计算）
            # causal → 屏蔽上三角（future），对角线保留（自己看自己）
            _mask[key] = torch.triu(
                torch.ones((n, n), dtype=torch.bool, device=device), diagonal=1
            )
        return _mask[key]

    def _call(q, k, v, causal, variant):
        B, N, H, D = q.shape
        kw = dict(head_num=H, input_layout="BSND", scale=1.0 / math.sqrt(D))
        if causal:  # ★ full 模式【不能】传 mask
            if variant["mask"]:
                n = variant["mask_n"] if variant["mask_n"] else N
                kw["atten_mask"] = _get_mask(n, q.device)
            if variant["sparse"] is not None:
                kw["sparse_mode"] = variant["sparse"]
        out = fn(q, k, v, **kw)
        return out[0] if isinstance(out, (tuple, list)) else out

    CAUSAL_VARIANTS = [
        dict(mask=True, mask_n=None, sparse=0),  # 显式 [N,N] mask（实测你的版本可用）
        dict(mask=True, mask_n=2048, sparse=0),  # 显式 [2048,2048] mask
        dict(mask=False, mask_n=None, sparse=2),  # 只给 sparse_mode
    ]
    FULL_VARIANTS = [
        dict(mask=False, mask_n=None, sparse=None),  # 不传任何 mask（= 全通）
        dict(mask=False, mask_n=None, sparse=0),  # 显式声明 default 模式
    ]

    def run(q_bnhd, k_bnhd, v_bnhd, causal, block_dim=0):
        key = bool(causal)
        if key in _picked:
            return _call(q_bnhd, k_bnhd, v_bnhd, causal, _picked[key])

        last = None
        for var in CAUSAL_VARIANTS if causal else FULL_VARIANTS:
            try:
                out = _call(q_bnhd, k_bnhd, v_bnhd, causal, var)
                torch.npu.synchronize()
                _picked[key] = var
                print(f"  [vendor] causal={causal} 采用变体: {var}")
                return out
            except Exception as e:  # noqa: BLE001
                last = e
        raise RuntimeError(f"npu_fusion_attention 所有变体都失败，最后一个错误: {last}")

    return run


def make_triton():
    sys.path.insert(0, os.path.join(ROOT, "triton_lab"))
    from kernels.flash_attention.fa_fwd import fla_fwd  # noqa: E402

    def run(q_bnhd, k_bnhd, v_bnhd, causal, block_dim=0):
        # ⚠ triton_lab 的 fla_fwd 用的是 token-major [B, N, H, D]
        #   （N_q = Q.size(1)、head = Q.size(2)），和 CATLASS 那边一致。
        #   不要 permute 成 [B,H,N,D] —— 那样 N_q/head 会互换，结果全错。
        try:
            o, _lse = fla_fwd(q_bnhd, k_bnhd, v_bnhd, causal=causal)
        except TypeError:
            if causal:
                raise RuntimeError(
                    "fla_fwd 还不支持 causal 参数 —— 先在 triton_lab 里实现（见 bench/README.md）"
                )
            o, _lse = fla_fwd(q_bnhd, k_bnhd, v_bnhd)
        return o

    return run


# ---------------------------------------------------------------------------
# golden / 计时 / roofline
# ---------------------------------------------------------------------------
def golden(q, k, v, causal):
    qf, kf, vf = q.float(), k.float(), v.float()
    s = torch.einsum("bqhd,bkhd->bhqk", qf, kf) / math.sqrt(q.size(-1))
    if causal:
        n = q.size(1)
        m = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        s = s.masked_fill(m, float("-inf"))
    p = torch.softmax(s, dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", p, vf)


def timeit(fn, warmup=WARMUP, iters=ITERS, repeats=REPEATS):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()

    samples = []
    for _ in range(repeats):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.npu.synchronize()
        samples.append(start.elapsed_time(end) / iters)  # ms / iter

    med = statistics.median(samples)
    lo, hi = min(samples), max(samples)
    spread = (hi - lo) / med if med > 0 else float("inf")
    return med, lo, hi, spread


def flops_of(batch, heads, n, d, causal):
    # S = Q K^T : 2*N*N*D ;  O = P V : 2*N*N*D  （每个 head）
    f = 4.0 * batch * heads * n * n * d
    return f / 2.0 if causal else f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["catlass", "triton", "vendor"], default=None)
    ap.add_argument(
        "--graph",
        action="store_true",
        help="对 triton 后端启用图捕获（消除 Python launcher 开销；"
        "只在 device < 0.42ms 时有收益，N<=1024）",
    )
    ap.add_argument(
        "--graph-copy",
        action="store_true",
        help="图捕获时把输入 copy_ 也计入（默认不计：纯 replay = device-only 时间）",
    )
    ap.add_argument("--mode", choices=["full", "causal", "both"], default="both")
    ap.add_argument("--skip-correctness", action="store_true")
    ap.add_argument(
        "--peak-tflops",
        type=float,
        default=None,
        help="datasheet 峰值 fp16 TFLOPS，填了才输出 %%peak",
    )
    ap.add_argument(
        "--so",
        default=os.path.join(ROOT, "catlass_lab/catlass_op/build/libcatlass_fat.so"),
    )
    ap.add_argument("--atol", type=float, default=2e-2)
    ap.add_argument("--cases", default=None,
                    help="覆盖内置 CASES，格式 N:H[,N:H...]（B=1, D=128）")
    ap.add_argument("--json", default=None,
                    help="把逐 case 结果写成 JSON（供 bench_all.py 汇总）")
    args = ap.parse_args()

    cases = CASES
    if args.cases:
        cases = [(1, int(h), int(n), 128) for n, h in (c.split(":") for c in args.cases.split(","))]
    json_rows = []
    backends = {}
    if args.only in (None, "catlass"):
        backends["catlass"] = make_catlass(args.so)
    if args.only in (None, "triton"):
        backends["triton"] = make_triton()
    if args.only in (None, "vendor"):
        try:
            backends["vendor"] = make_vendor()
        except Exception as e:  # noqa: BLE001
            print(f"跳过 vendor 后端: {e}")

    modes = [False, True] if args.mode == "both" else [args.mode == "causal"]

    print(f"device: {torch.npu.get_device_name(0)}")
    print(f"warmup={WARMUP} iters={ITERS} repeats={REPEATS}（每轮取中位数）\n")

    rows = []
    for batch, heads, n, d in cases:
        if d != 128:
            print(
                f"skip B={batch} H={heads} N={n} D={d}（CATLASS 只支持 head_dim=128）"
            )
            continue

        torch.manual_seed(0)
        q = torch.randn(batch, n, heads, d, dtype=torch.float16).npu()
        k = torch.randn(batch, n, heads, d, dtype=torch.float16).npu()
        v = torch.randn(batch, n, heads, d, dtype=torch.float16).npu()

        print(f"── B={batch} H={heads} N={n} D={d} ──")

        for causal in modes:
            tag = "causal" if causal else "full  "
            ref = (
                None
                if args.skip_correctness
                else golden(q.cpu(), k.cpu(), v.cpu(), causal)
            )

            for name, fn in backends.items():
                tag_g = ""
                gf = None
                if args.graph and name == "triton":
                    sys.path.insert(0, HERE)
                    from graph_fa import GraphedFLA  # noqa: PLC0415

                    gf = GraphedFLA(batch, n, heads, d, causal=causal)
                    if args.graph_copy:
                        call = lambda: gf(q, k, v)  # 图 + 输入拷贝（真实口径）
                        tag_g = " [图+拷贝]"
                    else:
                        call = gf.replay_only  # 纯 replay（device-only）
                        tag_g = " [图 replay-only]"
                else:
                    call = lambda: fn(q, k, v, causal)

                # ---- 闸门 ----
                if ref is not None:
                    try:
                        o = gf(q, k, v) if gf is not None else fn(q, k, v, causal)
                        torch.npu.synchronize()
                    except Exception as e:  # noqa: BLE001
                        print(f"  [{tag}] {name:<8} 跑不通: {e}")
                        continue
                    err = (o.cpu().float() - ref).abs().max().item()
                    ok = err < args.atol
                    print(
                        f"  [{tag}] {'OK ' if ok else 'FAIL'} {name:<8} max_err={err:.5f}"
                    )
                    if not ok:
                        print(f"           → 跳过计时（算错的数字没有意义）")
                        continue

                # ---- 计时 ----
                try:
                    med, lo, hi, spread = timeit(call)
                except Exception as e:  # noqa: BLE001
                    print(f"  [{tag}] {name:<8} 计时失败: {e}")
                    continue
                f = flops_of(batch, heads, n, d, causal)
                tflops = f / (med * 1e-3) / 1e12
                peak = (
                    f"  {100.0 * tflops / args.peak_tflops:5.1f}% peak"
                    if args.peak_tflops
                    else ""
                )
                warn = "  ⚠ 波动大" if spread > 0.10 else ""
                print(
                    f"  [{tag}]     {name:<8}{tag_g} {med:8.3f} ms  [{lo:.3f}, {hi:.3f}]  "
                    f"{tflops:7.1f} TFLOPS{peak}{warn}"
                )
                rows.append((name, causal, batch, heads, n, med, tflops))
                json_rows.append({"name": name, "causal": int(bool(causal)), "batch": batch,
                                  "heads": heads, "n": n, "d": 128, "ms": med,
                                  "tflops": tflops})

                # 顺便把 device-only 时间也打出来，方便和上面的数对比
                if gf is not None:
                    med_r, lo_r, hi_r, _ = timeit(gf.replay_only)
                    json_rows.append({"name": name + "-device", "causal": int(bool(causal)),
                                      "batch": batch, "heads": heads, "n": n, "d": 128,
                                      "ms": med_r, "tflops": float("nan")})
                    print(
                        f"  [{tag}]       └─ device-only (纯 replay) {med_r:8.3f} ms  "
                        f"[{lo_r:.3f}, {hi_r:.3f}]"
                    )
                    del gf  # 及时释放图内存池
        print()

    # ---- 汇总 ----
    if rows:
        by_case = {}
        for name, causal, batch, heads, n, med, tflops in rows:
            by_case.setdefault((causal, batch, heads, n), {})[name] = med
        print("=" * 74)
        print(
            f"{'mode':<8}{'case':<20}{'catlass':>11}{'triton':>11}{'vendor':>11}"
            f"{'triton/vendor':>15}"
        )
        for (causal, batch, heads, n), d_ in sorted(
            by_case.items(), key=lambda x: (x[0][0], x[0][3])
        ):
            c = d_.get("catlass", float("nan"))
            t = d_.get("triton", float("nan"))
            vd = d_.get("vendor", float("nan"))
            ratio = t / vd if vd == vd and vd > 0 else float("nan")
            mode = "causal" if causal else "full"
            print(
                f"{mode:<8}{f'B{batch} H{heads} N{n}':<20}{c:>11.3f}{t:>11.3f}{vd:>11.3f}"
                f"{ratio:>15.2f}x"
            )
        print("=" * 74)
        print("triton/vendor > 1 表示我们的 Triton kernel 比厂商慢，< 1 表示更快")
        if args.graph:
            print("⚠ 本次启用了图捕获：数字里已没有 Triton 的 Python launcher 开销，")
            print("  不可与未开图的结果、或与 CATLASS（C++ 直调）横向比较。")
        print(
            "注意：causal 行两边 FLOPs 相同，但绝对 FLOPs 只有 full 的一半，不要跨行比较 TFLOPS"
        )

    if args.json:
        import json as _json
        with open(args.json, "w") as f:
            _json.dump(json_rows, f, indent=1)
        print(f"[json] 已写入 {args.json}（{len(json_rows)} 条）")


if __name__ == "__main__":
    main()
