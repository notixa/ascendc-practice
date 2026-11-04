"""
graph_fa.py —— 用图捕获消除 FA 后端的 launcher 开销（Triton / CATLASS 通用）

背景（实测）：
    CPU 提交 / 次 = 0.426 ms
    端到端   / 次 = 0.427 ms     ← 两者相等 ⇒ 纯 CPU 提交瓶颈
    device 实际只跑 ~0.06 ms（N=512）

图捕获把这些 CPU 开销一次性录进一张图，之后只 replay()：
    - Triton 的 50 个参数绑定     → 不再执行
    - torch.zeros_like / zeros    → 录进图私有内存池
    - get_device_properties(0)    → 只在捕获时查一次
    - Python 侧所有算术            → 不再执行

⚠️ 它只省 CPU，不缩短 device 执行时间。

用法：
    python3 bench/graph_fa.py
"""

import math
import os
import sys

import torch
import torch_npu

HERE = os.path.dirname(os.path.abspath(__file__))


class GraphedFA:
    """把任意 FA 后端的调用录成一张图，之后只 replay。

    call(q, k, v, causal) -> Tensor | (Tensor, ...)   ← 与 bench_fa.py 的 backend 签名一致
    """

    def __init__(self, B, N, H, D, call, causal=True, warmup=3, label=""):
        dev = "npu"
        self.causal = causal
        self.shape = (B, N, H, D)
        self.call = call
        self.label = label

        # 静态 buffer：图里录的是【固定地址】，所以输入必须先拷进这里
        self.q = torch.zeros(B, N, H, D, dtype=torch.float16, device=dev)
        self.k = torch.zeros_like(self.q)
        self.v = torch.zeros_like(self.q)

        # ---- ① 预热（必须在捕获前）----
        # 1) 触发 JIT / 首次分配 / tiling 缓存填充（这些都不能发生在捕获里）
        # 2) 让 caching allocator 把内存块准备好
        for _ in range(warmup):
            call(self.q, self.k, self.v, causal)
        torch.npu.synchronize()

        # ---- ② 捕获 ----
        # 捕获期间不能有 .item() / print(tensor) / synchronize() / 依赖 tensor 值的分支
        self.g = torch.npu.NPUGraph()
        try:
            with torch.npu.graph(self.g):
                self._out = call(self.q, self.k, self.v, causal)
        except Exception as e:                       # noqa: BLE001
            print(f"[capture 失败{(' ' + label) if label else ''}] {type(e).__name__}: {e}")
            print("  → 若是 OpCommand 与图捕获冲突，用 -DFAT_NO_OPCOMMAND=ON 重新构建 .so")
            print("  → 或改用显式 capture_begin/capture_end（见文件末尾注释）")
            raise
        torch.npu.synchronize()
        print(f"[capture ok{(' ' + label) if label else ''}] "
              f"O={tuple(self.o.shape)} causal={causal}")

    @property
    def o(self):
        return self._out[0] if isinstance(self._out, (tuple, list)) else self._out

    def replay_only(self):
        """连输入拷贝都不做 —— 纯 replay 的地板价"""
        self.g.replay()

    def __call__(self, q, k, v):
        self.q.copy_(q)
        self.k.copy_(k)
        self.v.copy_(v)
        self.g.replay()
        return self.o


class GraphedFLA(GraphedFA):
    """Triton 的 fla_fwd 专用包装（保持原接口）"""

    def __init__(self, B, N, H, D, causal=True, warmup=3):
        # 惰性 import：只有真用 Triton 时才需要 triton_lab 在 sys.path 上。
        # 否则 --only catlass --graph 也会因为顶层 import 而失败。
        tl = os.path.join(os.path.dirname(HERE), "triton_lab")
        if tl not in sys.path:
            sys.path.insert(0, tl)
        from kernels.flash_attention.fa_fwd import fla_fwd  # noqa: PLC0415

        super().__init__(B, N, H, D,
                         call=lambda q, k, v, c: fla_fwd(q, k, v, causal=c),
                         causal=causal, warmup=warmup, label="triton")


def bench(fn, warmup=10, iters=50, repeats=5):
    """与 bench/bench_fa.py 完全一致的计时口径"""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()

    samples = []
    for _ in range(repeats):
        st = torch.npu.Event(enable_timing=True)
        en = torch.npu.Event(enable_timing=True)
        st.record()
        for _ in range(iters):
            fn()
        en.record()
        torch.npu.synchronize()
        samples.append(st.elapsed_time(en) / iters)

    samples.sort()
    med = samples[len(samples) // 2]
    return med, samples[0], samples[-1]


def golden(q, k, v, causal):
    s = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) / math.sqrt(q.size(-1))
    if causal:
        n = q.size(1)
        m = torch.triu(torch.ones((n, n), dtype=torch.bool, device=s.device), diagonal=1)
        s = s.masked_fill(m, float("-inf"))
    return torch.einsum("bhqk,bkhd->bqhd", torch.softmax(s, -1), v.float())


def main():
    CASES = [(1, 512, 8, 128), (1, 1024, 8, 128), (1, 2048, 8, 128), (1, 4096, 8, 128)]
    print(f"device: {torch.npu.get_device_name(0)}\n")

    for (B, N, H, D) in CASES:
        torch.manual_seed(0)
        q = torch.randn(B, N, H, D, dtype=torch.float16, device="npu")
        k = torch.randn(B, N, H, D, dtype=torch.float16, device="npu")
        v = torch.randn(B, N, H, D, dtype=torch.float16, device="npu")

        print(f"── B={B} H={H} N={N} D={D} ──")

        for causal in (False, True):
            tag = "causal" if causal else "full  "
            gf = GraphedFLA(B, N, H, D, causal=causal)

            # 正确性：replay 后用静态 buffer 里的输入算 golden
            gf(q, k, v)
            torch.npu.synchronize()
            err = (gf.o.cpu().float() - golden(gf.q.cpu(), gf.k.cpu(), gf.v.cpu(), causal)).abs().max().item()
            ok = "OK " if err < 2e-2 else "FAIL"
            print(f"  [{tag}] {ok} graphed max_err={err:.5f}")

            # 基线：未录图的裸调用（fla_fwd 在 GraphedFLA 里已惰性导入）
            triton_lab = os.path.join(os.path.dirname(HERE), "triton_lab")
            if triton_lab not in sys.path:
                sys.path.insert(0, triton_lab)
            from kernels.flash_attention.fa_fwd import fla_fwd as _fla_fwd  # noqa: PLC0415
            t_base, lo, hi = bench(lambda: _fla_fwd(q, k, v, causal=causal))
            t_call, lo2, hi2 = bench(lambda: gf(q, k, v))
            t_replay, lo3, hi3 = bench(lambda: gf.replay_only())

            print(f"  [{tag}]   原始 fla_fwd        {t_base:7.3f} ms  [{lo:.3f}, {hi:.3f}]")
            print(f"  [{tag}]   图 + 输入拷贝        {t_call:7.3f} ms  [{lo2:.3f}, {hi2:.3f}]"
                  f"   speedup {t_base / t_call:5.2f}x")
            print(f"  [{tag}]   图 (纯 replay)      {t_replay:7.3f} ms  [{lo3:.3f}, {hi3:.3f}]"
                  f"   speedup {t_base / t_replay:5.2f}x")
        print()


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# 如果 `with torch.npu.graph(g)` 这个上下文管理器在你的版本上不可用，
# 换成显式的 begin/end（语义相同）：
#
#     self.g = torch.npu.NPUGraph()
#     self.g.capture_begin()
#     self.o, self.lse = fla_fwd(self.q, self.k, self.v, causal=causal)
#     self.g.capture_end()
#
# 另外 `torch.npu.make_graphed_callables(fn, sample_args)` 也能用，
# 但它要求 fn 是 nn.Module（你这是普通函数），包一层反而更绕，不推荐。
# ---------------------------------------------------------------------------
