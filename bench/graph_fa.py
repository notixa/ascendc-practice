"""
graph_fa.py —— 用图捕获消除 Triton 的 Python launcher 开销

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
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "triton_lab"))
from kernels.flash_attention.fa_fwd import fla_fwd  # noqa: E402


class GraphedFLA:
    """把 fla_fwd 录成一张图，之后只 replay。fla_fwd 本身不用改。"""

    def __init__(self, B, N, H, D, causal=True, warmup=3):
        dev = "npu"
        self.causal = causal
        self.shape = (B, N, H, D)

        # 静态 buffer：图里录的是【固定地址】，所以输入必须先拷进这里
        self.q = torch.zeros(B, N, H, D, dtype=torch.float16, device=dev)
        self.k = torch.zeros_like(self.q)
        self.v = torch.zeros_like(self.q)

        # ---- ① 预热（必须在捕获前）----
        # 1) 触发 Triton JIT 编译（编译不能发生在捕获里）
        # 2) 让 caching allocator 把内存块准备好
        for _ in range(warmup):
            fla_fwd(self.q, self.k, self.v, causal=causal)
        torch.npu.synchronize()

        # ---- ② 捕获 ----
        # 捕获期间：不能有 .item()/print(tensor)/synchronize()/依赖 tensor 值的分支
        #           fla_fwd 里这两条都满足 ✓
        self.g = torch.npu.NPUGraph()
        try:
            with torch.npu.graph(self.g):
                self.o, self.lse = fla_fwd(self.q, self.k, self.v, causal=causal)
        except Exception as e:                       # noqa: BLE001
            print(f"[capture 失败] {type(e).__name__}: {e}")
            print("→ 试试 capture_begin/capture_end 写法（见文件末尾注释）")
            raise
        torch.npu.synchronize()
        print(f"[capture ok] O={tuple(self.o.shape)} L={tuple(self.lse.shape)}")

    def replay_only(self):
        """连输入拷贝都不做 —— 纯 replay 的地板价"""
        self.g.replay()

    def __call__(self, q, k, v):
        self.q.copy_(q)
        self.k.copy_(k)
        self.v.copy_(v)
        self.g.replay()
        return self.o


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
        m = torch.triu(torch.ones(n, n, dtype=torch.bool), device=s.device)
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

            t_base, lo, hi = bench(lambda: fla_fwd(q, k, v, causal=causal))
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
