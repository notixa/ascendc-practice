import os

os.environ["TE_PARALLEL_COMPILER"] = "1"
os.environ["TE_PARALLEL_NUM"] = "1"
os.environ["TBE_PARALLEL_NUM"] = "1"
os.environ["MAX_COMPILE_CORE_NUMBER"] = "1"
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import time
import math
import torch
import torch.nn.functional as F
import torch_npu
import flash_attention_npu

torch.npu.set_device(0)

PEAK_TFLOPS = 8.192  # Atlas 200I DK A2 FP16 Peak (1 Core @ 1.0 GHz)
PEAK_BW_GBPS = 25.6  # LPDDR4X 带宽峰值


def time_sync(fn, warmup=5, iters=15):
    """精确 NPU 同步计时，剔除首轮及异常点，取中位数与平均值"""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()

    times = []
    for _ in range(iters):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    times.sort()
    return {
        "median": times[len(times) // 2],
        "min": times[0],
        "mean": sum(times) / len(times),
        "p90": times[int(len(times) * 0.9)],
    }


def get_peak_memory(fn):
    """测量函数运行期间真实新增的物理分配显存 (MB)"""
    torch.npu.synchronize()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats(0)
    fn()
    torch.npu.synchronize()
    return torch.npu.max_memory_allocated(0) / (1024.0 * 1024.0)


def profile_naive_breakdown(q, k, v, scale):
    """深入剖析 Naive Attention 各个子算子的耗时分布 (Matmul1, Scale+Softmax, Matmul2)"""
    torch.npu.synchronize()
    # 1. Q @ K^T
    t0 = time.perf_counter()
    scores = torch.matmul(q, k.transpose(-1, -2))
    torch.npu.synchronize()
    t_mm1 = (time.perf_counter() - t0) * 1000.0

    # 2. Scale + Softmax
    t0 = time.perf_counter()
    scores = scores * scale
    p = F.softmax(scores, dim=-1)
    torch.npu.synchronize()
    t_softmax = (time.perf_counter() - t0) * 1000.0

    # 3. P @ V
    t0 = time.perf_counter()
    out = torch.matmul(p, v)
    torch.npu.synchronize()
    t_mm2 = (time.perf_counter() - t0) * 1000.0

    return t_mm1, t_softmax, t_mm2


def run_case(b, h, nq, nk, d, test_naive=True):
    print(
        f"\n=========================================================================================="
    )
    print(f"Realistic Benchmark: Batch={b}, Heads={h}, N_q={nq}, N_k={nk}, d={d}")
    print(
        f"=========================================================================================="
    )
    scale = 1.0 / math.sqrt(d)
    total_heads = b * h

    # 构造标准 4D [B, H, N, d] 或 3D [H, N, d] 输入
    if b == 1:
        shape_q = (h, nq, d)
        shape_k = (h, nk, d)
    else:
        shape_q = (b, h, nq, d)
        shape_k = (b, h, nk, d)

    q = torch.randn(shape_q, dtype=torch.half, device="npu:0")
    k = torch.randn(shape_k, dtype=torch.half, device="npu:0")
    v = torch.randn(shape_k, dtype=torch.half, device="npu:0")
    do = torch.randn(shape_q, dtype=torch.half, device="npu:0")
    torch.npu.synchronize()

    # Theoretical FLOPs:
    # Forward = 4 * B * H * N_q * N_k * d
    # Backward = 10 * B * H * N_q * N_k * d
    flops_fwd = 4.0 * total_heads * nq * nk * d
    flops_bwd = 10.0 * total_heads * nq * nk * d

    # 1. Custom FlashAttention Forward
    fn_custom_fwd = lambda: flash_attention_npu.forward(q, k, v, scale)
    stat_c_fwd = time_sync(fn_custom_fwd, warmup=5, iters=15)
    t_c_fwd = stat_c_fwd["median"]
    tflops_c_fwd = (flops_fwd / (t_c_fwd * 1e-3)) / 1e12
    mfu_c_fwd = (tflops_c_fwd / PEAK_TFLOPS) * 100.0
    mem_c_fwd = get_peak_memory(fn_custom_fwd)

    # 2. Custom FlashAttention Backward
    o_c, l_c = flash_attention_npu.forward(q, k, v, scale)
    torch.npu.synchronize()
    fn_custom_bwd = lambda: flash_attention_npu.backward(do, q, k, v, o_c, l_c, scale)
    stat_c_bwd = time_sync(fn_custom_bwd, warmup=5, iters=15)
    t_c_bwd = stat_c_bwd["median"]
    tflops_c_bwd = (flops_bwd / (t_c_bwd * 1e-3)) / 1e12
    mfu_c_bwd = (tflops_c_bwd / PEAK_TFLOPS) * 100.0

    print(f"[Custom FlashAttention]")
    print(
        f"  Forward:  Total = {t_c_fwd:.3f} ms | Per-Head = {t_c_fwd / total_heads:.3f} ms | TFLOPS = {tflops_c_fwd:.3f} ({mfu_c_fwd:.1f}% MFU) | Peak Mem = {mem_c_fwd:.2f} MB"
    )
    print(
        f"  Backward: Total = {t_c_bwd:.3f} ms | Per-Head = {t_c_bwd / total_heads:.3f} ms | TFLOPS = {tflops_c_bwd:.3f} ({mfu_c_bwd:.1f}% MFU)"
    )

    # 3. PyTorch Naive Attention
    t_n_fwd = 0.0
    t_n_bwd = 0.0
    mem_n_fwd = 0.0
    speedup_fwd = 0.0
    speedup_bwd = 0.0

    # 检查是否可能引发 OOM：中间注意矩阵 B * H * N_q * N_k * 2 字节
    attn_matrix_mb = (total_heads * nq * nk * 2) / (1024.0 * 1024.0)
    print(f"  Intermediate Attention Matrix Size: {attn_matrix_mb:.2f} MB")

    if test_naive:
        if attn_matrix_mb > 1500:
            print(
                f"  [Notice] Attention matrix > 1.5GB, skipping Naive benchmark to prevent OOM hang on Atlas 200I (3.5GB RAM)."
            )
        else:
            try:
                fn_naive_fwd = lambda: torch.matmul(
                    F.softmax(torch.matmul(q, k.transpose(-1, -2)) * scale, dim=-1), v
                )
                # 自适应迭代轮次
                iters_n = 1 if nq >= 1024 else 5
                stat_n_fwd = time_sync(
                    fn_naive_fwd, warmup=1 if nq < 1024 else 0, iters=iters_n
                )
                t_n_fwd = stat_n_fwd["median"]
                mem_n_fwd = get_peak_memory(fn_naive_fwd)
                speedup_fwd = t_n_fwd / max(t_c_fwd, 1e-5)

                # 分解子算子开销
                t_mm1, t_soft, t_mm2 = profile_naive_breakdown(q, k, v, scale)

                print(f"[PyTorch Naive Attention]")
                print(
                    f"  Forward:  Total = {t_n_fwd:.3f} ms | Per-Head = {t_n_fwd / total_heads:.3f} ms | Peak Mem = {mem_n_fwd:.2f} MB"
                )
                print(
                    f"    Sub-op Breakdown: QK^T = {t_mm1:.2f} ms, Softmax = {t_soft:.2f} ms, PV = {t_mm2:.2f} ms"
                )

                # 反向测试
                if nq <= 1024:

                    def fn_naive_bwd():
                        q_in = q.detach().clone().requires_grad_(True)
                        k_in = k.detach().clone().requires_grad_(True)
                        v_in = v.detach().clone().requires_grad_(True)
                        out = torch.matmul(
                            F.softmax(
                                torch.matmul(q_in, k_in.transpose(-1, -2)) * scale,
                                dim=-1,
                            ),
                            v_in,
                        )
                        out.backward(do)

                    stat_n_bwd = time_sync(
                        fn_naive_bwd, warmup=0, iters=1 if nq >= 1024 else 3
                    )
                    t_n_bwd = stat_n_bwd["median"]
                    speedup_bwd = t_n_bwd / max(t_c_bwd, 1e-5)
                    print(
                        f"  Backward: Total = {t_n_bwd:.3f} ms | Per-Head = {t_n_bwd / total_heads:.3f} ms"
                    )
            except Exception as e:
                print(f"  [Naive Error]: {e}")

    if t_n_fwd > 0:
        print(
            f">>> Realistic Speedup: Forward = {speedup_fwd:.2f}x | Backward = {speedup_bwd:.2f}x | Memory Reduction = {mem_n_fwd / max(mem_c_fwd, 0.01):.1f}x"
        )

    return {
        "b": b,
        "h": h,
        "nq": nq,
        "nk": nk,
        "d": d,
        "t_c_fwd": t_c_fwd,
        "t_c_bwd": t_c_bwd,
        "tflops_fwd": tflops_c_fwd,
        "tflops_bwd": tflops_c_bwd,
        "mem_c": mem_c_fwd,
        "t_n_fwd": t_n_fwd,
        "t_n_bwd": t_n_bwd,
        "mem_n": mem_n_fwd,
        "speedup_fwd": speedup_fwd,
        "speedup_bwd": speedup_bwd,
    }


if __name__ == "__main__":
    cases = [
        # (B, H, N_q, N_k, d, test_naive)
        # 1. 真实大模型单批次多头：N 从 512 增至 1024、2048
        (1, 8, 512, 512, 64, True),
        (1, 8, 1024, 1024, 64, True),
        (1, 8, 2048, 2048, 64, True),
        # 2. 真实大模型多并发 (Batch = 2)
        (2, 4, 512, 512, 64, True),
        (2, 4, 1024, 1024, 64, True),
        (2, 8, 512, 512, 64, True),
        (2, 8, 1024, 1024, 64, True),
        (2, 8, 2048, 2048, 64, True),
    ]

    all_res = []
    for b, h, nq, nk, d, test_naive in cases:
        try:
            r = run_case(b, h, nq, nk, d, test_naive)
            all_res.append(r)
        except Exception as e:
            print(f"Error on ({b}, {h}, {nq}): {e}")

    print("\n" + "=" * 120)
    print("                    REAL-WORLD SCENARIO ATTENTION BENCHMARK SUMMARY")
    print(
        " Target Hardware: Ascend 310B4 (Atlas 200I DK A2) | FP16 Peak: 8.192 TFLOPS | RAM: 3.5GB"
    )
    print("=" * 120)
    print(
        f"| {'Case Config':<24} | {'FA Fwd (ms)':<12} | {'Naive Fwd (ms)':<15} | {'Fwd Speedup':<12} | {'FA Bwd (ms)':<12} | {'FA TFLOPS':<10} | {'Mem Cut':<10} |"
    )
    print(
        "|"
        + "-" * 26
        + "|"
        + "-" * 14
        + "|"
        + "-" * 17
        + "|"
        + "-" * 14
        + "|"
        + "-" * 14
        + "|"
        + "-" * 12
        + "|"
        + "-" * 12
        + "|"
    )
    for r in all_res:
        cfg = f"B={r['b']} H={r['h']} N={r['nq']}"
        fwd_c = f"{r['t_c_fwd']:.2f}"
        fwd_n = f"{r['t_n_fwd']:.2f}" if r["t_n_fwd"] > 0 else "OOM/Skipped"
        sp_fwd = f"{r['speedup_fwd']:.2f}x" if r["speedup_fwd"] > 0 else "N/A"
        bwd_c = f"{r['t_c_bwd']:.2f}"
        tflops = f"{r['tflops_fwd']:.2f}"
        mem_cut = (
            f"{r['mem_n'] / max(r['mem_c'], 0.01):.1f}x" if r["mem_n"] > 0 else "N/A"
        )
        print(
            f"| {cfg:<24} | {fwd_c:<12} | {fwd_n:<15} | {sp_fwd:<12} | {bwd_c:<12} | {tflops:<10} | {mem_cut:<10} |"
        )
    print("=" * 120)
