import os
os.environ["TE_PARALLEL_COMPILER"] = "1"
os.environ["TE_PARALLEL_NUM"] = "1"
os.environ["TBE_PARALLEL_NUM"] = "1"
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import sys
import time
import argparse
import torch
import torch.nn.functional as F

try:
    import torch_npu
    HAS_NPU = torch.npu.is_available()
except ImportError:
    HAS_NPU = False

try:
    import flash_attention_npu
    HAS_CUSTOM_FA = True
except ImportError:
    HAS_CUSTOM_FA = False

try:
    import flash_attn
    from flash_attn import flash_attn_func
    HAS_TRI_DAO = True
except Exception:
    HAS_TRI_DAO = False


# ==============================================================================
# 1. 权威标杆实现库 (Authoritative Reference Baselines)
# ==============================================================================

class AuthoritativeBaselines:
    """
    封装各大权威 FlashAttention 实现:
    1. 华为官方生产级算子: torch_npu.npu_fusion_attention (aclnnFlashAttentionScore)
    2. PyTorch 官方标准算子: torch.nn.functional.scaled_dot_product_attention (SDPA)
    3. Tri Dao 原版接口 (若环境安装 flash_attn)
    4. PyTorch 原生未融合基线 (Naive Attention)
    """

    @staticmethod
    def detect_available_baselines():
        detected = []
        if not HAS_NPU:
            return [("cpu_naive", "PyTorch CPU Standard Attention")]

        # 1. 探测华为官方 npu_fusion_attention (aclnnFlashAttentionScore)
        if hasattr(torch_npu, "npu_fusion_attention"):
            try:
                q = torch.randn(1, 1, 64, 64, dtype=torch.half, device="npu:0")
                k = torch.randn(1, 1, 64, 64, dtype=torch.half, device="npu:0")
                v = torch.randn(1, 1, 64, 64, dtype=torch.half, device="npu:0")
                _ = torch_npu.npu_fusion_attention(
                    q, k, v, head_num=1, input_layout="BNSD", scale=0.125, keep_prob=1.0
                )
                torch.npu.synchronize()
                detected.append(("npu_fusion", "Huawei Official npu_fusion_attention (aclnnFlashAttentionScore)"))
            except Exception as e:
                pass

        # 2. 探测 PyTorch 官方 SDPA (scaled_dot_product_attention)
        try:
            q = torch.randn(1, 1, 64, 64, dtype=torch.half, device="npu:0")
            k = torch.randn(1, 1, 64, 64, dtype=torch.half, device="npu:0")
            v = torch.randn(1, 1, 64, 64, dtype=torch.half, device="npu:0")
            with torch.no_grad():
                _ = F.scaled_dot_product_attention(q, k, v, scale=0.125)
            torch.npu.synchronize()
            detected.append(("sdpa", "PyTorch Official F.scaled_dot_product_attention (SDPA)"))
        except Exception as e:
            pass

        # 3. 探测 Tri Dao flash_attn
        if HAS_TRI_DAO:
            detected.append(("tri_dao", "Tri Dao Official flash-attention (flash_attn_func)"))

        # 4. 保底基线：PyTorch 原生注意力
        detected.append(("naive", "PyTorch Naive Unfused Attention (Matmul + Softmax)"))
        return detected

    @staticmethod
    def run_npu_fusion_fwd(q, k, v, scale):
        """华为官方 npu_fusion_attention (4D BNSD [1, 1, S, D])."""
        q_4d = q.unsqueeze(0).unsqueeze(1)
        k_4d = k.unsqueeze(0).unsqueeze(1)
        v_4d = v.unsqueeze(0).unsqueeze(1)
        res = torch_npu.npu_fusion_attention(
            q_4d, k_4d, v_4d, head_num=1, input_layout="BNSD", scale=scale, keep_prob=1.0
        )
        return res[0].squeeze(0).squeeze(1)

    @staticmethod
    def run_npu_fusion_bwd(q, k, v, do, scale):
        """华为官方 npu_fusion_attention_grad 反向."""
        q_4d = q.unsqueeze(0).unsqueeze(1)
        k_4d = k.unsqueeze(0).unsqueeze(1)
        v_4d = v.unsqueeze(0).unsqueeze(1)
        do_4d = do.unsqueeze(0).unsqueeze(1)

        out, s_max, s_sum, s_out, seed, offset, numels = torch_npu.npu_fusion_attention(
            q_4d, k_4d, v_4d, head_num=1, input_layout="BNSD", scale=scale, keep_prob=1.0
        )

        if hasattr(torch_npu, "npu_fusion_attention_grad"):
            dq, dk, dv, _ = torch_npu.npu_fusion_attention_grad(
                q_4d, k_4d, v_4d, do_4d,
                softmax_max=s_max,
                softmax_sum=s_sum,
                softmax_out=s_out,
                attention_in=out,
                head_num=1,
                input_layout="BNSD",
                scale=scale,
                keep_prob=1.0
            )
            return dq.squeeze(0).squeeze(1), dk.squeeze(0).squeeze(1), dv.squeeze(0).squeeze(1)
        else:
            raise NotImplementedError("torch_npu.npu_fusion_attention_grad not available")

    @staticmethod
    def run_sdpa_fwd(q, k, v, scale):
        """PyTorch 官方 SDPA 前向."""
        q_4d = q.unsqueeze(0).unsqueeze(0)
        k_4d = k.unsqueeze(0).unsqueeze(0)
        v_4d = v.unsqueeze(0).unsqueeze(0)
        out = F.scaled_dot_product_attention(q_4d, k_4d, v_4d, scale=scale)
        return out.squeeze(0).squeeze(0)

    @staticmethod
    def run_sdpa_bwd(q, k, v, do, scale):
        """PyTorch 官方 SDPA 反向 (通过 Autograd)."""
        q_4d = q.unsqueeze(0).unsqueeze(0).detach().clone().requires_grad_(True)
        k_4d = k.unsqueeze(0).unsqueeze(0).detach().clone().requires_grad_(True)
        v_4d = v.unsqueeze(0).unsqueeze(0).detach().clone().requires_grad_(True)
        do_4d = do.unsqueeze(0).unsqueeze(0)

        out = F.scaled_dot_product_attention(q_4d, k_4d, v_4d, scale=scale)
        out.backward(do_4d)
        return q_4d.grad.squeeze(0).squeeze(0), k_4d.grad.squeeze(0).squeeze(0), v_4d.grad.squeeze(0).squeeze(0)

    @staticmethod
    def run_naive_fwd(q, k, v, scale):
        """未融合标准注意力前向."""
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        p = F.softmax(scores, dim=-1)
        out = torch.matmul(p, v)
        return out

    @staticmethod
    def run_naive_bwd(q, k, v, do, scale):
        """未融合标准注意力反向."""
        q_in = q.detach().clone().requires_grad_(True)
        k_in = k.detach().clone().requires_grad_(True)
        v_in = v.detach().clone().requires_grad_(True)

        scores = torch.matmul(q_in, k_in.transpose(-1, -2)) * scale
        p = F.softmax(scores, dim=-1)
        out = torch.matmul(p, v_in)
        out.backward(do)
        return q_in.grad, k_in.grad, v_in.grad


# ==============================================================================
# 2. 精准计时与指标计算 (Measurement Engine)
# ==============================================================================

def measure_latency_ms(fn, *args, warmup=10, iters=30):
    """NPU 强制同步精准计时 (记录 min 与 median 毫秒数)."""
    for _ in range(warmup):
        fn(*args)
    if HAS_NPU:
        torch.npu.synchronize()

    latencies = []
    for _ in range(iters):
        if HAS_NPU:
            torch.npu.synchronize()
            t0 = time.perf_counter()
            fn(*args)
            torch.npu.synchronize()
            t1 = time.perf_counter()
        else:
            t0 = time.perf_counter()
            fn(*args)
            t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    latencies.sort()
    return {
        "min": latencies[0],
        "median": latencies[len(latencies) // 2],
        "mean": sum(latencies) / len(latencies),
    }


def compute_tflops(N_q, N_k, d, time_ms, is_bwd=False):
    """计算实际有效算力吞吐 (TFLOPS).
    Attention FLOPs 计算标准:
      - Forward: 2 * (QK^T) + 2 * (PV) = 4 * N_q * N_k * d
      - Backward: 约 2.5 倍前向 = 10 * N_q * N_k * d
    """
    factor = 10.0 if is_bwd else 4.0
    total_flops = factor * float(N_q) * float(N_k) * float(d)
    return (total_flops / (time_ms * 1e-3)) / 1e12


# ==============================================================================
# 3. 对比测试执行器 (Comparison Runner)
# ==============================================================================

def benchmark_shape(N_q, N_k, d, primary_baseline_key, warmup=10, iters=30, include_bwd=True):
    device = "npu:0" if HAS_NPU else "cpu"
    scale = 1.0 / (d ** 0.5)

    q = torch.randn((N_q, d), dtype=torch.half).to(device)
    k = torch.randn((N_k, d), dtype=torch.half).to(device)
    v = torch.randn((N_k, d), dtype=torch.half).to(device)
    do = torch.randn((N_q, d), dtype=torch.half).to(device)

    # 1. 自定义 Ascend C FlashAttention 耗时
    custom_fwd_fn = lambda: flash_attention_npu.forward(q, k, v, scale)
    custom_fwd_stat = measure_latency_ms(custom_fwd_fn, warmup=warmup, iters=iters)
    custom_fwd_ms = custom_fwd_stat["median"]
    custom_fwd_tflops = compute_tflops(N_q, N_k, d, custom_fwd_ms, is_bwd=False)

    # 2. 权威标杆实现耗时 (根据 primary_baseline_key 路由)
    if primary_baseline_key == "npu_fusion":
        auth_fwd_fn = lambda: AuthoritativeBaselines.run_npu_fusion_fwd(q, k, v, scale)
    elif primary_baseline_key == "sdpa":
        auth_fwd_fn = lambda: AuthoritativeBaselines.run_sdpa_fwd(q, k, v, scale)
    else:
        auth_fwd_fn = lambda: AuthoritativeBaselines.run_naive_fwd(q, k, v, scale)

    auth_fwd_stat = measure_latency_ms(auth_fwd_fn, warmup=warmup, iters=iters)
    auth_fwd_ms = auth_fwd_stat["median"]
    auth_fwd_tflops = compute_tflops(N_q, N_k, d, auth_fwd_ms, is_bwd=False)

    # 3. 未融合朴素实现耗时 (Naive Baseline，作为显存优化基线)
    naive_fwd_fn = lambda: AuthoritativeBaselines.run_naive_fwd(q, k, v, scale)
    naive_fwd_stat = measure_latency_ms(naive_fwd_fn, warmup=warmup, iters=iters)
    naive_fwd_ms = naive_fwd_stat["median"]

    row_data = {
        "N_q": N_q,
        "N_k": N_k,
        "d": d,
        "custom_fwd_ms": custom_fwd_ms,
        "auth_fwd_ms": auth_fwd_ms,
        "naive_fwd_ms": naive_fwd_ms,
        "fwd_ratio_vs_auth": auth_fwd_ms / max(custom_fwd_ms, 1e-6),  # > 1.0x 表示比官方权威更快
        "fwd_speedup_vs_naive": naive_fwd_ms / max(custom_fwd_ms, 1e-6),
        "custom_fwd_tflops": custom_fwd_tflops,
        "auth_fwd_tflops": auth_fwd_tflops,
    }

    # 4. 反向评测 (可选)
    if include_bwd:
        o_c, l_c = flash_attention_npu.forward(q, k, v, scale)
        custom_bwd_fn = lambda: flash_attention_npu.backward(do, q, k, v, o_c, l_c, scale)
        custom_bwd_stat = measure_latency_ms(custom_bwd_fn, warmup=warmup, iters=iters)
        custom_bwd_ms = custom_bwd_stat["median"]
        custom_bwd_tflops = compute_tflops(N_q, N_k, d, custom_bwd_ms, is_bwd=True)

        # 标杆反向
        try:
            if primary_baseline_key == "npu_fusion":
                auth_bwd_fn = lambda: AuthoritativeBaselines.run_npu_fusion_bwd(q, k, v, do, scale)
            elif primary_baseline_key == "sdpa":
                auth_bwd_fn = lambda: AuthoritativeBaselines.run_sdpa_bwd(q, k, v, do, scale)
            else:
                auth_bwd_fn = lambda: AuthoritativeBaselines.run_naive_bwd(q, k, v, do, scale)
            auth_bwd_stat = measure_latency_ms(auth_bwd_fn, warmup=warmup, iters=iters)
            auth_bwd_ms = auth_bwd_stat["median"]
            auth_bwd_tflops = compute_tflops(N_q, N_k, d, auth_bwd_ms, is_bwd=True)
        except Exception as e:
            # 若官方接口不开放独立反向，回退到 Naive Autograd
            auth_bwd_fn = lambda: AuthoritativeBaselines.run_naive_bwd(q, k, v, do, scale)
            auth_bwd_stat = measure_latency_ms(auth_bwd_fn, warmup=warmup, iters=iters)
            auth_bwd_ms = auth_bwd_stat["median"]
            auth_bwd_tflops = compute_tflops(N_q, N_k, d, auth_bwd_ms, is_bwd=True)

        # 朴素反向
        naive_bwd_fn = lambda: AuthoritativeBaselines.run_naive_bwd(q, k, v, do, scale)
        naive_bwd_stat = measure_latency_ms(naive_bwd_fn, warmup=warmup, iters=iters)
        naive_bwd_ms = naive_bwd_stat["median"]

        row_data.update({
            "custom_bwd_ms": custom_bwd_ms,
            "auth_bwd_ms": auth_bwd_ms,
            "naive_bwd_ms": naive_bwd_ms,
            "bwd_ratio_vs_auth": auth_bwd_ms / max(custom_bwd_ms, 1e-6),
            "bwd_speedup_vs_naive": naive_bwd_ms / max(custom_bwd_ms, 1e-6),
            "custom_bwd_tflops": custom_bwd_tflops,
            "auth_bwd_tflops": auth_bwd_tflops,
        })

    return row_data


# ==============================================================================
# 4. 可视化报告打印 (Report Formatter)
# ==============================================================================

def print_benchmark_comparison_table(results, baseline_name, include_bwd=True):
    print("\n" + "=" * 118)
    print(f" FLASHATTENTION PERFORMANCE BENCHMARK: Custom Ascend C vs. Authoritative Reference")
    print(f" Primary Authoritative Baseline: {baseline_name}")
    print("=" * 118)

    if not include_bwd:
        header = (
            f"| {'Shape (N_q, N_k, d)':<20} | {'Our Fwd (ms)':<14} | {'Official Fwd (ms)':<18} "
            f"| {'vs. Official':<14} | {'vs. Naive Attn':<16} | {'Our TFLOPS':<12} |"
        )
        sep = "|" + "-" * 22 + "|" + "-" * 16 + "|" + "-" * 20 + "|" + "-" * 16 + "|" + "-" * 18 + "|" + "-" * 14 + "|"
        print(header)
        print(sep)
        for r in results:
            shape_str = f"({r['N_q']}, {r['N_k']}, {r['d']})"
            ratio_str = f"{r['fwd_ratio_vs_auth']:.2f}x ({'Faster' if r['fwd_ratio_vs_auth']>=1.0 else 'Slower'})"
            naive_str = f"{r['fwd_speedup_vs_naive']:.2f}x faster"
            print(
                f"| {shape_str:<20} | {r['custom_fwd_ms']:>14.3f} | {r['auth_fwd_ms']:>18.3f} "
                f"| {ratio_str:>14} | {naive_str:>16} | {r['custom_fwd_tflops']:>12.2f} |"
            )
    else:
        header = (
            f"| {'Shape (N_q, N_k, d)':<18} | {'Fwd (Our / Off)':<17} | {'Fwd vs Off':<11} "
            f"| {'Bwd (Our / Off)':<17} | {'Bwd vs Off':<11} | {'vs. Naive (F/B)':<17} | {'Our TFLOPS (F/B)':<18} |"
        )
        sep = (
            "|" + "-" * 20 + "|" + "-" * 19 + "|" + "-" * 13 + "|"
            + "-" * 19 + "|" + "-" * 13 + "|" + "-" * 19 + "|" + "-" * 20 + "|"
        )
        print(header)
        print(sep)
        for r in results:
            shape_str = f"({r['N_q']}, {r['N_k']}, {r['d']})"
            fwd_str = f"{r['custom_fwd_ms']:.2f} / {r['auth_fwd_ms']:.2f}"
            bwd_str = f"{r['custom_bwd_ms']:.2f} / {r['auth_bwd_ms']:.2f}"
            fwd_ratio_str = f"{r['fwd_ratio_vs_auth']:.2f}x"
            bwd_ratio_str = f"{r['bwd_ratio_vs_auth']:.2f}x"
            naive_str = f"{r['fwd_speedup_vs_naive']:.1f}x / {r['bwd_speedup_vs_naive']:.1f}x"
            tflops_str = f"{r['custom_fwd_tflops']:.1f} / {r['custom_bwd_tflops']:.1f}"

            print(
                f"| {shape_str:<18} | {fwd_str:<17} | {fwd_ratio_str:>11} "
                f"| {bwd_str:<17} | {bwd_ratio_str:>11} | {naive_str:<17} | {tflops_str:<18} |"
            )

    print("=" * 118)
    print("指标解读指南:")
    print("  1. Fwd / Bwd (Our / Off): 毫秒数 (自定义实现耗时 / 权威标杆耗时).")
    print("  2. vs Off (对比权威实现):")
    print("     - >= 1.0x: 自定义算子速度超越或打平官方权威实现 (如 1.05x 表示比官方快 5%).")
    print("     - < 1.0x:  自定义算子达到官方实现的百分比 (如 0.85x 表示性能达到官方的 85%).")
    print("  3. vs. Naive (F/B): 相较于未融合的 PyTorch 原生标准实现的加速倍率.")
    print("  4. Our TFLOPS: 自定义算子在当前硬件上压榨出的实际浮点计算能力 (数值越高说明算力利用率越高).\n")


def main():
    parser = argparse.ArgumentParser(description="Ascend C FlashAttention vs. Authoritative Reference Benchmark")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations (default: 10)")
    parser.add_argument("--iters", type=int, default=30, help="Number of timing iterations (default: 30)")
    parser.add_argument("--no-bwd", action="store_true", help="Skip backward benchmark and evaluate forward only")
    parser.add_argument("--device", type=int, default=0, help="NPU device id (default: 0)")
    parser.add_argument(
        "--baseline",
        type=str,
        default="auto",
        choices=["auto", "npu_fusion", "sdpa", "naive"],
        help="Target authoritative baseline to compare against (default: auto)",
    )
    args = parser.parse_args()

    if not HAS_CUSTOM_FA:
        print("[ERROR] Custom flash_attention_npu extension not found. Please compile it first.")
        sys.exit(1)

    if HAS_NPU:
        torch.npu.set_device(args.device)

    # 探测环境支持的标杆实现
    available_baselines = AuthoritativeBaselines.detect_available_baselines()
    print("\n[Authoritative FlashAttention Implementations Detected in Environment]")
    for key, desc in available_baselines:
        print(f"  * [{key}]: {desc}")

    # 选择比对的主标杆
    if args.baseline == "auto":
        primary_key, primary_desc = available_baselines[0]
    else:
        primary_key = args.baseline
        primary_desc = dict(available_baselines).get(primary_key, primary_key)

    print(f"\n[Active Comparison Target]")
    print(f"  Authoritative Baseline: {primary_desc}")
    print(f"  Test Device:            {'NPU: ' + str(torch.npu.get_device_name(args.device)) if HAS_NPU else 'CPU'}")
    print(f"  Warmup / Timing Iters:  {args.warmup} / {args.iters}")
    print(f"  Benchmark Backward:     {not args.no_bwd}")

    # 典型测试矩阵：小块、长序列、非对称
    benchmark_shapes = [
        (64, 64, 64),
        (64, 128, 64),
        (128, 64, 64),
        (128, 128, 64),
        (256, 256, 64),
        (512, 512, 64),
    ]

    results = []
    for (nq, nk, d) in benchmark_shapes:
        print(f"\n>>> Benchmarking Shape ({nq}, {nk}, {d})...")
        try:
            stat = benchmark_shape(
                nq, nk, d,
                primary_baseline_key=primary_key,
                warmup=args.warmup,
                iters=args.iters,
                include_bwd=not args.no_bwd
            )
            results.append(stat)
            fwd_our = stat['custom_fwd_ms']
            fwd_auth = stat['auth_fwd_ms']
            ratio = stat['fwd_ratio_vs_auth']
            print(f"    Done: Our Fwd = {fwd_our:.3f} ms, Official Fwd = {fwd_auth:.3f} ms (Ratio: {ratio:.2f}x)")
        except Exception as e:
            print(f"    [SKIP] Shape ({nq}, {nk}, {d}) encountered error: {e}")

    if results:
        print_benchmark_comparison_table(results, primary_desc, include_bwd=not args.no_bwd)


if __name__ == "__main__":
    main()
