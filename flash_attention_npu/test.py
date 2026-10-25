import os

os.environ["TE_PARALLEL_COMPILER"] = "1"
os.environ["TE_PARALLEL_NUM"] = "1"
os.environ["TBE_PARALLEL_NUM"] = "1"
os.environ["MAX_COMPILE_CORE_NUMBER"] = "1"
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
import sys
import torch
import torch.nn.functional as F

try:
    import torch_npu

    has_npu = torch.npu.is_available()
except ImportError:
    has_npu = False

try:
    import flash_attention_npu
except ImportError:
    print(
        "[ERROR] Cannot import flash_attention_npu module. Please compile via: python3 setup.py build_ext --inplace"
    )
    sys.exit(1)


class FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale=None):
        if scale is not None:
            o, l = flash_attention_npu.forward(q, k, v, scale)
        else:
            o, l = flash_attention_npu.forward(q, k, v)
        ctx.save_for_backward(q, k, v, o, l)
        ctx.scale = scale
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, l = ctx.saved_tensors
        if ctx.scale is not None:
            dq, dk, dv = flash_attention_npu.backward(do, q, k, v, o, l, ctx.scale)
        else:
            dq, dk, dv = flash_attention_npu.backward(do, q, k, v, o, l)
        return dq, dk, dv, None


def flash_attention(q, k, v, scale=None):
    """PyTorch wrapper with autograd support for Ascend C FlashAttention."""
    return FlashAttentionFunction.apply(q, k, v, scale)


def golden_flash_attention(q, k, v, do=None, scale=None):
    d = q.shape[-1]
    if scale is None:
        scale = 1.0 / (d**0.5)

    q_f = q.cpu().float().clone().detach().requires_grad_(True)
    k_f = k.cpu().float().clone().detach().requires_grad_(True)
    v_f = v.cpu().float().clone().detach().requires_grad_(True)
    do_f = do.cpu().float() if do is not None else None

    scores = torch.matmul(q_f, k_f.transpose(-1, -2)) * scale
    p = F.softmax(scores, dim=-1)
    o = torch.matmul(p, v_f)
    l = torch.logsumexp(scores, dim=-1).half()

    if do_f is not None:
        loss = (o * do_f).sum()
        loss.backward()
        return o.half(), l, q_f.grad.half(), k_f.grad.half(), v_f.grad.half()

    return o.half(), l


def run_test_case(name, N_q, N_k, d, scale=None, atol=2e-2, rtol=2e-2):
    device = "npu:0" if has_npu else "cpu"

    torch.manual_seed(42)
    q = torch.randn((N_q, d), dtype=torch.half, device=device)
    k = torch.randn((N_k, d), dtype=torch.half, device=device)
    v = torch.randn((N_k, d), dtype=torch.half, device=device)
    do = torch.randn((N_q, d), dtype=torch.half, device=device)

    # 1. 前向计算
    if scale is not None:
        o_custom, l_custom = flash_attention_npu.forward(q, k, v, scale)
    else:
        o_custom, l_custom = flash_attention_npu.forward(q, k, v)

    # 2. 反向计算
    if scale is not None:
        dq_custom, dk_custom, dv_custom = flash_attention_npu.backward(
            do, q, k, v, o_custom, l_custom, scale
        )
    else:
        dq_custom, dk_custom, dv_custom = flash_attention_npu.backward(
            do, q, k, v, o_custom, l_custom
        )

    # 3. PyTorch 原生算子作为黄金标杆 (Golden)
    o_golden, l_golden, dq_golden, dk_golden, dv_golden = golden_flash_attention(
        q, k, v, do=do, scale=scale
    )

    # 4. 误差比对
    diff_o = torch.max(
        torch.abs(o_custom.cpu().float() - o_golden.cpu().float())
    ).item()
    diff_l = torch.max(
        torch.abs(l_custom.cpu().float() - l_golden.cpu().float())
    ).item()
    diff_dq = torch.max(
        torch.abs(dq_custom.cpu().float() - dq_golden.cpu().float())
    ).item()
    diff_dk = torch.max(
        torch.abs(dk_custom.cpu().float() - dk_golden.cpu().float())
    ).item()
    diff_dv = torch.max(
        torch.abs(dv_custom.cpu().float() - dv_golden.cpu().float())
    ).item()

    passed_o = torch.allclose(
        o_custom.cpu().float(), o_golden.cpu().float(), atol=atol, rtol=rtol
    )
    passed_l = torch.allclose(
        l_custom.cpu().float(), l_golden.cpu().float(), atol=atol, rtol=rtol
    )
    passed_dq = torch.allclose(
        dq_custom.cpu().float(), dq_golden.cpu().float(), atol=atol, rtol=rtol
    )
    passed_dk = torch.allclose(
        dk_custom.cpu().float(), dk_golden.cpu().float(), atol=atol, rtol=rtol
    )
    passed_dv = torch.allclose(
        dv_custom.cpu().float(), dv_golden.cpu().float(), atol=atol, rtol=rtol
    )

    passed = passed_o and passed_l and passed_dq and passed_dk and passed_dv
    status = "PASS" if passed else "FAIL"

    shape_str = f"Q:[{N_q},{d}], K:[{N_k},{d}]"
    print(
        f"{name:<42} | {shape_str:<22} | dO:{diff_o:6.3f} dL:{diff_l:6.3f} | dQ:{diff_dq:6.3f} dK:{diff_dk:6.3f} dV:{diff_dv:6.3f} | {status}"
    )

    if not passed:
        print(f"  [FAIL DETAILS]")
        if not passed_o:
            print(f"    O  custom[:4]: {o_custom.cpu().float().flatten()[:4].tolist()}")
            print(f"    O  golden[:4]: {o_golden.cpu().float().flatten()[:4].tolist()}")
        if not passed_l:
            print(f"    L  custom[:4]: {l_custom.cpu().float().flatten()[:4].tolist()}")
            print(f"    L  golden[:4]: {l_golden.cpu().float().flatten()[:4].tolist()}")
        if not passed_dq:
            print(
                f"    dQ custom[:4]: {dq_custom.cpu().float().flatten()[:4].tolist()}"
            )
            print(
                f"    dQ golden[:4]: {dq_golden.cpu().float().flatten()[:4].tolist()}"
            )
        if not passed_dk:
            print(
                f"    dK custom[:4]: {dk_custom.cpu().float().flatten()[:4].tolist()}"
            )
            print(
                f"    dK golden[:4]: {dk_golden.cpu().float().flatten()[:4].tolist()}"
            )
        if not passed_dv:
            print(
                f"    dV custom[:4]: {dv_custom.cpu().float().flatten()[:4].tolist()}"
            )
            print(
                f"    dV golden[:4]: {dv_golden.cpu().float().flatten()[:4].tolist()}"
            )

    # 5. 端到端 Autograd 测试
    q_auto = q.clone().detach().requires_grad_(True)
    k_auto = k.clone().detach().requires_grad_(True)
    v_auto = v.clone().detach().requires_grad_(True)
    o_auto = flash_attention(q_auto, k_auto, v_auto, scale)
    loss_auto = (o_auto * do).sum()
    loss_auto.backward()

    passed_autograd = (
        torch.allclose(
            q_auto.grad.cpu().float(), dq_golden.cpu().float(), atol=atol, rtol=rtol
        )
        and torch.allclose(
            k_auto.grad.cpu().float(), dk_golden.cpu().float(), atol=atol, rtol=rtol
        )
        and torch.allclose(
            v_auto.grad.cpu().float(), dv_golden.cpu().float(), atol=atol, rtol=rtol
        )
    )
    if not passed_autograd:
        print(f"  [WARN] Autograd wrapper check failed for {name}")
        passed = False

    return passed


def main():
    print("=" * 125)
    print(
        "   Ascend C FlashAttention Forward & Backward Operator Test Suite (Atlas 200I DK A2 / 310B)"
    )
    print("=" * 125)
    print(
        f"[INFO] Using Device: {'Ascend NPU' if has_npu else 'CPU (Direct ACL Host Launch)'}"
    )
    print(f"[INFO] Successfully loaded 'flash_attention_npu' module.\n")

    test_cases = [
        ("Base Block [N_q=64, N_k=64, d=64]", 64, 64, 64, None),
        ("Multi-Block K [N_q=64, N_k=128, d=64]", 64, 128, 64, None),
        ("Multi-Block Q [N_q=128, N_k=64, d=64]", 128, 64, 64, None),
        ("Square Multi-Block [N_q=128, N_k=128, d=64]", 128, 128, 64, None),
        ("Custom Scale 0.125 [N_q=64, N_k=64, d=64]", 64, 64, 64, 0.125),
        ("Deep Sequence [N_q=128, N_k=256, d=64]", 128, 256, 64, None),
    ]

    all_passed = True
    for name, N_q, N_k, d, scale in test_cases:
        try:
            passed = run_test_case(name, N_q, N_k, d, scale=scale)
            if not passed:
                all_passed = False
        except Exception as e:
            if "100039" in str(e) or "stub library" in str(e):
                print(
                    f"[INFO] ACL Stub Environment detected ({e}). Real NPU execution requires running on Atlas board."
                )
                return
            else:
                print(f"[ERROR] Exception in test case '{name}': {e}")
                all_passed = False

    print("=" * 125)
    if all_passed:
        print(
            "🎉 All FlashAttention Forward & Backward Test Cases Passed Successfully!"
        )
    else:
        print("❌ Some test cases failed. Please inspect logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
