# pyright: reportUnreachable=false, reportArgumentType=false
# 1) @triton.jit 的 __call__ 永远 raise，pyright 会把后面的代码判成 unreachable；
# 2) tl.constexpr 是给 Triton 编译器看的标记，运行时就是 int/bool，
#    pyright 会把它当成独立类型，误报“实参与 constexpr 不兼容”。

import torch
import torch_npu
import triton
import triton.backends.ascend.runtime
import triton.language as tl
from torch_npu.npu import get_device_properties

# 注意：BKV=256 在 N 不能被 256 整除时（如 S=1000）会触发
# "load2d 写地址越界" 的 aicore 异常（507015），尾块修复前不要加回来。
# 32x256 已实测崩溃；64x256 在 S=1000 编译 UB 超限。
_TILES = [
    (128, 128),
    (96, 128),
    (64, 128),
    (32, 128),
    (32, 64),
]


@triton.autotune(
    configs=[triton.Config({"BLOCK_Q": bq, "BLOCK_K": bk}) for bq, bk in _TILES],
    key=["N_q", "N_k", "IS_CAUSAL"],
)
@triton.jit
def fla_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    l_ptr,
    stride_qb,
    stride_qq,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vh,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_oh,
    stride_od,
    stride_lb,
    stride_lh,
    stride_lq,
    batch,
    head,
    scale,
    N_q: tl.constexpr,
    N_k: tl.constexpr,
    delta_qk: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    EVEN_Q = N_q % BLOCK_Q == 0
    EVEN_K = N_k % BLOCK_K == 0
    blocks_q = tl.cdiv(N_q, BLOCK_Q)
    taskNum = batch * head * blocks_q
    taskIdx = tl.program_id(0)
    coreNum = tl.num_programs(0)
    for task in range(taskIdx, taskNum, coreNum):
        b = task // (head * blocks_q)
        i = blocks_q - 1 - ((task % (head * blocks_q)) // head)
        h = (task % (head * blocks_q)) % head
        q_base = q_ptr + b * stride_qb + h * stride_qh
        o_base = o_ptr + b * stride_ob + h * stride_oh
        l_base = l_ptr + b * stride_lb + h * stride_lh
        k_base = k_ptr + b * stride_kb + h * stride_kh
        v_base = v_ptr + b * stride_vb + h * stride_vh
        q_block_ptr = tl.make_block_ptr(
            q_base,
            (N_q, HEAD_DIM),
            strides=(stride_qq, stride_qd),
            offsets=(i * BLOCK_Q, 0),
            block_shape=(BLOCK_Q, HEAD_DIM),
            order=(1, 0),
        )
        o_block_ptr = tl.make_block_ptr(
            o_base,
            (N_q, HEAD_DIM),
            strides=(stride_oq, stride_od),
            offsets=(i * BLOCK_Q, 0),
            block_shape=(BLOCK_Q, HEAD_DIM),
            order=(1, 0),
        )
        l_block_ptr = tl.make_block_ptr(
            l_base,
            (N_q,),
            strides=(stride_lq,),
            offsets=(i * BLOCK_Q,),
            block_shape=(BLOCK_Q,),
            order=(0,),
        )
        k_block_ptr = tl.make_block_ptr(
            k_base,
            (N_k, HEAD_DIM),
            strides=(stride_kk, stride_kd),
            offsets=(0, 0),
            block_shape=(BLOCK_K, HEAD_DIM),
            order=(1, 0),
        )
        v_block_ptr = tl.make_block_ptr(
            v_base,
            (N_k, HEAD_DIM),
            strides=(stride_vk, stride_vd),
            offsets=(0, 0),
            block_shape=(BLOCK_K, HEAD_DIM),
            order=(1, 0),
        )
        mi = tl.full([BLOCK_Q], -1e38, dtype=tl.float32)
        li = tl.zeros([BLOCK_Q], dtype=tl.float32)
        acc = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)

        if EVEN_Q:
            q = tl.load(q_block_ptr)
        else:
            q = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")

        q = (q * scale).to(q_ptr.dtype.element_ty)

        no_mask_end = (i * BLOCK_Q if IS_CAUSAL else N_k) + delta_qk
        no_mask_block_num = no_mask_end // BLOCK_K
        no_mask_upper = no_mask_block_num * BLOCK_K if IS_CAUSAL else N_k

        q_start_r = i * BLOCK_Q

        acc, li, mi = fla_fwd_kernel_inner_loop(
            acc,
            li,
            mi,
            q,
            k_block_ptr,
            v_block_ptr,
            q_start_r,
            0,
            no_mask_upper,
            N_k,
            delta_qk,
            BLOCK_Q=BLOCK_Q,
            BLOCK_K=BLOCK_K,
            EVEN_K=EVEN_K,
            IS_DIAG=False,
        )

        if IS_CAUSAL:
            visible_end = tl.minimum(N_q, (i + 1) * BLOCK_Q) + delta_qk
            acc, li, mi = fla_fwd_kernel_inner_loop(
                acc,
                li,
                mi,
                q,
                k_block_ptr,
                v_block_ptr,
                q_start_r,
                no_mask_upper,
                visible_end,
                N_k,
                delta_qk,
                BLOCK_Q=BLOCK_Q,
                BLOCK_K=BLOCK_K,
                EVEN_K=EVEN_K,
                IS_DIAG=True,
            )

        acc = acc / li[:, None]
        li = (mi + tl.log2(li)) * 0.6931471805599453
        if EVEN_Q:
            tl.store(o_block_ptr, acc.to(o_ptr.dtype.element_ty))
            tl.store(l_block_ptr, li)
        else:
            tl.store(o_block_ptr, acc.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))
            tl.store(l_block_ptr, li, boundary_check=(0,))


@triton.jit
def fla_fwd_kernel_inner_loop(
    acc,
    li,
    mi,
    q,
    k_block_ptr,
    v_block_ptr,
    q_start_r,
    lower_bound,
    upper_bound,
    N_k,
    delta_qk,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    IS_DIAG: tl.constexpr,
):
    k_block_ptr = tl.advance(k_block_ptr, (lower_bound, 0))
    v_block_ptr = tl.advance(v_block_ptr, (lower_bound, 0))
    for j in range(lower_bound, upper_bound, BLOCK_K):
        if EVEN_K:
            k = tl.load(k_block_ptr)
            v = tl.load(v_block_ptr)
            s = tl.dot(q, tl.trans(k))
        else:
            k = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
            v = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")
            k_offsets = j + tl.arange(0, BLOCK_K)
            s = tl.where(k_offsets[None, :] < N_k, tl.dot(q, tl.trans(k)), -1.0e6)

        if IS_DIAG:
            q_idx = q_start_r + tl.arange(0, BLOCK_Q)
            k_idx = j + tl.arange(0, BLOCK_K)
            s = tl.where(k_idx[None, :] <= q_idx[:, None] + delta_qk, s, -1.0e6)
        m_prev = mi
        mi = tl.maximum(m_prev, tl.max(s, axis=-1))
        s = tl.exp2(s - mi[:, None].to(v.dtype))
        s = s.to(tl.float16)
        factor = tl.exp2(m_prev - mi)
        acc = factor[:, None] * acc
        acc = tl.dot(s, v, acc)
        li = factor * li + tl.sum(s.to(tl.float32), axis=-1)
        k_block_ptr = tl.advance(k_block_ptr, (BLOCK_K, 0))
        v_block_ptr = tl.advance(v_block_ptr, (BLOCK_K, 0))
    return acc, li, mi


def fla_fwd(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, causal=False):
    assert Q.stride(3) == K.stride(3) == V.stride(3) == 1
    assert Q.stride(2) == K.stride(2) == V.stride(2)
    batch = Q.size(dim=0)
    N_q = Q.size(dim=1)
    N_k = K.size(dim=1)
    head = Q.size(dim=2)
    head_dim = Q.size(dim=3)

    O = torch.zeros_like(Q, device=Q.device)
    L = torch.zeros([batch, head, N_q], dtype=torch.float32, device=Q.device)

    LOG2E = 1.4426950408889634
    scale = head_dim**-0.5 * LOG2E

    # 带 tl.dot 的 kernel 用 AI Core 数；纯 Vector 算子用 vector_core_num
    cores = get_device_properties(0).cube_core_num

    delta_qk = N_k - N_q

    grid = lambda meta: (min(batch * head * triton.cdiv(N_q, meta["BLOCK_Q"]), cores),)
    fla_fwd_kernel[grid](
        Q,
        K,
        V,
        O,
        L,
        Q.stride(0),
        Q.stride(1),
        Q.stride(2),
        Q.stride(3),
        K.stride(0),
        K.stride(1),
        K.stride(2),
        K.stride(3),
        V.stride(0),
        V.stride(1),
        V.stride(2),
        V.stride(3),
        O.stride(0),
        O.stride(1),
        O.stride(2),
        O.stride(3),
        L.stride(0),
        L.stride(1),
        L.stride(2),
        batch,
        head,
        scale,
        N_q=N_q,
        N_k=N_k,
        delta_qk=delta_qk,
        HEAD_DIM=head_dim,
        IS_CAUSAL=causal,
    )
    return O, L
