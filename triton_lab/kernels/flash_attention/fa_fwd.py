# pyright: reportUnreachable=false, reportArgumentType=false
# 1) @triton.jit 的 __call__ 永远 raise，pyright 会把后面的代码判成 unreachable；
# 2) tl.constexpr 是给 Triton 编译器看的标记，运行时就是 int/bool，
#    pyright 会把它当成独立类型，误报“实参与 constexpr 不兼容”。

import torch
import torch_npu
import triton
import triton.language as tl
from torch_npu.npu import get_device_properties


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
    N_q,
    N_k,
    head,
    scale,
    delta_qk,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    EVEN_Q: tl.constexpr,
    EVEN_K: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    MULTI_BLOCK: tl.constexpr,
):
    blocks_q = tl.cdiv(N_q, BLOCK_Q)
    taskNum = batch * head * blocks_q
    taskIdx = tl.program_id(0)
    coreNum = tl.num_programs(0)
    for task in range(taskIdx, taskNum, coreNum):
        b = task // (head * blocks_q)
        i = blocks_q - ((task % (head * blocks_q)) // head)
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
        m = tl.full([BLOCK_Q], -1e38, dtype=tl.float32)
        l = tl.zeros([BLOCK_Q], dtype=tl.float32)
        o = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)
        T_k_full = tl.cdiv(N_k, BLOCK_K)
        if IS_CAUSAL:
            T_k = tl.minimum(tl.cdiv((i + 1) * BLOCK_Q + delta_qk, BLOCK_K), T_k_full)
        else:
            T_k = T_k_full
        if EVEN_Q:
            q = tl.load(q_block_ptr) * scale
        else:
            q = (
                tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")
                * scale
            ).to(q_ptr.dtype.element_ty)
        for j in range(T_k):
            if EVEN_K:
                k = tl.load(k_block_ptr)
                v = tl.load(v_block_ptr)
                s = tl.dot(q, tl.trans(k))
            else:
                k = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
                v = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")
                s = tl.dot(q, tl.trans(k))
                k_offsets = j * BLOCK_K + tl.arange(0, BLOCK_K)
                s = tl.where(k_offsets[None, :] < N_k, s, -float("inf"))
            if IS_CAUSAL:
                q_idx = i * BLOCK_Q + tl.arange(0, BLOCK_Q)
                k_idx = j * BLOCK_K + tl.arange(0, BLOCK_K)
                s = tl.where(k_idx[None, :] <= q_idx[:, None], s, -float("inf"))
            m_prev = m
            if MULTI_BLOCK:
                m = tl.maximum(
                    m_prev, tl.max(s, axis=-1), propagate_nan=tl.PropagateNan.ALL
                )
            else:
                m = tl.maximum(m_prev, tl.max(s, axis=-1))
            s = tl.exp(s - m[:, None])
            factor = tl.exp(m_prev - m)
            l = factor * l + tl.sum(s, axis=-1)
            o = factor[:, None] * o
            o = tl.dot(s.to(v.dtype), v, o)
            k_block_ptr = tl.advance(k_block_ptr, (BLOCK_K, 0))
            v_block_ptr = tl.advance(v_block_ptr, (BLOCK_K, 0))
        o = o / l[:, None]
        l = m + tl.log(l)
        if EVEN_Q:
            tl.store(o_block_ptr, o.to(o_ptr.dtype.element_ty))
            tl.store(l_block_ptr, l)
        else:
            tl.store(o_block_ptr, o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))
            tl.store(l_block_ptr, l, boundary_check=(0,))


def base_tile(head_dim, n_q, n_k):
    bk = min(128, ((n_k + 15) // 16) * 16)  # ceil16(S_K)，不超 128
    bq = 64
    while bq * bk > 8192 and bq > 32:  # scores 32KB 预算
        bq //= 2
    bd = ((head_dim + 15) // 16) * 16  # ceil16，不是 next_pow2
    while bq * bd * 2 > 64 * 1024 and bq > 32:  # q tile 64KB 预算
        bq //= 2
    return bq, bk, bd


def fla_fwd(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, causal=False):
    assert Q.stride(3) == K.stride(3) == V.stride(3) == 1
    assert Q.stride(2) == K.stride(2) == V.stride(2)
    batch = Q.size(dim=0)
    N_q = Q.size(dim=1)
    N_k = K.size(dim=1)
    head = Q.size(dim=2)
    head_dim = Q.size(dim=3)

    BLOCK_Q, BLOCK_K, HEAD_DIM = base_tile(head_dim, N_q, N_k)

    O = torch.zeros_like(Q, device=Q.device)
    L = torch.zeros([batch, head, N_q], dtype=torch.float32, device=Q.device)

    scale = head_dim**-0.5

    # 带 tl.dot 的 kernel 用 AI Core 数；纯 Vector 算子用 vector_core_num
    cores = get_device_properties(0).cube_core_num

    multi_block = (N_k + BLOCK_K - 1) // BLOCK_K > 1

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
        N_q,
        N_k,
        head,
        scale,
        delta_qk=delta_qk,
        BLOCK_Q=BLOCK_Q,
        BLOCK_K=BLOCK_K,
        HEAD_DIM=HEAD_DIM,
        EVEN_Q=N_q % BLOCK_Q == 0,
        EVEN_K=N_k % BLOCK_K == 0,
        IS_CAUSAL=causal,
        MULTI_BLOCK=multi_block,
    )
    return O, L
