import warnings

import torch
import torch_npu
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
from torch_npu.npu import get_device_properties

_DEVICE_PROPS = None


def _get_core_counts():
    """缓存 device properties：每次调用都查驱动会带来可观的 host 开销。"""
    global _DEVICE_PROPS
    if _DEVICE_PROPS is None:
        _DEVICE_PROPS = get_device_properties(0)
    return _DEVICE_PROPS.cube_core_num, _DEVICE_PROPS.vector_core_num


@triton.jit
def flash_decoding_s1_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    mid_o_ptr,
    mid_lse_ptr,
    stride_qb,
    stride_qh,
    stride_qg,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_mid_og,
    stride_mid_od,
    stride_mid_lb,
    stride_mid_lh,
    stride_mid_ls,
    stride_mid_lg,
    scale,
    batch_num,
    group: tl.constexpr,
    head_kv,
    seq_len,
    HEAD_DIM: tl.constexpr,
    SPLIT_S: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    prog_num = tl.num_programs(0)
    S_BLOCKS = tl.cdiv(seq_len, SPLIT_S)
    task_num = batch_num * head_kv * S_BLOCKS
    for taskId in range(pid, task_num, prog_num):
        b = taskId // (head_kv * S_BLOCKS)
        h = (taskId % (head_kv * S_BLOCKS)) // S_BLOCKS
        split_id = (taskId % (head_kv * S_BLOCKS)) % S_BLOCKS

        q_base = q_ptr + b * stride_qb + h * stride_qh
        k_base = k_ptr + b * stride_kb + h * stride_kh
        v_base = v_ptr + b * stride_vb + h * stride_vh
        mid_o_base = (
            mid_o_ptr
            + b * stride_mid_ob
            + h * stride_mid_oh
            + split_id * stride_mid_os
        )
        mid_lse_base = (
            mid_lse_ptr
            + b * stride_mid_lb
            + h * stride_mid_lh
            + split_id * stride_mid_ls
        )

        q_block_ptr = tl.make_block_ptr(
            q_base,
            (group, HEAD_DIM),
            strides=(stride_qg, stride_qd),
            offsets=(0, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        mid_o_block_ptr = tl.make_block_ptr(
            mid_o_base,
            (group, HEAD_DIM),
            strides=(stride_mid_og, stride_mid_od),
            offsets=(0, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        mid_lse_block_ptr = tl.make_block_ptr(
            mid_lse_base,
            (group,),
            strides=(stride_mid_lg,),
            offsets=(0,),
            block_shape=(BLOCK_M,),
            order=(0,),
        )

        m = tl.full([BLOCK_M], -6.0e4, dtype=tl.float32)
        lse = tl.full([BLOCK_M], 0.0, dtype=tl.float32)
        acc = tl.full([BLOCK_M, HEAD_DIM], 0.0, dtype=tl.float32)

        q = tl.load(q_block_ptr, boundary_check=(0,), padding_option="zero")

        kv_start_r = split_id * SPLIT_S
        kv_end_r = tl.minimum((split_id + 1) * SPLIT_S, seq_len)

        kv_offsets = tl.arange(0, BLOCK_S)

        for kv_row in range(kv_start_r, kv_end_r, BLOCK_S):
            k_block_ptr = tl.make_block_ptr(
                k_base,
                (seq_len, HEAD_DIM),
                strides=(stride_ks, stride_kd),
                offsets=(kv_row, 0),
                block_shape=(BLOCK_S, HEAD_DIM),
                order=(1, 0),
            )
            v_block_ptr = tl.make_block_ptr(
                v_base,
                (seq_len, HEAD_DIM),
                strides=(stride_vs, stride_vd),
                offsets=(kv_row, 0),
                block_shape=(BLOCK_S, HEAD_DIM),
                order=(1, 0),
            )

            k = tl.load(k_block_ptr, boundary_check=(0,), padding_option="zero")
            score = tl.dot(q, tl.trans(k)).to(tl.float16)

            score = score * scale.to(tl.float16)
            score = tl.where(
                (kv_offsets + kv_row)[None, :] < kv_end_r, score, -6.0e4
            ).to(tl.float16)
            m_prev = m
            m = tl.maximum(m_prev, tl.max(score, -1))
            p = tl.math.exp2(score - m[:, None])
            alpha = tl.math.exp2(m_prev - m)
            lse = lse * alpha + tl.sum(p, -1)

            v = tl.load(v_block_ptr, boundary_check=(0,), padding_option="zero")
            # exp2 可能把 fp16 提升到 fp32；tl.dot 要求两边同 dtype。
            otmp = tl.dot(p.to(v.dtype), v)
            acc = acc * alpha[:, None] + otmp

        mid_o = acc / lse[:, None]
        mid_lse = m + tl.math.log2(lse)

        tl.store(mid_o_block_ptr, mid_o.to(tl.float16), boundary_check=(0,))
        tl.store(mid_lse_block_ptr, mid_lse, boundary_check=(0,))


@triton.jit
def flash_decoding_s2_kernel(
    mid_o_ptr,
    mid_lse_ptr,
    out_ptr,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_mid_og,
    stride_mid_od,
    stride_mid_lb,
    stride_mid_lh,
    stride_mid_ls,
    stride_mid_lg,
    stride_ob,
    stride_oh,
    stride_og,
    stride_od,
    batch,
    head_kv,
    group: tl.constexpr,
    BLOCK_G: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_SPLITS_PAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    prog_num = tl.num_programs(0)
    task_num = batch * head_kv

    for taskIdx in range(pid, task_num, prog_num):
        b = taskIdx // head_kv
        h = taskIdx % head_kv
        o_base = out_ptr + b * stride_ob + h * stride_oh
        lse_base = mid_lse_ptr + b * stride_mid_lb + h * stride_mid_lh
        lse_block_ptr = tl.make_block_ptr(
            lse_base,
            (NUM_SPLITS, group),
            strides=(stride_mid_ls, stride_mid_lg),
            offsets=(0, 0),
            block_shape=(NUM_SPLITS_PAD, BLOCK_G),
            order=(1, 0),
        )
        o_block_ptr = tl.make_block_ptr(
            o_base,
            (group, HEAD_DIM),
            strides=(stride_og, stride_od),
            offsets=(0, 0),
            block_shape=(BLOCK_G, HEAD_DIM),
            order=(1, 0),
        )

        lse = tl.trans(
            tl.load(lse_block_ptr, boundary_check=(0, 1), padding_option="zero")
        )
        lse_split_arange = tl.arange(0, NUM_SPLITS_PAD)
        lse_mask = lse_split_arange[None, :] < NUM_SPLITS
        lse = tl.where(lse_mask, lse, -6.0e4)
        lse_max = tl.max(lse, -1)
        A = tl.math.exp2(lse - lse_max[:, None])
        deno = tl.sum(A, -1)
        W = A / deno[:, None]
        acc = tl.full([BLOCK_G, HEAD_DIM], 0.0, dtype=tl.float32)
        for s in range(0, NUM_SPLITS):
            mid_o_base = (
                mid_o_ptr + b * stride_mid_ob + h * stride_mid_oh + s * stride_mid_os
            )
            mid_o_block_ptr = tl.make_block_ptr(
                mid_o_base,
                (group, HEAD_DIM),
                strides=(stride_mid_og, stride_mid_od),
                offsets=(0, 0),
                block_shape=(BLOCK_G, HEAD_DIM),
                order=(1, 0),
            )
            mid_o = tl.load(mid_o_block_ptr, boundary_check=(0,), padding_option="zero")
            w = al.extract_slice(W, offsets=(0, s), sizes=(BLOCK_G, 1), strides=(1, 1))
            w = tl.reshape(w, (BLOCK_G,))
            acc = acc + mid_o * w[:, None]

        tl.store(o_block_ptr, acc.to(tl.float16), boundary_check=(0,))


def flash_decoding(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    split_s: int = 512,
    block_s: int = 128,
):
    """Split-KV FlashDecoding。

    Parameters
    ----------
    split_s:
        每个 split 负责的 KV 长度，``NUM_SPLITS = ceil(seq_len / split_s)``。
        必须 >= ``block_s``。不要求整除 ``block_s``：kernel 的 mask 是按
        ``kv_end_r`` 做的，非整除也正确；但对齐到 ``block_s`` 时边界更整齐，
        通常更快。
    block_s:
        内层 KV tile 大小，必须是 2 的幂（64/128/256）。
    """
    if block_s <= 0 or (block_s & (block_s - 1)) != 0:
        raise ValueError(f"block_s 必须是 2 的幂，收到 {block_s}")
    if split_s < block_s:
        raise ValueError(f"split_s ({split_s}) 必须 >= block_s ({block_s})")
    if split_s % block_s != 0:
        warnings.warn(
            f"split_s={split_s} 不是 block_s={block_s} 的整数倍：split 边界不会"
            "对齐到 BLOCK_S，正确性由 kv_end_r mask 保证，但会多读一些被 mask 的"
            "行，通常更慢。",
            stacklevel=2,
        )

    assert Q.dtype == torch.float16
    assert Q.dim() == 4 and K.dim() == 4 and V.dim() == 4
    assert Q.shape[0] == K.shape[0] == V.shape[0]
    assert Q.shape[1] == K.shape[1] == V.shape[1]
    assert Q.shape[3] == K.shape[3] == V.shape[3]
    assert K.shape[2] == V.shape[2]
    assert Q.shape[2] > 0

    batch_num = Q.size(0)
    head_kv = Q.size(1)
    group = Q.size(2)
    dim = Q.size(3)
    seq_len = K.size(2)

    scale = dim**-0.5 * 1.44269504
    SPLIT_S = split_s
    BLOCK_S = block_s
    NUM_SPLITS = triton.cdiv(seq_len, SPLIT_S)
    NUM_SPLITS_PAD = triton.next_power_of_2(NUM_SPLITS)
    BLOCK_M = max(16, triton.next_power_of_2(group))
    BLOCK_G = triton.next_power_of_2(group)

    cube_cores, vector_cores = _get_core_counts()
    task_num_p1 = batch_num * head_kv * NUM_SPLITS
    task_num_p2 = batch_num * head_kv
    grid_p1 = (min(task_num_p1, cube_cores),)
    grid_p2 = (min(task_num_p2, vector_cores),)

    mid_o = torch.empty(
        [batch_num, head_kv, NUM_SPLITS, group, dim],
        dtype=torch.float16,
        device=Q.device,
    )

    mid_lse = torch.empty(
        [batch_num, head_kv, NUM_SPLITS, group],
        dtype=torch.float32,
        device=Q.device,
    )

    out = torch.empty(
        [batch_num, head_kv, group, dim], dtype=torch.float16, device=Q.device
    )

    flash_decoding_s1_kernel[grid_p1](
        Q,
        K,
        V,
        mid_o,
        mid_lse,
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
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        mid_o.stride(3),
        mid_o.stride(4),
        mid_lse.stride(0),
        mid_lse.stride(1),
        mid_lse.stride(2),
        mid_lse.stride(3),
        scale,
        batch_num,
        group,
        head_kv,
        seq_len,
        HEAD_DIM=dim,
        SPLIT_S=SPLIT_S,
        BLOCK_S=BLOCK_S,
        BLOCK_M=BLOCK_M,
    )

    flash_decoding_s2_kernel[grid_p2](
        mid_o,
        mid_lse,
        out,
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        mid_o.stride(3),
        mid_o.stride(4),
        mid_lse.stride(0),
        mid_lse.stride(1),
        mid_lse.stride(2),
        mid_lse.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        batch_num,
        head_kv,
        group,
        BLOCK_G,
        NUM_SPLITS,
        NUM_SPLITS_PAD,
        HEAD_DIM=dim,
    )
    return out
