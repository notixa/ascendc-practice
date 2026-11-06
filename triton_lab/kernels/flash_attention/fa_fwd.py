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

_TILES = [
    (64, 512),
    (64, 256),
    (96, 128),
    (64, 128),
    (32, 128),
    (32, 64),
]


def _pick_tile(n_q, n_k):
    # (64,256)：一次迭代覆盖 256 列 KV，实测在整除形状上最快。
    # BKV=256 的尾块 bug 未修（N 非整除时 aicore 507015），所以只在整除时使用。
    # (64,512)：BKV=512 是 UB 允许的最大值，实测比 (64,256) 快 1.42x。
    # 守卫 n_k > 512：实测 (64,512) 在 N_k == BLOCK_K == 512 且 non-causal 时
    #   触发 aicore 507015（单次迭代 + 末尾 advance 越过 block_ptr 边界）。
    #   N_k >= 1024 的多迭代情形正常（1024/2048/4096 non-causal 均通过）。
    # BKV=512 同样有尾块 bug（N 非整除时 aicore 507015），只在整除时使用。
    if n_k % 512 == 0 and n_k > 512:
        return 64, 512
    if n_k % 256 == 0:
        return 64, 256
    return 32, 128


@triton.jit
def fla_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
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
        mi = tl.full([BLOCK_Q], -1e38, dtype=tl.float32)
        li = tl.zeros([BLOCK_Q], dtype=tl.float32)
        acc = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)

        q = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")

        q = q * scale.to(tl.float16)

        q_start_r = i * BLOCK_Q

        end = N_k
        if IS_CAUSAL:
            end = tl.minimum(q_start_r + BLOCK_Q, N_k) + delta_qk

        q_idx = q_start_r + tl.arange(0, BLOCK_Q)
        for j in range(0, end, BLOCK_K):
            k_block_ptr = tl.make_block_ptr(
                k_base,
                (N_k, HEAD_DIM),
                (stride_kk, stride_kd),
                (j, 0),
                (BLOCK_K, HEAD_DIM),
                (1, 0),
            )
            v_block_ptr = tl.make_block_ptr(
                v_base,
                (N_k, HEAD_DIM),
                (stride_vk, stride_vd),
                (j, 0),
                (BLOCK_K, HEAD_DIM),
                (1, 0),
            )
            k = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
            v = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")
            s = tl.dot(q, tl.trans(k)).to(tl.float16)

            k_idx = j + tl.arange(0, BLOCK_K)
            if IS_CAUSAL:
                s = tl.where(
                    (k_idx[None, :] < N_k)
                    & (k_idx[None, :] <= q_idx[:, None] + delta_qk),
                    s,
                    -6.0e4,
                )
            else:
                s = tl.where(k_idx[None, :] < N_k, s, -6.0e4)
            m_prev = mi
            mi = tl.maximum(m_prev, tl.max(s, axis=-1))
            s = tl.exp((s - mi[:, None].to(tl.float16)).to(tl.float16))
            factor = tl.exp(m_prev - mi)
            acc = factor[:, None] * acc
            pv = tl.dot(s, v)
            acc = acc + pv
            li = factor * li + tl.sum(s.to(tl.float32), axis=-1)

        acc = acc / li[:, None]
        tl.store(o_block_ptr, acc.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))


def fla_fwd(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, causal=False):
    assert Q.stride(3) == K.stride(3) == V.stride(3) == 1
    assert Q.stride(2) == K.stride(2) == V.stride(2)
    batch = Q.size(dim=0)
    N_q = Q.size(dim=1)
    N_k = K.size(dim=1)
    head = Q.size(dim=2)
    head_dim = Q.size(dim=3)

    O = torch.zeros_like(Q, device=Q.device)

    scale = head_dim**-0.5

    cores = get_device_properties(0).cube_core_num

    delta_qk = N_k - N_q

    BLOCK_Q, BLOCK_K = _pick_tile(N_q, N_k)
    grid = (min(batch * head * triton.cdiv(N_q, BLOCK_Q), cores),)
    fla_fwd_kernel[grid](
        Q,
        K,
        V,
        O,
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
        batch,
        head,
        scale,
        N_q=N_q,
        N_k=N_k,
        delta_qk=delta_qk,
        BLOCK_Q=BLOCK_Q,
        BLOCK_K=BLOCK_K,
        HEAD_DIM=head_dim,
        IS_CAUSAL=causal,
    )
    return O
