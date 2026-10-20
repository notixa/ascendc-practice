"""FlashAttention forward —— 从 CS336 assignment2 改写过来的落点。

迁移顺序（别跳，见 README 第 4 节）：
    1. 先原样搬过来，只做"删 GPU 参数 + 改 BLOCK 大小"两件事；
    2. 跑通单 head / causal / 无 paged cache 的最小 case，和 torch golden 对齐；
    3. 再补 GQA、变长、fp16/bf16；
    4. 最后才谈 autotune / BLOCK 搜索。

必改点（对照 README 第 3 节逐条打勾）：
    [ ] 删 num_warps / num_stages / num_ctas
    [ ] BLOCK_M / BLOCK_N 按 UB 192KB 重算，别照抄 128×128
    [ ] tl.dot 的 trans_* 参数 → 显式 tl.trans
    [ ] grid 只用 program 数，不传核数
"""

import triton
import triton.language as tl


@triton.jit
def _fa_fwd_kernel(
    Q, K, V, O,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    """参考骨架：每题一个 program 处理 BLOCK_M 行 Q。"""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    # online softmax 状态
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    q = tl.load(Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk,
                mask=offs_m[:, None] < N_CTX, other=0.0)

    for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        n_mask = (start_n + offs_n) < N_CTX
        k = tl.load(K + k_offset + (start_n + offs_n)[:, None] * stride_kn + offs_d[None, :] * stride_kk,
                    mask=n_mask[:, None], other=0.0)
        # Ascend 上 tl.dot 的转置参数支持受限，这里显式转置
        qk = tl.dot(q, tl.trans(k)) * sm_scale

        causal = offs_m[:, None] >= (start_n + offs_n)[None, :]
        qk = tl.where(causal & n_mask[None, :], qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v = tl.load(V + v_offset + (start_n + offs_n)[:, None] * stride_vk + offs_d[None, :] * stride_vn,
                    mask=n_mask[:, None], other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(O + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on,
             acc.to(O.dtype.element_ty), mask=offs_m[:, None] < N_CTX)


def attention_forward(q, k, v, causal=True, sm_scale=None):
    """q/k/v: (Z, H, N_CTX, HEAD_DIM)，返回 (Z, H, N_CTX, HEAD_DIM)。"""
    Z, H, N_CTX, HEAD_DIM = q.shape
    assert causal, "当前骨架只实现了 causal"
    sm_scale = sm_scale or (HEAD_DIM ** -0.5)

    o = q.new_empty((Z, H, N_CTX, HEAD_DIM))
    BLOCK_M, BLOCK_N = 64, 64
    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H)
    _fa_fwd_kernel[grid](
        q, k, v, o,
        *q.stride(), *k.stride(), *v.stride(), *o.stride(),
        Z, H, N_CTX, sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM,
    )
    return o


if __name__ == "__main__":
    try:
        import torch
    except ModuleNotFoundError:
        raise SystemExit("没装 torch：见 kernels/vector_add.py 里的安装提示")
    torch.manual_seed(0)
    Z, H, N_CTX, D = 1, 2, 256, 64
    q = torch.randn(Z, H, N_CTX, D, dtype=torch.float16, device="cpu")
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # torch golden（causal）
    scale = D ** -0.5
    s = (q.float() @ k.float().transpose(-1, -2)) * scale
    mask = torch.tril(torch.ones(N_CTX, N_CTX, dtype=torch.bool))
    s = s.masked_fill(~mask, float("-inf"))
    p = torch.softmax(s, dim=-1).to(torch.float16)
    golden = p.float() @ v.float()

    out = attention_forward(q, k, v)
    err = (out.float() - golden).abs().max().item()
    print("max abs err =", err)
    print("fa fwd:", "PASS" if err < 1e-2 else "FAIL")
