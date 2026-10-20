"""冒烟测试：有卡就真跑，没卡自动 skip（这样在 x86 无 NPU 机器上 `pytest` 也是绿的）。"""

import importlib.util

import pytest

HAS_TRITON = importlib.util.find_spec("triton") is not None


def _has_npu():
    try:
        import torch

        return hasattr(torch, "npu") and torch.npu.is_available()
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not HAS_TRITON, reason="triton 未安装（先 uv sync）")


def test_triton_importable():
    import triton

    assert triton.__version__


@pytest.mark.skipif(not _has_npu(), reason="当前机器没有可用 NPU")
def test_vector_add_on_npu():
    import torch

    from kernels.vector_add import _add_kernel
    import triton

    n = 1 << 16
    x = torch.rand(n, dtype=torch.float32, device="npu")
    y = torch.rand(n, dtype=torch.float32, device="npu")
    out = torch.empty_like(x)
    BLOCK = 1024
    _add_kernel[(triton.cdiv(n, BLOCK),)](x, y, out, n, BLOCK_SIZE=BLOCK)
    torch.npu.synchronize()
    assert torch.allclose(out, x + y, atol=1e-5, rtol=1e-5)
