"""最小类型存根：`triton.extension.buffer.language`（惯用别名 `bl`）。

真实模块的 `__init__.py` 有完整的 `__all__` re-export，静态分析本来能解析；
但它的 `core.py` 里 `address_space.to_ir()` 的签名引用了 `ir.builder`，
而 `ir` 来自无存根的 `triton._C.libtriton`，会让 `address_space` 相关名字退化成
Unknown。这里补一份干净的声明，保证 `bl.alloc(...)` 系列能补全。

运行时行为完全由真实实现决定，本存根不参与。
"""

from typing import Any, List, Optional

import triton.language as tl

class bl_dtype(tl.dtype): ...
class buffer_type(tl.dtype): ...

class address_space:
    """Buffer 的地址空间。具体实例由 accelerate address space group 提供。"""
    def to_ir(self, builder: Any) -> Any: ...

class buffer:
    """一块本地内存区域（UB / L1 / L0A / L0B / L0C 之一）。"""
    def __init__(
        self,
        element_ty: tl.dtype,
        shape: List[Any],
        space: Optional[address_space] = ...,
        strides: Optional[List[Any]] = ...,
    ) -> None: ...
    def to_ir(self, builder: Any) -> Any: ...

def builtin(fn: Any) -> Any: ...
def is_builtin(fn: Any) -> bool: ...

def alloc(
    etype: tl.dtype,
    shape: List[tl.constexpr],
    _address_space: Optional[address_space] = ...,
    is_mem_unique: bool = False,
    _semantic: Any = ...,
) -> buffer:
    """在本地内存里分配一块区域。

    :param etype: 元素类型
    :param shape: 形状（constexpr 列表）
    :param _address_space: 后端特定的地址空间，通常传
        ``al.ascend_address_space.UB`` / ``.L1`` / ``.L0A`` / ``.L0B`` / ``.L0C``
    :param is_mem_unique: 是否独占（不复用）该 buffer
    """
    ...

def to_buffer(
    tensor: tl.tensor,
    space: Optional[address_space] = ...,
    bind_buffer: Optional[buffer] = ...,
    _semantic: Any = ...,
) -> buffer: ...

def to_tensor(
    memref: buffer,
    writable: bool = True,
    target_shape: Any = ...,
    _semantic: Any = ...,
) -> tl.tensor: ...

def subview(
    src: buffer,
    offsets: List[tl.constexpr],
    sizes: List[tl.constexpr],
    strides: List[tl.constexpr],
    _semantic: Any = ...,
) -> buffer:
    """从源 buffer 切一个子视图（做多槽/分片用）。"""
    ...
