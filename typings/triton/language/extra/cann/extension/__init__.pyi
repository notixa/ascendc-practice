"""最小类型存根：`triton.language.extra.cann.extension`（惯用别名 `al`）。

为什么需要这个存根
------------------
真实的 `extension/__init__.py` 第 2 行是：

    from triton._C.libtriton.ascend import ir as _ascend_ir

`triton._C.libtriton.ascend` 是编译好的 .so，**没有任何 .pyi 存根**，而且整个
triton 包里也没有 `py.typed` 标记。于是 pyright 把 `_ascend_ir` 判成 Unknown，
凡是派生自它的名字（CORE / PIPE / MODE / EVENT_ID / ascend_address_space /
FixpipeDMAMode ...）在 IDE 里全部无法补全，甚至会把同包里纯 Python 的 `scope`
一起连带成 Unknown。

这份存根只声明我们实际用到的 API，让补全和跳转可用。它是**类型检查用的**，
不参与运行，也不影响 Triton 编译。

枚举成员的真实值在 .so 里，这里用 `Any` 声明 —— 能补出名字，但成员语义是松的，
这是当前唯一可行且够用的折中。
"""

from typing import Any, List, Optional, Union

import triton.language as tl

# --------------------------------------------------------------------------
# 枚举（真实定义来自 triton._C.libtriton.ascend.ir）
# --------------------------------------------------------------------------
class CORE:
    VECTOR: Any
    CUBE: Any
    CUBE_OR_VECTOR: Any
    CUBE_AND_VECTOR: Any

class MODE:
    SIMD: Any
    SIMT: Any
    MIX: Any

class PIPE:
    PIPE_S: Any
    PIPE_V: Any
    PIPE_MTE1: Any
    PIPE_MTE2: Any
    PIPE_MTE3: Any
    PIPE_FIX: Any

class IteratorType:
    Parallel: Any
    Broadcast: Any

class EVENT_ID: ...
class SYNC_HINT: ...
class SYNC_IN_VF: ...

class FixpipeDMAMode:
    NZ2ND: Any
    ND2NZ: Any

class FixpipeDualDstMode:
    NO_DUAL: Any
    DUAL: Any

class FixpipePreQuantMode: ...
class FixpipePreReluMode: ...

ascend_address_space: Any
int64: Any

# --------------------------------------------------------------------------
# 作用域：显式指定代码块跑在 cube 还是 vector 核上
# --------------------------------------------------------------------------
class scope:
    def __init__(self, core_mode: str, **kwargs: Any) -> None: ...
    def __enter__(self) -> "scope": ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool: ...

# --------------------------------------------------------------------------
# 跨核同步（semaphore 语义：set 让 per-event 计数 +1，wait 在计数 >0 时通过并 -1）
# 必须成对出现在匹配的 scope 内；event_id 取值 [0, 15]
# --------------------------------------------------------------------------
def sync_block_set(
    sender: str,
    receiver: str,
    event_id: int,
    sender_pipe: Optional[Any] = ...,
    receiver_pipe: Optional[Any] = ...,
) -> None: ...

def sync_block_wait(
    sender: str,
    receiver: str,
    event_id: int,
    sender_pipe: Optional[Any] = ...,
    receiver_pipe: Optional[Any] = ...,
) -> None: ...

def sync_block_all(mode: str, event_id: int) -> None: ...
    # mode: "all_cube" | "all_vector" | "all" | "all_sub_vector"

# --------------------------------------------------------------------------
# 流水 / 缓冲
# --------------------------------------------------------------------------
def multibuffer(src: tl.tensor, size: int) -> tl.tensor: ...
    # size 只支持 2（源码里有 assert buffer_size == 2）

def compile_hint(*args: Any, **kwargs: Any) -> None: ...

def copy(src: Union[tl.tensor, Any], dst: Union[tl.tensor, Any]) -> None: ...
    # UB -> UB 或 UB -> L1

def copy_from_ub_to_l1(src: Any, dst: Any) -> None: ...
    # 已废弃；且仅 Ascend910_95 支持

def fixpipe(
    src: tl.tensor,
    dst: Optional[Any] = ...,
    dma_mode: Any = ...,
    dual_dst_mode: Any = ...,
) -> Optional[tl.tensor]: ...
    # L0C -> UB，仅 Ascend910_95 series 支持

# --------------------------------------------------------------------------
# slice / 元素级
# --------------------------------------------------------------------------
def insert_slice(dst: tl.tensor, src: tl.tensor, offsets: List[int]) -> tl.tensor: ...
def extract_slice(
    src: tl.tensor, offsets: List[int], sizes: List[int], strides: List[int]
) -> tl.tensor: ...
def get_element(src: tl.tensor, index: List[int]) -> tl.tensor: ...
def sort(x: tl.tensor, dim: int = ..., descending: bool = ...) -> tl.tensor: ...
def flip(x: tl.tensor, dim: int = ...) -> tl.tensor: ...
def cast(x: tl.tensor, dtype: Any) -> tl.tensor: ...

# --------------------------------------------------------------------------
# 子核 / 调试
# --------------------------------------------------------------------------
def sub_vec_id() -> tl.tensor: ...
def sub_vec_num() -> tl.tensor: ...
def debug_barrier() -> None: ...

# --------------------------------------------------------------------------
# 自定义算子注册
# --------------------------------------------------------------------------
def custom(*args: Any, **kwargs: Any) -> Any: ...
def custom_semantic(*args: Any, **kwargs: Any) -> Any: ...
def register_custom_op(*args: Any, **kwargs: Any) -> Any: ...
class SyncEventSlot: ...

def builtin(fn: Any) -> Any: ...
def is_builtin(fn: Any) -> bool: ...

# parallel / math 转出
def parallel(*args: Any, **kwargs: Any) -> Any: ...
def atan2(*args: Any, **kwargs: Any) -> Any: ...
def isfinited(*args: Any, **kwargs: Any) -> Any: ...
def finitef(*args: Any, **kwargs: Any) -> Any: ...
def conv1d(*args: Any, **kwargs: Any) -> Any: ...
def conv2d(*args: Any, **kwargs: Any) -> Any: ...
def dot(*args: Any, **kwargs: Any) -> Any: ...
