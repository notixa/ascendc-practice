"""环境体检：确认 triton-ascend 装对了、后端是 ascend、target 正常。

用法：
    uv run python env_report.py

判读标准（第一次在 910B 机器上跑）：
    - triton version:  3.2.x / 3.6.0（不是上游 3.7 之类）
    - triton 路径:     在 triton_lab/.venv 里
    - ascend 后端:     存在
    - target:          能打印出 ATLAS/A2 之类的型号，而不是报错
"""

import importlib
import os
import sys
import warnings
from pathlib import Path

# triton-ascend 3.6.0 的 cann extension 在导入时会刷一堆 FutureWarning，先压掉
warnings.filterwarnings("ignore", category=FutureWarning)


def line(k, v):
    print(f"{k:<22}{v}")


def main():
    line("python", sys.version.split()[0])
    line("interpreter", sys.executable)

    print("-" * 60)

    try:
        import triton

        line("triton version", triton.__version__)
        line("triton path", Path(triton.__file__).parent)
        line("triton dist", triton.__file__)
    except Exception as e:  # noqa: BLE001
        line("triton", f"导入失败: {e!r}")
        print("\n→ 先 `uv sync`，或检查是不是装到了别的 venv 里。")
        return 1

    # 后端是否存在（上游 triton 只有 nvidia/amd）
    try:
        backend_dir = Path(triton.__file__).parent / "backends"
        backends = (
            sorted(p.name for p in backend_dir.iterdir() if p.is_dir())
            if backend_dir.is_dir()
            else []
        )
        line("backends", backends or "(未找到 backends 目录)")
        if "ascend" not in backends:
            print(
                "\n⚠ 没有 ascend 后端：装的是上游 GPU 版 triton，不是 triton-ascend。"
            )
    except Exception as e:  # noqa: BLE001
        line("backends", f"检查失败: {e!r}")

    # 当前 target（需要能初始化 driver）
    try:
        from triton.runtime import driver

        target = driver.active.get_current_target()
        line("current target", target)
    except Exception as e:  # noqa: BLE001
        line("current target", f"获取失败（无设备时正常）: {type(e).__name__}: {e}")

    print("-" * 60)

    # 设备侧信息
    try:
        from triton.runtime import driver

        props = driver.active.utils.get_device_properties(0)
        line("device properties", props)
    except Exception as e:  # noqa: BLE001
        line("device properties", f"不可用: {type(e).__name__}: {e}")

    # CANN / torch 侧
    line("ASCEND_HOME_PATH", os.environ.get("ASCEND_HOME_PATH", "(未设置)"))
    for mod in ("torch", "torch_npu"):
        try:
            m = importlib.import_module(mod)
            line(f"{mod} version", getattr(m, "__version__", "?"))
        except Exception as e:  # noqa: BLE001
            line(f"{mod} version", f"未安装/导入失败: {type(e).__name__}")

    try:
        import torch

        if hasattr(torch, "npu"):
            line("torch.npu.available", torch.npu.is_available())
            if torch.npu.is_available():
                line("npu device", torch.npu.get_device_name(0))
    except Exception as e:  # noqa: BLE001
        line("torch.npu", f"不可用: {type(e).__name__}")

    print("-" * 60)
    print("下一步：uv run python kernels/vector_add.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
