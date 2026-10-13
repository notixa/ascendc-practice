import os
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
import sys
import subprocess
from pathlib import Path
from setuptools import setup

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension

CURRENT_DIR = Path(__file__).parent.resolve()
BUILD_DIR = CURRENT_DIR / "build"

def get_cann_path():
    candidates = [
        os.getenv("ASCEND_HOME_PATH"),
        os.getenv("ASCEND_TOOLKIT_HOME"),
        "/usr/local/Ascend/ascend-toolkit/latest",
        str(Path.home() / "Ascend/ascend-toolkit/latest"),
    ]
    for path in candidates:
        if path and os.path.isdir(path):
            return path
    return candidates[2]

ASCEND_TOOLKIT_PATH = get_cann_path()
SOC_VERSION = os.getenv("SOC_VERSION", "Ascend310B1")

def get_kernel_lib():
    candidates = [
        BUILD_DIR / "lib" / "libflash_attention_kernel.a",
        BUILD_DIR / "libflash_attention_kernel.a",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return str(candidates[0])

def build_kernel_with_cmake():
    """Build Ascend C kernel into static library using CMake and CANN toolchain."""
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    cmake_cmd = [
        "cmake",
        f"-S{CURRENT_DIR}",
        f"-B{BUILD_DIR}",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DSOC_VERSION={SOC_VERSION}",
        f"-DASCEND_CANN_PACKAGE_PATH={ASCEND_TOOLKIT_PATH}",
        "-DASCEND_KERNEL_LAUNCH_ONLY=ON",
    ]
    build_cmd = ["cmake", "--build", str(BUILD_DIR), "-j4"]

    print(f"\n[setup.py] >>> Configuring Ascend C kernel via CMake (SOC_VERSION={SOC_VERSION})...")
    subprocess.check_call(cmake_cmd)
    print("[setup.py] >>> Building libflash_attention_kernel.a with ccec...")
    subprocess.check_call(build_cmd)
    print(f"[setup.py] >>> Ascend C kernel built successfully at {get_kernel_lib()}!\n")

class CustomBuildExtension(BuildExtension):
    def run(self):
        if (CURRENT_DIR / "CMakeLists.txt").exists():
            build_kernel_with_cmake()
        for ext in self.extensions:
            ext.extra_objects = [get_kernel_lib()]
        super().run()

include_dirs = [
    str(CURRENT_DIR / "csrc"),
    os.path.join(ASCEND_TOOLKIT_PATH, "include"),
    os.path.join(ASCEND_TOOLKIT_PATH, "include", "tikcpp"),
    os.path.join(ASCEND_TOOLKIT_PATH, "include", "tikcpp", "tikcfw"),
]

cann_lib64 = os.path.join(ASCEND_TOOLKIT_PATH, "lib64")
cann_stub64 = os.path.join(ASCEND_TOOLKIT_PATH, "runtime", "lib64", "stub")
library_dirs = [cann_lib64]
extra_link_args = [f"-Wl,-rpath,{cann_lib64}"]
if os.path.isdir(cann_stub64):
    library_dirs.append(cann_stub64)
    extra_link_args.append(f"-Wl,-rpath,{cann_stub64}")
libraries = [
    "ascendcl",
    "runtime",
]

try:
    import torch_npu
    torch_npu_dir = os.path.dirname(torch_npu.__file__)
    npu_inc = os.path.join(torch_npu_dir, "include")
    npu_lib = os.path.join(torch_npu_dir, "lib")
    if os.path.exists(npu_inc):
        include_dirs.append(npu_inc)
    if os.path.exists(npu_lib):
        library_dirs.append(npu_lib)
        libraries.append("torch_npu")
        extra_link_args.append(f"-Wl,-rpath,{npu_lib}")
    print(f"[setup.py] Found torch_npu at {torch_npu_dir}, enabled NPU integration.")
except ImportError:
    print("[setup.py] torch_npu not found, building with Pure ACL fallback mode.")

ext_modules = [
    CppExtension(
        name="flash_attention_npu",
        sources=["csrc/flash_attention_binding.cpp"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_objects=[get_kernel_lib()],
        extra_compile_args=["-O3", "-std=c++17"],
        extra_link_args=extra_link_args,
    )
]

setup(
    name="flash_attention_npu",
    version="0.1.0",
    author="Ascend C Developer",
    description="Custom FlashAttention FP16 PyTorch Extension Operator for Ascend C",
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExtension},
)
