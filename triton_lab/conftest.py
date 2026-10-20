import sys
from pathlib import Path

# 让 `pytest` 能 import 项目根下的 kernels/ 和 env_report
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
