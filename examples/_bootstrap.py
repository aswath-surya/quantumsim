"""Make `import proxysim` work whether or not the package is pip-installed.

If you ran `pip install -e .` this is a no-op; otherwise it puts the repo root
on sys.path so the examples still run from a fresh checkout.
"""

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

RESULTS_DIR = os.path.join(_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
