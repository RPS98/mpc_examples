"""Pytest configuration: make the in-tree packages importable."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "python" / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
