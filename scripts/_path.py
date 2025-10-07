from __future__ import annotations

import sys
from pathlib import Path


def project_root() -> Path:
    """Return repository root and ensure it is importable."""
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


# Prime the path once on import so downstream scripts can simply import this module.
project_root()


__all__ = ["project_root"]
