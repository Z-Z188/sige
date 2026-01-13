from __future__ import annotations

import importlib
from types import ModuleType
from typing import Optional

_EXT: Optional[ModuleType] = None


def get_extension(*, build_if_missing: bool = True) -> ModuleType:
    """
    Return the loaded `sige3d_cuda` extension module.

    - If already loaded, returns cached module.
    - If missing and `build_if_missing=True`, attempts JIT build via `build_sige3d_cuda()`.
    """
    global _EXT  # noqa: PLW0603
    if _EXT is not None:
        return _EXT

    try:
        _EXT = importlib.import_module("sige3d_cuda")
        return _EXT
    except Exception:
        if not build_if_missing:
            raise
        from .build import build_sige3d_cuda

        _EXT = build_sige3d_cuda(verbose=False)
        return _EXT