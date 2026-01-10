import importlib.util
import os
from types import ModuleType


def _load_module(module_name: str, path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load torch kernel module at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_GATHER_PATH = os.path.join(_BASE_DIR, "cuda", "gather_kernel.py")
_SCATTER_PATH = os.path.join(_BASE_DIR, "cuda", "scatter_kernel.py")
_SCATTER_GATHER_PATH = os.path.join(_BASE_DIR, "cuda", "scatter_gather_kernel.py")

_gather_module = _load_module("sige_cuda_gather_kernel_torch", _GATHER_PATH)
_scatter_module = _load_module("sige_cuda_scatter_kernel_torch", _SCATTER_PATH)
_scatter_gather_module = _load_module("sige_cuda_scatter_gather_kernel_torch", _SCATTER_GATHER_PATH)

gather = _gather_module.gather
scatter = _scatter_module.scatter
scatter_with_block_residual = _scatter_module.scatter_with_block_residual
scatter_gather = _scatter_gather_module.scatter_gather
get_scatter_map = _scatter_gather_module.get_scatter_map

__all__ = [
    "gather",
    "scatter",
    "scatter_with_block_residual",
    "scatter_gather",
    "get_scatter_map",
]
