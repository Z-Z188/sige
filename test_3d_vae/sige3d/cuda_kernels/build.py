from __future__ import annotations

from pathlib import Path


def build_sige3d_cuda(*, verbose: bool = False):
    """
    Build and load the `sige3d_cuda` torch extension.

    This requires:
      - PyTorch with CUDA available
      - a working CUDA toolchain (nvcc)
    """
    import torch
    from torch.utils.cpp_extension import load

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; cannot build sige3d_cuda.")

    this_dir = Path(__file__).resolve().parent
    sources = [
        str(this_dir / "pybind_cuda.cpp"),
        str(this_dir / "ops_2d.cu"),
        str(this_dir / "ops_3d.cu"),
        str(this_dir / "ops_norm.cu"),
    ]

    extra_cflags = ["-O3", "-std=c++17"]
    extra_cuda_cflags = ["-O3"]

    return load(
        name="sige3d_cuda",
        sources=sources,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        with_cuda=True,
        verbose=verbose,
   )