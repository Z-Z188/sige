from __future__ import annotations

from collections.abc import Callable, Iterable

import torch

from sige3d import SIGEModel3d

Chunker = Callable[[torch.Tensor], Iterable[torch.Tensor]]


@torch.no_grad()
def stream_forward_in_chunks(
    model: SIGEModel3d,
    x: torch.Tensor,
    *,
    chunker: Chunker,
    cache_per_chunk: bool,
    start_cache_id: int = 0,
) -> torch.Tensor:
    """
    Run a SIGE model chunk-by-chunk on the time axis while preserving causal stream caches.

    Notes on caches:
    - `model.clear_stream_cache()` controls causal Conv3d temporal caches (frame-level feature cache).
    - `model.set_cache_id()` controls SIGE full-baseline caching used by Scatter/ScatterGather.
      For pure streaming full inference (no later sparse pass), set `cache_per_chunk=False`
      so we always overwrite `cache_id=0` and avoid growing the baseline cache.

    Shapes:
      - x: [B, C, T, H, W]
      - returns: [B, C_out, T, H_out, W_out]
    """
    outputs: list[torch.Tensor] = []
    cache_id = int(start_cache_id)

    for chunk_id, x_chunk in enumerate(chunker(x)):
        # x_chunk: [B, C, chunkT, H, W]
        if cache_per_chunk:
            model.set_cache_id(cache_id + chunk_id)
        else:
            model.set_cache_id(0)
        outputs.append(model(x_chunk))

    return torch.cat(outputs, dim=2) if outputs else x[:, :0]


class ChunkStream:
    """
    Stateful helper for streaming inference.

    Call `.reset()` at the beginning of a new stream (new video), then feed chunks sequentially.
    """

    def __init__(self, model: SIGEModel3d, *, cache_per_chunk: bool):
        self.model = model
        self.cache_per_chunk = bool(cache_per_chunk)
        self._cache_id = 0

    def reset(self, *, clear_baseline_cache: bool = True):
        if clear_baseline_cache:
            self.model.clear_cache()
        self.model.clear_stream_cache()
        self._cache_id = 0

    @torch.no_grad()
    def __call__(self, x_chunk: torch.Tensor) -> torch.Tensor:
        # x_chunk: [B, C, chunkT, H, W]
        self.model.set_cache_id(self._cache_id if self.cache_per_chunk else 0)
        out = self.model(x_chunk)
        self._cache_id += 1
        return out

