from __future__ import annotations

import torch


def iter_5_4_4_chunks(x: torch.Tensor):
    """
    StreamDiffusionV2-style temporal chunking.

    Input shape: [B, C, T, H, W]
    Yields chunks: [B, C, chunkT, H, W] with chunk sizes 5,4,4,...
    """
    t_total = int(x.shape[2])
    start = 0
    first = True
    while start < t_total:
        step = 5 if first else 4
        first = False
        end = min(start + step, t_total)
        yield x[:, :, start:end]
        start = end

