from __future__ import annotations

import torch

from ..activation import activation
from torch.autograd.profiler import record_function


def gather3d(
    x: torch.Tensor,
    b_size_h: int,
    b_size_w: int,
    active_indices: torch.Tensor,
    scale: torch.Tensor | None = None,
    shift: torch.Tensor | None = None,
    activation_name: str = "identity",
    activation_first: bool = False,
    rms_norm_fn = None,   # ✅ 新增：可传 RMS_norm forward
    is_cache_gather: bool = False, # 不需要norm和激活
) -> torch.Tensor:
    """
    A 3D (T,H,W) extension of `sige/cuda/gather_kernel.py` (torch reference).

    - Input:  `x` with shape [B, C, T, H, W] (NO spatial padding applied)
    - Output: gathered blocks with shape [B*num_active, C, T, b_size_h, b_size_w]

    Note:
      - We only gather on spatial (H,W) using `active_indices`.
      - Temporal causal context is handled by the causal Conv3d module via per-layer caches.
    """
    with record_function("sige3d::gather3d"):
        b, c, t, h, w = x.shape
        num_active = int(active_indices.size(0))
        r, s = int(b_size_h), int(b_size_w)

        # output blocks: [B, num_active, C, T, r, s] -> flatten to [B*num_active, C, T, r, s]
        output = torch.zeros((b, num_active, c, t, r, s), dtype=x.dtype, device=x.device)
        if num_active == 0:
            return output.view(b * num_active, c, t, r, s)

        # `scale` and `shift` are expected to be broadcastable to `x` (e.g. [B,C,1,1,1]).
        for ib, (bi_h, bi_w) in enumerate(active_indices.tolist()):
            h0 = max(bi_h, 0)
            h1 = min(bi_h + r, h)
            w0 = max(bi_w, 0)
            w1 = min(bi_w + s, w)
            if h0 >= h1 or w0 >= w1:
                continue

            # Where the valid region lands inside the (r,s) output block.
            dh0 = h0 - bi_h
            dh1 = dh0 + (h1 - h0)
            dw0 = w0 - bi_w
            dw1 = dw0 + (w1 - w0)

            block = x[:, :, :, h0:h1, w0:w1]

            # 先做 RMS_Norm
            if not is_cache_gather:
                if rms_norm_fn is not None:
                    block = rms_norm_fn(block)
                if scale is not None:
                    scale_gather = scale[:, :, :, h0:h1, w0:w1]   # [B,1,T,hh,ww]
                    block = block * scale_gather
                if shift is not None:
                    block = block + shift
                block = activation(block, activation_name)

            output[:, ib, :, :, dh0:dh1, dw0:dw1] = block

        return output.view(b * num_active, c, t, r, s)
