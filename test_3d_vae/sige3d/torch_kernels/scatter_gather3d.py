from __future__ import annotations

import torch

from .activation import activation
from torch.autograd.profiler import record_function


def get_scatter_map(
    h: int,
    w: int,
    b_size_h: int,
    b_size_w: int,
    k_size_h: int,
    k_size_w: int,
    offset_h: int,
    offset_w: int,
    stride_h: int,
    stride_w: int,
    active_indices: torch.Tensor,
) -> torch.Tensor:
    """
    3D extension of get_scatter_map in `sige/cuda/scatter_gather_kernel.py`.
    Still a pure spatial (H,W) map: [H, W, 3] -> (block_id, intra_h, intra_w).
    """
    scatter_map = torch.full((h, w, 3), -1, dtype=torch.int32, device=active_indices.device)
    r = (b_size_h - k_size_h) // stride_h + 1
    s = (b_size_w - k_size_w) // stride_w + 1
    for ib, (ai_h, ai_w) in enumerate(active_indices.tolist()):
        bi_h = (offset_h + ai_h) // stride_h
        bi_w = (offset_w + ai_w) // stride_w
        for intra_bh in range(r):
            hh = bi_h + intra_bh
            if hh < 0 or hh >= h:
                continue
            for intra_bw in range(s):
                ww = bi_w + intra_bw
                if ww < 0 or ww >= w:
                    continue
                scatter_map[hh, ww, 0] = ib
                scatter_map[hh, ww, 1] = intra_bh
                scatter_map[hh, ww, 2] = intra_bw
    return scatter_map


def scatter_gather3d(
    x: torch.Tensor,
    y: torch.Tensor,
    b_size_h: int,
    b_size_w: int,
    active_indices: torch.Tensor,
    scatter_map: torch.Tensor,
    scale: torch.Tensor | None = None,
    shift: torch.Tensor | None = None,
    activation_name: str = "identity",
    activation_first: bool = False,
    rms_norm_fn = None,   # ✅ 新增：可传 RMS_norm forward

) -> torch.Tensor:
    """
    3D (T,H,W) extension of `sige/cuda/scatter_gather_kernel.py` (torch reference).

    - `x`: updated conv outputs as blocks, shape [B*num_active, C, T, rx, sx]
    - `y`: baseline full tensor, shape [B, C, T, H, W]
    - returns: gathered blocks for the next conv, shape [B*num_active, C, T, b_size_h, b_size_w]
    """
    with record_function("sige3d::scatter_gather3d"):
        b, c, t, h, w = y.shape
        num_active = int(active_indices.size(0))
        ro, so = int(b_size_h), int(b_size_w)
        rx, sx = int(x.size(-2)), int(x.size(-1))

        # output blocks: [B, num_active, C, T, ro, so] -> flatten to [B*num_active, C, T, ro, so]
        output = torch.zeros((b, num_active, c, t, ro, so), dtype=x.dtype, device=x.device)
        if num_active == 0:
            return output.view(b * num_active, c, t, ro, so)

        x_blocks = x.view(b, num_active, c, t, rx, sx)

        if scatter_map.device.type != "cpu":
            scatter_map_cpu = scatter_map.detach().cpu()
        else:
            scatter_map_cpu = scatter_map

        for ib, (bi_h, bi_w) in enumerate(active_indices.tolist()):
            for intra_bh in range(ro):
                hh = bi_h + intra_bh
                if hh < 0 or hh >= h:
                    continue
                for intra_bw in range(so):
                    ww = bi_w + intra_bw
                    if ww < 0 or ww >= w:
                        continue
                    bx, hx, wx = scatter_map_cpu[hh, ww].tolist()
                    if bx >= 0:
                        # z: [B, C, T]
                        z = x_blocks[:, bx, :, :, hx, wx]
                    else:
                        z = y[:, :, :, hh, ww]

                    if rms_norm_fn is not None:
                        # z: [B, C, T] -> [B, C, T, 1, 1]，让 gamma: [C,1,1] 能正确广播
                        z = rms_norm_fn(z.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)

                    if not activation_first:
                        if scale is not None:
                            z = z * scale[:, :, :, hh, ww]
                        if shift is not None:
                            z = z + shift
                        z = activation(z, activation_name)
                    else:
                        z = activation(z, activation_name)
                        if scale is not None:
                            z = z * scale
                        if shift is not None:
                            z = z + shift
                    output[:, ib, :, :, intra_bh, intra_bw] = z

        return output.view(b * num_active, c, t, ro, so)
