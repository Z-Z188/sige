from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.autograd.profiler import record_function


def _as_flow_hw2(flow: torch.Tensor) -> torch.Tensor:
    # Accept (H, W, 2), (2, H, W), (1, 2, H, W) -> (H, W, 2)
    if flow.dim() == 4:
        if int(flow.size(0)) != 1 or int(flow.size(1)) != 2:
            raise ValueError(f"Unsupported flow shape: {tuple(flow.shape)} (expected (1,2,H,W))")
        flow = flow[0]  # (2,H,W)

    if flow.dim() == 3 and int(flow.size(-1)) == 2:
        return flow

    if flow.dim() == 3 and int(flow.size(0)) == 2:
        return flow.permute(1, 2, 0).contiguous()

    raise ValueError(f"Unsupported flow shape: {tuple(flow.shape)} (expected (H,W,2) or (2,H,W))")


def normalize_flow(flow: torch.Tensor, h: int, w: int, *, device: torch.device) -> torch.Tensor:
    """
    Normalize flow to shape (H, W, 2) on `device`.

    If input resolution differs from (h,w), bilinearly resize and scale the
    displacement so it stays in pixel units at the target resolution.
    """
    if not torch.is_tensor(flow):
        flow = torch.as_tensor(flow)

    flow = _as_flow_hw2(flow).to(device=device, dtype=torch.float32)

    h0, w0 = int(flow.size(0)), int(flow.size(1))
    if (h0, w0) == (h, w):
        return flow

    flow_chw = flow.permute(2, 0, 1).unsqueeze(0)  # (1,2,H,W)
    flow_rs = F.interpolate(flow_chw, size=(h, w), mode="bilinear", align_corners=True)

    # Keep displacement in pixel units after resizing.
    if h0 > 1 and w0 > 1 and h > 1 and w > 1:
        flow_rs[:, 0].mul_((w - 1) / (w0 - 1))
        flow_rs[:, 1].mul_((h - 1) / (h0 - 1))
    else:
        flow_rs[:, 0].mul_(w / max(w0, 1))
        flow_rs[:, 1].mul_(h / max(h0, 1))

    return flow_rs[0].permute(1, 2, 0).contiguous()


def forward_warp_cache_5d(cache: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Backward-warp cached features by bilinear sampling (grid_sample).

    - cache: (B, C, T, H, W)
    - flow:  (H, W, 2) / (2, H, W) / (1, 2, H, W), (dx, dy) in pixel units
    - output: same shape, output[..., y, x] samples input at (x+dx, y+dy)
    """
    with record_function("sige3d::forward_warp_cache_5d"):
        if cache.dim() != 5:
            raise ValueError(f"cache must be 5D (B,C,T,H,W); got {tuple(cache.shape)}")
        b, c, t, h, w = cache.shape

        flow = normalize_flow(flow, int(h), int(w), device=cache.device)

        # Base grid in pixel coords: (x,y) with x in [0,w-1], y in [0,h-1].
        y = torch.arange(h, device=cache.device, dtype=torch.float32).view(h, 1)
        x = torch.arange(w, device=cache.device, dtype=torch.float32).view(1, w)
        base_x = x.expand(h, w)
        base_y = y.expand(h, w)

        sample_x = base_x + flow[..., 0]
        sample_y = base_y + flow[..., 1]

        # Normalize to [-1, 1] for grid_sample (align_corners=True).
        w_denom = float(max(w - 1, 1))
        h_denom = float(max(h - 1, 1))
        x_grid = 2.0 * sample_x / w_denom - 1.0
        y_grid = 2.0 * sample_y / h_denom - 1.0
        grid = torch.stack([x_grid, y_grid], dim=-1)  # (H,W,2)

        # Warp each time slice independently with the same spatial grid.
        cache_2d = cache.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, h, w)
        grid_bt = grid.unsqueeze(0).expand(b * t, h, w, 2).to(dtype=cache_2d.dtype)
        with record_function("sige3d::forward_warp_cache_5d::grid_sample"):
            warped = F.grid_sample(
                cache_2d,
                grid_bt,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
        return warped.view(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
