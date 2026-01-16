from __future__ import annotations

import torch

from ..activation import activation
from .backend import use_cuda_kernels


def _get_scatter_map_torch(
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



def _extract_rms_norm_params(rms_norm_fn, x: torch.Tensor):
    if rms_norm_fn is None:
        return None, None, None
    mod = getattr(rms_norm_fn, "__self__", None)
    if mod is None:
        return None, None, None
    gamma = getattr(mod, "gamma", None)
    if not torch.is_tensor(gamma):
        return None, None, None
    eps = float(getattr(mod, "eps", 1e-6))
    bias = getattr(mod, "bias", None)
    gamma_t = gamma.to(device=x.device, dtype=x.dtype).contiguous().view(-1)
    bias_t = bias.to(device=x.device, dtype=x.dtype).contiguous().view(-1) if torch.is_tensor(bias) else None
    return gamma_t, bias_t, eps


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
    if use_cuda_kernels() and active_indices.is_cuda:
        try:
            from ._sige_cuda import get_sige3d_cuda_ext

            ext = get_sige3d_cuda_ext()
            if active_indices.dtype != torch.int32:
                active_indices = active_indices.to(dtype=torch.int32)
            return ext.get_scatter_map(
                int(h),
                int(w),
                int(b_size_h),
                int(b_size_w),
                int(k_size_h),
                int(k_size_w),
                int(offset_h),
                int(offset_w),
                int(stride_h),
                int(stride_w),
                active_indices.contiguous(),
            )
        except Exception:
            pass
    return _get_scatter_map_torch(
        h,
        w,
        b_size_h,
        b_size_w,
        k_size_h,
        k_size_w,
        offset_h,
        offset_w,
        stride_h,
        stride_w,
        active_indices,
    )


def _scatter_gather3d_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    b_size_h: int,
    b_size_w: int,
    active_indices: torch.Tensor,
    scatter_map: torch.Tensor,
    activation_name: str = "identity",
    rms_norm_fn = None,   # ✅ 新增：可传 RMS_norm forward
) -> torch.Tensor:
    """
    3D (T,H,W) extension of `sige/cuda/scatter_gather_kernel.py` (torch reference).

    - `x`: updated conv outputs as blocks, shape [B*num_active, C, T, rx, sx]
    - `y`: baseline full tensor, shape [B, C, T, H, W]
    - returns: gathered blocks for the next conv, shape [B*num_active, C, T, b_size_h, b_size_w]
    """
    b, c, t, h, w = y.shape
    num_active = int(active_indices.size(0))
    ro, so = int(b_size_h), int(b_size_w)
    rx, sx = int(x.size(-2)), int(x.size(-1))

    # output blocks: [B, num_active, C, T, ro, so] -> flatten to [B*num_active, C, T, ro, so]
    output = torch.zeros((b, num_active, c, t, ro, so), dtype=x.dtype, device=x.device)
    if num_active == 0:
        return output.view(b * num_active, c, t, ro, so)

    x_blocks = x.view(b, num_active, c, t, rx, sx)

    for ib, (bi_h, bi_w) in enumerate(active_indices.tolist()):
        for intra_bh in range(ro):
            hh = bi_h + intra_bh
            if hh < 0 or hh >= h:
                continue
            for intra_bw in range(so):
                ww = bi_w + intra_bw
                if ww < 0 or ww >= w:
                    continue
                bx, hx, wx = scatter_map[hh, ww].tolist()
                if bx >= 0:
                    # z: [B, C, T]
                    z = x_blocks[:, bx, :, :, hx, wx]
                else:
                    z = y[:, :, :, hh, ww]

                if rms_norm_fn is not None:
                    # z: [B, C, T] -> [B, C, T, 1, 1]，让 gamma: [C,1,1] 能正确广播
                    z = rms_norm_fn(z.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
                    z = activation(z, activation_name)

                output[:, ib, :, :, intra_bh, intra_bw] = z

    return output.view(b * num_active, c, t, ro, so)


def scatter_gather3d(
    x: torch.Tensor,
    y: torch.Tensor,
    b_size_h: int,
    b_size_w: int,
    active_indices: torch.Tensor,
    scatter_map: torch.Tensor,
    activation_name: str = "identity",
    rms_norm_fn=None,
) -> torch.Tensor:
    if use_cuda_kernels() and x.is_cuda and y.is_cuda:
        act = (activation_name or "identity").strip().lower()
        if act not in {"identity", "silu", "swish"}:
            return _scatter_gather3d_torch(
                x,
                y,
                b_size_h,
                b_size_w,
                active_indices,
                scatter_map,
                activation_name,
                rms_norm_fn=rms_norm_fn,
            )
        try:
            from ._sige_cuda import get_sige3d_cuda_ext

            ext = get_sige3d_cuda_ext()
            if active_indices.device != x.device or active_indices.dtype != torch.int32:
                active_indices = active_indices.to(device=x.device, dtype=torch.int32)
            if scatter_map.device != x.device or scatter_map.dtype != torch.int32:
                scatter_map = scatter_map.to(device=x.device, dtype=torch.int32)

            gamma = None
            bias = None
            eps = 1e-6
            if rms_norm_fn is not None:
                gamma, bias, eps = _extract_rms_norm_params(rms_norm_fn, x)
                if gamma is None:
                    return _scatter_gather3d_torch(
                        x,
                        y,
                        b_size_h,
                        b_size_w,
                        active_indices,
                        scatter_map,
                        activation_name,
                        rms_norm_fn=rms_norm_fn,
                    )

            return ext.scatter_gather3d(
                x,
                y,
                int(b_size_h),
                int(b_size_w),
                active_indices.contiguous(),
                scatter_map.contiguous(),
                gamma,
                bias,
                float(eps),
                act,
            )
        except Exception as e:
            raise RuntimeError("SIGE CUDA failed") from e

    return _scatter_gather3d_torch(
        x,
        y,
        b_size_h,
        b_size_w,
        active_indices,
        scatter_map,
        activation_name,
        rms_norm_fn=rms_norm_fn,
    )
