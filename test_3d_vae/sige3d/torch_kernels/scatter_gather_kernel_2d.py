import torch

from ..activation import activation
from .backend import use_cuda_kernels


# slow
# def get_scatter_map2d(
#     h: int,
#     w: int,
#     b_size_h: int,
#     b_size_w: int,
#     k_size_h: int,
#     k_size_w: int,
#     offset_h: int,
#     offset_w: int,
#     stride_h: int,
#     stride_w: int,
#     active_indices: torch.Tensor,
# ) -> torch.Tensor:
#     """PyTorch reference for get_scatter_map in scatter_gather_kernel.cu."""
#     scatter_map = torch.full(
#         (h, w, 3),
#         -1,
#         dtype=torch.int32,
#         device=active_indices.device,
#     )
#     r = (b_size_h - k_size_h) // stride_h + 1
#     s = (b_size_w - k_size_w) // stride_w + 1
#     for ib, (ai_h, ai_w) in enumerate(active_indices.tolist()):
#         bi_h = (offset_h + ai_h) // stride_h
#         bi_w = (offset_w + ai_w) // stride_w
#         for intra_bh in range(r):
#             hh = bi_h + intra_bh
#             if hh < 0 or hh >= h:
#                 continue
#             for intra_bw in range(s):
#                 ww = bi_w + intra_bw
#                 if ww < 0 or ww >= w:
#                     continue
#                 scatter_map[hh, ww, 0] = ib
#                 scatter_map[hh, ww, 1] = intra_bh
#                 scatter_map[hh, ww, 2] = intra_bw
#     return scatter_map


# fast
def get_scatter_map2d(
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
    """CUDA implementation (fallback to torch vectorized when unavailable)."""
    device = active_indices.device
    scatter_map = torch.full((h, w, 3), -1, dtype=torch.int32, device=device)

    # r,s: 每个 block 会覆盖的输出点数（沿 H/W 的次数）
    r = (b_size_h - k_size_h) // stride_h + 1
    s = (b_size_w - k_size_w) // stride_w + 1

    if active_indices.numel() == 0:
        return scatter_map

    if active_indices.is_cuda and use_cuda_kernels():
        try:
            from ..cuda_kernels import get_extension

            ext = get_extension()
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
                active_indices.to(dtype=torch.int32).contiguous(),
            )
        except Exception:
            pass

    ai_h = active_indices[:, 0].to(torch.int64)  # [N]
    ai_w = active_indices[:, 1].to(torch.int64)  # [N]

    bi_h = (offset_h + ai_h) // stride_h  # [N]
    bi_w = (offset_w + ai_w) // stride_w  # [N]

    N = int(active_indices.size(0))
    ib = torch.arange(N, device=device, dtype=torch.int64)  # [N]

    intra_bh = torch.arange(r, device=device, dtype=torch.int64)  # [r]
    intra_bw = torch.arange(s, device=device, dtype=torch.int64)  # [s]

    hh = bi_h[:, None, None] + intra_bh[None, :, None]  # [N,r,1]
    ww = bi_w[:, None, None] + intra_bw[None, None, :]  # [N,1,s]

    hh = hh.expand(N, r, s)
    ww = ww.expand(N, r, s)

    valid = (hh >= 0) & (hh < h) & (ww >= 0) & (ww < w)
    hh = hh[valid]
    ww = ww[valid]

    block_id = ib[:, None, None].expand(N, r, s)[valid]
    intra_h = intra_bh[None, :, None].expand(N, r, s)[valid]
    intra_w = intra_bw[None, None, :].expand(N, r, s)[valid]

    scatter_map[hh, ww, 0] = block_id.to(torch.int32)
    scatter_map[hh, ww, 1] = intra_h.to(torch.int32)
    scatter_map[hh, ww, 2] = intra_w.to(torch.int32)
    return scatter_map


# slow
# def scatter_gather2d(
#     x: torch.Tensor,
#     y: torch.Tensor,
#     b_size_h: int,
#     b_size_w: int,
#     active_indices: torch.Tensor,
#     scatter_map: torch.Tensor,
#     scale: torch.Tensor | None = None,
#     shift: torch.Tensor | None = None,
#     activation_name: str = "identity",
#     activation_first: bool = False,
# ) -> torch.Tensor:
#     """PyTorch reference for scatter_gather in scatter_gather_kernel.cu."""
#     b, c, h, w = y.shape
#     num_active = active_indices.size(0)
#     ro, so = int(b_size_h), int(b_size_w)
#     rx, sx = x.size(2), x.size(3)
#
#     output = torch.zeros((b, num_active, c, ro, so), dtype=x.dtype, device=x.device)
#     if num_active == 0:
#         return output.reshape(b * num_active, c, ro, so)
#
#     x_blocks = x.reshape(b, num_active, c, rx, sx)
#     scale_y = scale.expand_as(y) if scale is not None else None
#     shift_y = shift.expand_as(y) if shift is not None else None
#
#     if scatter_map.device.type != "cpu":
#         scatter_map_cpu = scatter_map.detach().cpu()
#     else:
#         scatter_map_cpu = scatter_map
#
#     for ib, (bi_h, bi_w) in enumerate(active_indices.tolist()):
#         for intra_bh in range(ro):
#             hh = bi_h + intra_bh
#             if hh < 0 or hh >= h:
#                 continue
#             for intra_bw in range(so):
#                 ww = bi_w + intra_bw
#                 if ww < 0 or ww >= w:
#                     continue
#                 bx, hx, wx = scatter_map_cpu[hh, ww].tolist()
#                 if bx >= 0:
#                     z = x_blocks[:, bx, :, hx, wx]
#                 else:
#                     z = y[:, :, hh, ww]
#                 if not activation_first:
#                     if scale_y is not None:
#                         z = z * scale_y[:, :, hh, ww]
#                     if shift_y is not None:
#                         z = z + shift_y[:, :, hh, ww]
#                     z = activation(z, activation_name)
#                 else:
#                     z = activation(z, activation_name)
#                     if scale_y is not None:
#                         z = z * scale_y[:, :, hh, ww]
#                     if shift_y is not None:
#                         z = z + shift_y[:, :, hh, ww]
#                 output[:, ib, :, intra_bh, intra_bw] = z
#
#     return output.reshape(b * num_active, c, ro, so)


# fast
def _as_4d_broadcastable(
    v: torch.Tensor,
    B: int,
    C: int,
    H: int,
    W: int,
    device,
    dtype,
) -> torch.Tensor | None:
    """
    把 scale/shift 变成 4D，并 expand 到 [B,C,H,W]（要求可 broadcast）。
    支持: scalar, [C], [B,C], [B,C,H,W]
    """
    if v is None:
        return None
    if not torch.is_tensor(v):
        v = torch.tensor(v, device=device, dtype=dtype)
    if v.device != device:
        v = v.to(device)
    if v.dtype != dtype:
        v = v.to(dtype)

    if v.dim() == 0:  # []
        v = v.view(1, 1, 1, 1)
    elif v.dim() == 1:  # [C]
        v = v.view(1, C, 1, 1)
    elif v.dim() == 2:  # [B,C] or [1,C]
        v = v.view(v.size(0), v.size(1), 1, 1)
    elif v.dim() == 4:  # [B,C,H,W]
        pass
    else:
        raise ValueError(f"scale/shift dim not supported: {tuple(v.shape)}")

    target = (B, C, H, W)
    exp = []
    for cur, tar in zip(v.shape, target):
        if cur == tar:
            exp.append(cur)
        elif cur == 1:
            exp.append(tar)
        else:
            raise ValueError(f"not broadcastable: {tuple(v.shape)} -> {target}")
    return v.expand(*exp)


def _as_4d_broadcastable_contiguous_for_cuda(
    v: torch.Tensor,
    B: int,
    C: int,
    H: int,
    W: int,
    device,
    dtype,
) -> torch.Tensor | None:
    """
    Prepare scale/shift for the CUDA extension:
    - returns a contiguous 4D tensor with broadcastable shape (no `expand()` views).
    - supports: scalar, [C], [B,C], [B,C,H,W]
    """
    if v is None:
        return None
    if not torch.is_tensor(v):
        v = torch.tensor(v, device=device, dtype=dtype)
    if v.device != device:
        v = v.to(device)
    if v.dtype != dtype:
        v = v.to(dtype)

    if v.dim() == 0:  # []
        v = v.view(1, 1, 1, 1)
    elif v.dim() == 1:  # [C]
        v = v.view(1, C, 1, 1)
    elif v.dim() == 2:  # [B,C] or [1,C]
        v = v.view(v.size(0), v.size(1), 1, 1)
    elif v.dim() == 4:  # [B,C,H,W]
        pass
    else:
        raise ValueError(f"scale/shift dim not supported: {tuple(v.shape)}")

    return v.contiguous()


def scatter_gather2d(
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
) -> torch.Tensor:
    """CUDA implementation (fallback to torch vectorized when unavailable)."""
    B, C, H, W = y.shape
    num_active = int(active_indices.size(0))
    Ro, So = int(b_size_h), int(b_size_w)
    Rx, Sx = int(x.size(-2)), int(x.size(-1))

    if num_active == 0:
        return x.new_zeros((0, C, Ro, So))

    if x.is_cuda and use_cuda_kernels():
        try:
            from ..cuda_kernels import get_extension

            ext = get_extension()
            active_indices_cuda = active_indices.to(device=x.device, dtype=torch.int32).contiguous()
            scatter_map_cuda = scatter_map.to(device=x.device, dtype=torch.int32).contiguous()
            scale_cuda = _as_4d_broadcastable_contiguous_for_cuda(scale, B, C, H, W, x.device, x.dtype)
            shift_cuda = _as_4d_broadcastable_contiguous_for_cuda(shift, B, C, H, W, x.device, x.dtype)
            return ext.scatter_gather2d(
                x.contiguous(),
                y.contiguous(),
                Ro,
                So,
                active_indices_cuda,
                scatter_map_cuda,
                scale_cuda,
                shift_cuda,
                activation_name,
                activation_first,
            )
        except Exception:
            pass

    device = x.device
    dtype = x.dtype

    active_indices = active_indices.to(device=device)
    scatter_map = scatter_map.to(device=device)

    # ---------- 1) 构造每个输出 block 内所有位置的 (hh, ww) ----------
    bi_h = active_indices[:, 0].to(torch.int64)  # [N]
    bi_w = active_indices[:, 1].to(torch.int64)  # [N]
    dh = torch.arange(Ro, device=device, dtype=torch.int64)
    dw = torch.arange(So, device=device, dtype=torch.int64)
    dH, dW = torch.meshgrid(dh, dw, indexing="ij")  # [Ro,So]

    hh = bi_h[:, None, None] + dH[None, :, :]  # [N,Ro,So]
    ww = bi_w[:, None, None] + dW[None, :, :]  # [N,Ro,So]

    valid = (hh >= 0) & (hh < H) & (ww >= 0) & (ww < W)  # [N,Ro,So]
    valid_f = valid.to(dtype)

    HW = H * W
    idx_hw = (hh * W + ww)
    idx_hw_clamped = idx_hw.clamp(0, HW - 1)
    K = num_active * Ro * So

    # ---------- 2) 查 scatter_map 得到 (bx,hx,wx) ----------
    sm_flat = scatter_map.view(-1, 3).to(torch.int64)  # [HW,3]
    sm = sm_flat[idx_hw_clamped]  # [N,Ro,So,3]
    bx = sm[..., 0]
    hx = sm[..., 1]
    wx = sm[..., 2]
    bx = torch.where(valid, bx, torch.full_like(bx, -1))

    # ---------- 3) 并行 gather y ----------
    y_flat = y.reshape(B, C, HW)
    idx_y = idx_hw_clamped.reshape(1, 1, K).expand(B, C, K)
    y_vals = torch.gather(y_flat, -1, idx_y).reshape(B, C, num_active, Ro, So)
    y_vals = y_vals * valid_f[None, None, :, :, :]

    # ---------- 4) 并行 gather x ----------
    x_blocks = x.reshape(B, num_active, C, Rx, Sx)
    x_src = x_blocks.reshape(B, num_active, C, Rx * Sx).permute(0, 2, 1, 3).reshape(B, C, num_active * Rx * Sx)

    idx_rx = (hx * Sx + wx).clamp(0, Rx * Sx - 1)
    src_idx = (bx.clamp(min=0) * (Rx * Sx) + idx_rx).reshape(1, 1, K).expand(B, C, K)
    x_vals = torch.gather(x_src, -1, src_idx).reshape(B, C, num_active, Ro, So)

    use_x = (bx >= 0) & valid
    z = torch.where(use_x[None, None, :, :, :], x_vals, y_vals)
    z = z * valid_f[None, None, :, :, :]  # invalid -> 0

    # ---------- 5) scale / shift ----------
    scale4 = _as_4d_broadcastable(scale, B, C, H, W, device, dtype) if scale is not None else None
    shift4 = _as_4d_broadcastable(shift, B, C, H, W, device, dtype) if shift is not None else None

    if scale4 is not None:
        scale_flat = scale4.reshape(B, C, HW)
        scale_vals = torch.gather(scale_flat, -1, idx_y).reshape(B, C, num_active, Ro, So)
        scale_vals = scale_vals * valid_f[None, None, :, :, :] + (1.0 - valid_f[None, None, :, :, :])
    else:
        scale_vals = None

    if shift4 is not None:
        shift_flat = shift4.reshape(B, C, HW)
        shift_vals = torch.gather(shift_flat, -1, idx_y).reshape(B, C, num_active, Ro, So)
        shift_vals = shift_vals * valid_f[None, None, :, :, :]
    else:
        shift_vals = None

    # ---------- 6) activation / scale / shift 顺序 ----------
    if not activation_first:
        if scale_vals is not None:
            z = z * scale_vals
        if shift_vals is not None:
            z = z + shift_vals
        z = activation(z, activation_name)
    else:
        z = activation(z, activation_name)
        if scale_vals is not None:
            z = z * scale_vals
        if shift_vals is not None:
            z = z + shift_vals

    z = z * valid_f[None, None, :, :, :]  # ensure invalid -> 0

    return z.permute(0, 2, 1, 3, 4).reshape(B * num_active, C, Ro, So)
