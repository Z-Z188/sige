import torch

from .backend import use_cuda_kernels

# slow
# def scatter2d(
#     x: torch.Tensor,
#     y: torch.Tensor,
#     offset_h: int,
#     offset_w: int,
#     stride_h: int,
#     stride_w: int,
#     active_indices: torch.Tensor,
#     residual: torch.Tensor | None = None,
# ) -> torch.Tensor:
#     """PyTorch reference for sige/cuda/scatter_kernel.cu."""
#     b, c, h, w = y.shape
#     num_active = active_indices.size(0)
#     r, s = x.size(2), x.size(3)
#
#     # 不能直接修改y
#     # 这张图 必须保持不变, 是full前向算出来的 pre-computed 底图
#     output = y.clone()
#
#     if num_active == 0:
#         return output
#
#     x_blocks = x.reshape(b, num_active, c, r, s)
#     residual_y = residual.expand_as(y) if residual is not None else None
#
#     for ib, (ai_h, ai_w) in enumerate(active_indices.tolist()):
#         bi_h = (offset_h + ai_h) // stride_h
#         bi_w = (offset_w + ai_w) // stride_w
#         h0 = max(bi_h, 0)
#         h1 = min(bi_h + r, h)
#         w0 = max(bi_w, 0)
#         w1 = min(bi_w + s, w)
#         if h0 >= h1 or w0 >= w1:
#             continue
#         dh0 = h0 - bi_h
#         dh1 = dh0 + (h1 - h0)
#         dw0 = w0 - bi_w
#         dw1 = dw0 + (w1 - w0)
#
#         block = x_blocks[:, ib, :, dh0:dh1, dw0:dw1]
#         if residual_y is not None:
#             block = block + residual_y[:, :, h0:h1, w0:w1]
#         output[:, :, h0:h1, w0:w1] = block
#
#     return output


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
    把 residual 变成 4D，并 expand 到 [B,C,H,W]（要求可 broadcast）。
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
        raise ValueError(f"residual dim not supported: {tuple(v.shape)}")

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
    Prepare residual for the CUDA extension:
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
        raise ValueError(f"residual dim not supported: {tuple(v.shape)}")

    # Avoid passing expanded views to the extension (it assumes contiguous indexing).
    return v.contiguous()


def scatter2d(
    x: torch.Tensor,
    y: torch.Tensor,
    offset_h: int,
    offset_w: int,
    stride_h: int,
    stride_w: int,
    active_indices: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """PyTorch reference for sige/cuda/scatter_kernel.cu (vectorized)."""
    B, C, H, W = y.shape
    num_active = int(active_indices.size(0))
    r, s = int(x.size(-2)), int(x.size(-1))

    if num_active == 0:
        # 不能直接修改y：必须保持不变（full 前向算出来的 pre-computed 底图）
        return y.clone()

    if x.is_cuda and use_cuda_kernels():
        try:
            from ..cuda_kernels import get_extension

            ext = get_extension()
            device = x.device
            dtype = x.dtype
            active_indices_cuda = active_indices.to(device=device, dtype=torch.int32).contiguous()
            residual_cuda = _as_4d_broadcastable_contiguous_for_cuda(residual, B, C, H, W, device, dtype)
            return ext.scatter2d(
                x.contiguous(),
                y.contiguous(),
                int(offset_h),
                int(offset_w),
                int(stride_h),
                int(stride_w),
                active_indices_cuda,
                residual_cuda,
            )
        except Exception:
            pass

    # 不能直接修改y：必须保持不变（full 前向算出来的 pre-computed 底图）
    # output = y.clone()
    output = y
    device = output.device
    dtype = output.dtype

    active_indices = active_indices.to(device=device)
    ai_h = active_indices[:, 0].to(torch.int64)  # [N]
    ai_w = active_indices[:, 1].to(torch.int64)  # [N]

    bi_h = (offset_h + ai_h) // stride_h
    bi_w = (offset_w + ai_w) // stride_w

    dh = torch.arange(r, device=device, dtype=torch.int64)
    dw = torch.arange(s, device=device, dtype=torch.int64)
    dH, dW = torch.meshgrid(dh, dw, indexing="ij")  # [r,s]

    hh = bi_h[:, None, None] + dH[None, :, :]  # [N,r,s]
    ww = bi_w[:, None, None] + dW[None, :, :]  # [N,r,s]

    valid = (hh >= 0) & (hh < H) & (ww >= 0) & (ww < W)
    valid_flat = valid.reshape(-1)

    HW = H * W
    idx_hw = (hh * W + ww).reshape(-1)  # [K]
    idx_valid = idx_hw[valid_flat]  # [Kv]
    if idx_valid.numel() == 0:
        return output

    K = num_active * r * s
    x_blocks = x.reshape(B, num_active, C, r, s)
    x_vals_all = x_blocks.permute(0, 2, 1, 3, 4).reshape(B, C, K)  # [B,C,K]
    x_vals = x_vals_all[..., valid_flat]  # [B,C,Kv]

    idx_expand = idx_valid.reshape(1, 1, -1).expand(B, C, -1)

    if residual is not None:
        residual4 = _as_4d_broadcastable(residual, B, C, H, W, device, dtype)
        residual_flat = residual4.reshape(B, C, HW)
        res_vals = torch.gather(residual_flat, -1, idx_expand)
        x_vals = x_vals + res_vals

    output_flat = output.reshape(B, C, HW)
    output_flat.scatter_(2, idx_expand, x_vals)
    return output_flat.reshape(B, C, H, W)


# slow
# def scatter_with_block_residual2d(
#     x0: torch.Tensor,
#     y0: torch.Tensor,
#     x1: torch.Tensor,
#     y1: torch.Tensor,
#     offset_h: int,
#     offset_w: int,
#     stride_h: int,
#     stride_w: int,
#     active_indices0: torch.Tensor,
#     active_indices1: torch.Tensor,
# ) -> torch.Tensor:
#     """PyTorch reference for scatter_with_block_residual in scatter_kernel.cu."""
#
#     '''
#     x0: 主分支稀疏块的输出(main branch blocks)
#
#     y0: 主分支对应的“基底大图”
#
#     x1: shortcut 分支稀疏块输出(shortcut blocks)
#
#     y1: shortcut 分支的“基底大图”
#
#     其实输入的y0 = y0 + y1
#     elif self.mode == "full":
#         output = x + residual
#         self.original_outputs[self.cache_id] = output.contiguous()
#         self.original_residuals[self.cache_id] = residual.contiguous()
#
#
#     active_indices0: 主分支激活块的位置
#
#     active_indices1: shortcut 分支激活块的位置
#     '''
#
#     output = scatter(
#         x0,
#         y0,
#         offset_h,
#         offset_w,
#         stride_h,
#         stride_w,
#         active_indices0,
#         y1,   # scatter x0 + y1
#     )
#
#     b, c, h, w = y1.shape
#     num_active = active_indices1.size(0)
#     if num_active == 0:
#         return output
#
#     r, s = x1.size(2), x1.size(3)
#     x1_blocks = x1.reshape(b, num_active, c, r, s)
#
#     for ib, (bi_h, bi_w) in enumerate(active_indices1.tolist()):
#         h0 = max(bi_h, 0)
#         h1 = min(bi_h + r, h)
#         w0 = max(bi_w, 0)
#         w1 = min(bi_w + s, w)
#         if h0 >= h1 or w0 >= w1:
#             continue
#         dh0 = h0 - bi_h
#         dh1 = dh0 + (h1 - h0)
#         dw0 = w0 - bi_w
#         dw1 = dw0 + (w1 - w0)
#
#         # active_indices1 += x1 - y1
#         output[:, :, h0:h1, w0:w1] += (
#             x1_blocks[:, ib, :, dh0:dh1, dw0:dw1] - y1[:, :, h0:h1, w0:w1]
#         )
#
#     return output


# fast
def scatter_with_block_residual2d(
    x0: torch.Tensor,
    y0: torch.Tensor,
    x1: torch.Tensor,
    y1: torch.Tensor,
    offset_h: int,
    offset_w: int,
    stride_h: int,
    stride_w: int,
    active_indices0: torch.Tensor,
    active_indices1: torch.Tensor,
) -> torch.Tensor:
    """PyTorch reference for scatter_with_block_residual in scatter_kernel.cu (vectorized)."""
    if x0.is_cuda and use_cuda_kernels():
        try:
            from ..cuda_kernels import get_extension

            ext = get_extension()
            device = x0.device
            return ext.scatter_with_block_residual2d(
                x0.contiguous(),
                y0.contiguous(),
                x1.contiguous(),
                y1.contiguous(),
                int(offset_h),
                int(offset_w),
                int(stride_h),
                int(stride_w),
                active_indices0.to(device=device, dtype=torch.int32).contiguous(),
                active_indices1.to(device=device, dtype=torch.int32).contiguous(),
            )
        except Exception:
            pass

    output = scatter2d(
        x0,
        y0,
        offset_h,
        offset_w,
        stride_h,
        stride_w,
        active_indices0,
        y1,  # scatter x0 + y1
    )

    B, C, H, W = y1.shape
    num_active = int(active_indices1.size(0))
    if num_active == 0:
        return output

    r, s = int(x1.size(-2)), int(x1.size(-1))
    device = output.device

    active_indices1 = active_indices1.to(device=device)
    bi_h = active_indices1[:, 0].to(torch.int64)
    bi_w = active_indices1[:, 1].to(torch.int64)

    dh = torch.arange(r, device=device, dtype=torch.int64)
    dw = torch.arange(s, device=device, dtype=torch.int64)
    dH, dW = torch.meshgrid(dh, dw, indexing="ij")

    hh = bi_h[:, None, None] + dH[None, :, :]  # [N,r,s]
    ww = bi_w[:, None, None] + dW[None, :, :]  # [N,r,s]

    valid = (hh >= 0) & (hh < H) & (ww >= 0) & (ww < W)
    valid_flat = valid.reshape(-1)

    HW = H * W
    idx_hw = (hh * W + ww).reshape(-1)
    idx_valid = idx_hw[valid_flat]
    if idx_valid.numel() == 0:
        return output

    K = num_active * r * s
    x1_blocks = x1.reshape(B, num_active, C, r, s)
    x1_vals_all = x1_blocks.permute(0, 2, 1, 3, 4).reshape(B, C, K)
    x1_vals = x1_vals_all[..., valid_flat]

    idx_expand = idx_valid.reshape(1, 1, -1).expand(B, C, -1)

    y1_flat = y1.reshape(B, C, HW)
    y1_vals = torch.gather(y1_flat, -1, idx_expand)

    delta = x1_vals - y1_vals

    output_flat = output.reshape(B, C, HW)
    output_flat.scatter_add_(2, idx_expand, delta)
    return output_flat.reshape(B, C, H, W)
