from __future__ import annotations

import torch

from ..activation import activation
from torch.profiler import record_function
from .backend import use_cuda_kernels


# slow
# def gather3d(
#     x: torch.Tensor,
#     b_size_h: int,
#     b_size_w: int,
#     active_indices: torch.Tensor,
#     scale: torch.Tensor | None = None,
#     shift: torch.Tensor | None = None,
#     activation_name: str = "identity",
#     activation_first: bool = False,
#     rms_norm_fn = None,   # ✅ 新增：可传 RMS_norm forward
#     is_cache_gather: bool = False, # 不需要norm和激活
# ) -> torch.Tensor:
#     """
#     A 3D (T,H,W) extension of `sige/cuda/gather_kernel.py` (torch reference).
#
#     - Input:  `x` with shape [B, C, T, H, W] (NO spatial padding applied)
#     - Output: gathered blocks with shape [B*num_active, C, T, b_size_h, b_size_w]
#
#     Note:
#       - We only gather on spatial (H,W) using `active_indices`.
#       - Temporal causal context is handled by the causal Conv3d module via per-layer caches.
#     """
#     b, c, t, h, w = x.shape
#     num_active = int(active_indices.size(0))
#     r, s = int(b_size_h), int(b_size_w)
#
#     # output blocks: [B, num_active, C, T, r, s] -> flatten to [B*num_active, C, T, r, s]
#     output = torch.zeros((b, num_active, c, t, r, s), dtype=x.dtype, device=x.device)
#     if num_active == 0:
#         return output.view(b * num_active, c, t, r, s)
#
#     # `scale` and `shift` are expected to be broadcastable to `x` (e.g. [B,C,1,1,1]).
#     for ib, (bi_h, bi_w) in enumerate(active_indices.tolist()):
#         h0 = max(bi_h, 0)
#         h1 = min(bi_h + r, h)
#         w0 = max(bi_w, 0)
#         w1 = min(bi_w + s, w)
#         if h0 >= h1 or w0 >= w1:
#             continue
#
#         # Where the valid region lands inside the (r,s) output block.
#         dh0 = h0 - bi_h
#         dh1 = dh0 + (h1 - h0)
#         dw0 = w0 - bi_w
#         dw1 = dw0 + (w1 - w0)
#
#         block = x[:, :, :, h0:h1, w0:w1]
#
#         # 先做 RMS_Norm
#         if not is_cache_gather:
#             if rms_norm_fn is not None:
#                 block = rms_norm_fn(block)
#             if scale is not None:
#                 scale_gather = scale[:, :, :, h0:h1, w0:w1]   # [B,1,T,hh,ww]
#                 block = block * scale_gather
#             if shift is not None:
#                 block = block + shift
#             block = activation(block, activation_name)
#
#         output[:, ib, :, :, dh0:dh1, dw0:dw1] = block
#
#     return output.view(b * num_active, c, t, r, s)


# fast
def _as_5d_broadcastable(
    v: torch.Tensor,
    B: int,
    C: int,
    T: int,
    H: int,
    W: int,
    device,
    dtype,
) -> torch.Tensor | None:
    """
    把 scale/shift 变成 5D，并 expand 到 [B,C,T,H,W]（要求可 broadcast）。
    支持: scalar, [C], [B,C], [B,C,T], [B,C,H,W], [B,C,T,H,W]
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
        v = v.view(1, 1, 1, 1, 1)
    elif v.dim() == 1:  # [C]
        v = v.view(1, C, 1, 1, 1)
    elif v.dim() == 2:  # [B,C] or [1,C]
        v = v.view(v.size(0), v.size(1), 1, 1, 1)
    elif v.dim() == 3:  # [B,C,T]
        v = v.view(v.size(0), v.size(1), v.size(2), 1, 1)
    elif v.dim() == 4:  # [B,C,H,W]
        v = v.unsqueeze(2)  # -> [B,C,1,H,W]
    elif v.dim() == 5:  # [B,C,T,H,W]
        pass
    else:
        raise ValueError(f"scale/shift dim not supported: {tuple(v.shape)}")

    target = (B, C, T, H, W)
    exp = []
    for cur, tar in zip(v.shape, target):
        if cur == tar:
            exp.append(cur)
        elif cur == 1:
            exp.append(tar)
        else:
            raise ValueError(f"not broadcastable: {tuple(v.shape)} -> {target}")
    return v.expand(*exp)


def _as_5d_broadcastable_contiguous_for_cuda(
    v: torch.Tensor,
    B: int,
    C: int,
    T: int,
    H: int,
    W: int,
    device,
    dtype,
) -> torch.Tensor | None:
    """
    Prepare scale/shift for the CUDA extension:
    - returns a contiguous 5D tensor with broadcastable shape (no `expand()` views).
    - supports: scalar, [C], [B,C], [B,C,T], [B,C,H,W], [B,C,T,H,W]
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
        v = v.view(1, 1, 1, 1, 1)
    elif v.dim() == 1:  # [C]
        v = v.view(1, C, 1, 1, 1)
    elif v.dim() == 2:  # [B,C] or [1,C]
        v = v.view(v.size(0), v.size(1), 1, 1, 1)
    elif v.dim() == 3:  # [B,C,T]
        v = v.view(v.size(0), v.size(1), v.size(2), 1, 1)
    elif v.dim() == 4:  # [B,C,H,W]
        v = v.unsqueeze(2)  # -> [B,C,1,H,W]
    elif v.dim() == 5:  # [B,C,T,H,W]
        pass
    else:
        raise ValueError(f"scale/shift dim not supported: {tuple(v.shape)}")

    return v.contiguous()


def gather3d(
    x: torch.Tensor,
    b_size_h: int,
    b_size_w: int,
    active_indices: torch.Tensor,
    scale: torch.Tensor | None = None,
    shift: torch.Tensor | None = None,
    activation_name: str = "identity",
    activation_first: bool = False,
    rms_norm_fn=None,  # ✅ 新增：可传 RMS_norm forward
    is_cache_gather: bool = False,  # 不需要norm和激活
) -> torch.Tensor:
    """
    CUDA implementation (fallback to torch vectorized when unavailable).

    - Input:  `x` with shape [B, C, T, H, W] (NO spatial padding applied)
    - Output: gathered blocks with shape [B*num_active, C, T, b_size_h, b_size_w]
    """
    B, C, T, H, W = x.shape
    num_active = int(active_indices.size(0))
    r, s = int(b_size_h), int(b_size_w)

    if num_active == 0:
        return x.new_zeros((0, C, T, r, s))

    if x.is_cuda and use_cuda_kernels():
        try:
            from ..cuda_kernels import get_extension

            ext = get_extension()
            x_cuda = x.contiguous()
            active_indices_cuda = active_indices.to(device=x.device, dtype=torch.int32).contiguous()

            # cache gather: do not apply RMSNorm/scale/shift/activation
            if is_cache_gather:
                return ext.gather3d(
                    x_cuda,
                    r,
                    s,
                    active_indices_cuda,
                    None,
                    None,
                    "identity",
                    False,
                )

            # When RMSNorm is not involved, we can do everything inside CUDA.
            # NOTE: gather3d torch reference path ignores `activation_first`, so keep the same behavior here.
            if rms_norm_fn is None:
                scale_cuda = _as_5d_broadcastable_contiguous_for_cuda(scale, B, C, T, H, W, x.device, x.dtype)
                shift_cuda = _as_5d_broadcastable_contiguous_for_cuda(shift, B, C, T, H, W, x.device, x.dtype)
                return ext.gather3d(
                    x_cuda,
                    r,
                    s,
                    active_indices_cuda,
                    scale_cuda,
                    shift_cuda,
                    activation_name,
                    False,
                )

            # RMSNorm path: fully fused in CUDA (gather + RMSNorm + scale/shift + activation).
            norm_module = getattr(rms_norm_fn, "__self__", None)
            if norm_module is not None and hasattr(ext, "gather3d_rmsnorm"):
                if not bool(getattr(norm_module, "channel_first", True)):
                    raise RuntimeError("CUDA RMSNorm path requires channel_first=True")
                eps = float(getattr(norm_module, "eps", 1e-6))
                gamma = getattr(norm_module, "gamma", None)
                if gamma is None:
                    raise RuntimeError("rms_norm_fn has no gamma parameter")

                bias = getattr(norm_module, "bias", None)
                bias_tensor = bias if torch.is_tensor(bias) else None

                gamma_cuda = gamma.to(device=x.device, dtype=x.dtype).reshape(-1).contiguous()
                bias_cuda = None if bias_tensor is None else bias_tensor.to(device=x.device, dtype=x.dtype).reshape(-1).contiguous()

                scale_cuda = _as_5d_broadcastable_contiguous_for_cuda(scale, B, C, T, H, W, x.device, x.dtype)
                shift_cuda = _as_5d_broadcastable_contiguous_for_cuda(shift, B, C, T, H, W, x.device, x.dtype)

                return ext.gather3d_rmsnorm(
                    x_cuda,
                    r,
                    s,
                    active_indices_cuda,
                    gamma_cuda,
                    bias_cuda,
                    eps,
                    scale_cuda,
                    shift_cuda,
                    activation_name,
                )

        except Exception:
            pass

    device = x.device
    dtype = x.dtype

    active_indices = active_indices.to(device=device)

    bi_h = active_indices[:, 0].to(torch.int64)  # [N]
    bi_w = active_indices[:, 1].to(torch.int64)  # [N]

    dh = torch.arange(r, device=device, dtype=torch.int64)  # [r]
    dw = torch.arange(s, device=device, dtype=torch.int64)  # [s]
    dH, dW = torch.meshgrid(dh, dw, indexing="ij")  # [r,s]

    hh = bi_h[:, None, None] + dH[None, :, :]  # [N,r,s]
    ww = bi_w[:, None, None] + dW[None, :, :]  # [N,r,s]

    valid = (hh >= 0) & (hh < H) & (ww >= 0) & (ww < W)  # [N,r,s]
    valid_f = valid.to(dtype)

    HW = H * W
    idx_hw = (hh * W + ww)  # [N,r,s]
    idx_hw_clamped = idx_hw.clamp(0, HW - 1)

    K = num_active * r * s

    # ---------- gather x ----------
    x_flat = x.reshape(B, C, T, HW)  # [B,C,T,HW]
    idx_x = idx_hw_clamped.reshape(1, 1, 1, K).expand(B, C, T, K)
    z = torch.gather(x_flat, -1, idx_x).reshape(B, C, T, num_active, r, s)
    z = z * valid_f[None, None, None, :, :, :]  # invalid -> 0

    if not is_cache_gather:
        # ---------- optional RMSNorm ----------
        if rms_norm_fn is not None:
            z2 = z.reshape(B, C, -1).unsqueeze(-1).unsqueeze(-1)  # [B,C,*,1,1]
            z2 = rms_norm_fn(z2)
            z = z2.squeeze(-1).squeeze(-1).reshape(B, C, T, num_active, r, s)

        # ---------- scale / shift ----------
        if scale is not None:
            scale5 = _as_5d_broadcastable(scale, B, C, T, H, W, device, dtype)
            scale_flat = scale5.reshape(B, C, T, HW)
            scale_vals = torch.gather(scale_flat, -1, idx_x).reshape(B, C, T, num_active, r, s)
            scale_vals = scale_vals * valid_f[None, None, None, :, :, :] + (
                1.0 - valid_f[None, None, None, :, :, :]
            )
            z = z * scale_vals

        if shift is not None:
            shift5 = _as_5d_broadcastable(shift, B, C, T, H, W, device, dtype)
            shift_flat = shift5.reshape(B, C, T, HW)
            shift_vals = torch.gather(shift_flat, -1, idx_x).reshape(B, C, T, num_active, r, s)
            shift_vals = shift_vals * valid_f[None, None, None, :, :, :]
            z = z + shift_vals

        z = activation(z, activation_name)
        z = z * valid_f[None, None, None, :, :, :]  # ensure invalid -> 0

    # [B,C,T,N,r,s] -> [B*N,C,T,r,s]
    return z.permute(0, 3, 1, 2, 4, 5).reshape(B * num_active, C, T, r, s)
