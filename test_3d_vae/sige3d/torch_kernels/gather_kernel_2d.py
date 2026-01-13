import torch

from ..activation import activation
from .backend import use_cuda_kernels


# slow
# def gather2d(
#     x: torch.Tensor,
#     b_size_h: int,
#     b_size_w: int,
#     active_indices: torch.Tensor,
#     scale: torch.Tensor | None = None,
#     shift: torch.Tensor | None = None,
#     activation_name: str = "identity",
#     activation_first: bool = False,
# ) -> torch.Tensor:
#     """PyTorch reference for sige/cuda/gather_kernel.cu."""
#
#     # print("*" * 40)
#     # print("Use Pytorch!!!")
#     # print("*" * 40)
#
#     # 输入 x 没有 padding
#     b, c, h, w = x.shape
#     num_active = active_indices.size(0)
#     r, s = int(b_size_h), int(b_size_w)
#
#     output = torch.zeros((b, num_active, c, r, s), dtype=x.dtype, device=x.device)
#     if num_active == 0:
#         return output.view(b * num_active, c, r, s)
#
#     scale_x = scale.expand_as(x) if scale is not None else None
#     shift_x = shift.expand_as(x) if shift is not None else None
#
#     for ib, (bi_h, bi_w) in enumerate(active_indices.tolist()):
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
#         block = x[:, :, h0:h1, w0:w1]
#         if not activation_first:
#             if scale_x is not None:
#                 block = block * scale_x[:, :, h0:h1, w0:w1]
#             if shift_x is not None:
#                 block = block + shift_x[:, :, h0:h1, w0:w1]
#             block = activation(block, activation_name)
#         else:
#             block = activation(block, activation_name)
#             if scale_x is not None:
#                 block = block * scale_x[:, :, h0:h1, w0:w1]
#             if shift_x is not None:
#                 block = block + shift_x[:, :, h0:h1, w0:w1]
#
#         # output.shape == [1, 785, 16, 6, 6]
#         # block.shape  == [1, 16, 5, 5]
#         # output 左上角那一整圈（第 0 行、第 0 列）确实没有任何 x 的元素填进去
#         # 全是 0
#         output[:, ib, :, dh0:dh1, dw0:dw1] = block
#
#     # gather 的输出就是后续 conv2d / conv3d 的输入 block
#     return output.view(b * num_active, c, r, s)


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


def gather2d(
    x: torch.Tensor,
    b_size_h: int,
    b_size_w: int,
    active_indices: torch.Tensor,
    scale: torch.Tensor | None = None,
    shift: torch.Tensor | None = None,
    activation_name: str = "identity",
    activation_first: bool = False,
) -> torch.Tensor:
    """Kernel backend: CUDA extension (optional) or torch vectorized."""

    # 输入 x 没有 padding
    B, C, H, W = x.shape
    num_active = int(active_indices.size(0))
    r, s = int(b_size_h), int(b_size_w)

    if num_active == 0:
        return x.new_zeros((0, C, r, s))

    if x.is_cuda and use_cuda_kernels():
        try:
            from ..cuda_kernels import get_extension

            ext = get_extension()
            active_indices_cuda = active_indices.to(device=x.device, dtype=torch.int32).contiguous()
            scale_cuda = _as_4d_broadcastable_contiguous_for_cuda(scale, B, C, H, W, x.device, x.dtype)
            shift_cuda = _as_4d_broadcastable_contiguous_for_cuda(shift, B, C, H, W, x.device, x.dtype)
            return ext.gather2d(
                x.contiguous(),
                r,
                s,
                active_indices_cuda,
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
    x_flat = x.reshape(B, C, HW)
    idx_x = idx_hw_clamped.reshape(1, 1, K).expand(B, C, K)
    z = torch.gather(x_flat, -1, idx_x).reshape(B, C, num_active, r, s)
    z = z * valid_f[None, None, :, :, :]  # invalid -> 0

    # ---------- gather scale/shift ----------
    scale4 = _as_4d_broadcastable(scale, B, C, H, W, device, dtype) if scale is not None else None
    shift4 = _as_4d_broadcastable(shift, B, C, H, W, device, dtype) if shift is not None else None

    if scale4 is not None:
        scale_flat = scale4.reshape(B, C, HW)
        scale_vals = torch.gather(scale_flat, -1, idx_x).reshape(B, C, num_active, r, s)
        scale_vals = scale_vals * valid_f[None, None, :, :, :] + (1.0 - valid_f[None, None, :, :, :])
    else:
        scale_vals = None

    if shift4 is not None:
        shift_flat = shift4.reshape(B, C, HW)
        shift_vals = torch.gather(shift_flat, -1, idx_x).reshape(B, C, num_active, r, s)
        shift_vals = shift_vals * valid_f[None, None, :, :, :]
    else:
        shift_vals = None

    # ---------- activation / scale / shift ----------
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

    # [B,C,N,r,s] -> [B*N,C,r,s]
    return z.permute(0, 2, 1, 3, 4).reshape(B * num_active, C, r, s)

'''
output block (6×6):

      0   1   2   3   4   5
    ┌─────────────────────┐
0   │ 0   0   0   0   0   0 │  ← padding 行
1   │ 0  x00 x01 x02 x03 x04│
2   │ 0  x10 x11 x12 x13 x14│
3   │ 0  x20 x21 x22 x23 x24│
4   │ 0  x30 x31 x32 x33 x34│
5   │ 0  x40 x41 x42 x43 x44│
    └─────────────────────┘
    
'''
