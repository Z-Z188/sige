from __future__ import annotations

import torch

from ..activation import activation
from .backend import use_cuda_kernels

# slow
# def get_scatter_map(
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
#     """
#     3D extension of get_scatter_map in `sige/cuda/scatter_gather_kernel.py`.
#     Still a pure spatial (H,W) map: [H, W, 3] -> (block_id, intra_h, intra_w).
#     """
#     scatter_map = torch.full((h, w, 3), -1, dtype=torch.int32, device=active_indices.device)
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
def get_scatter_map(
    h, w,
    b_size_h, b_size_w,
    k_size_h, k_size_w,
    offset_h, offset_w,
    stride_h, stride_w,
    active_indices: torch.Tensor,
):
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

    # [N]
    ai_h = active_indices[:, 0].to(torch.int64)
    ai_w = active_indices[:, 1].to(torch.int64)

    # [N] 计算 block 左上角对应的输出基准位置
    bi_h = (offset_h + ai_h) // stride_h
    bi_w = (offset_w + ai_w) // stride_w

    N = active_indices.shape[0]
    ib = torch.arange(N, device=device, dtype=torch.int64)  # [N]

    # intra offsets
    intra_bh = torch.arange(r, device=device, dtype=torch.int64)  # [r]
    intra_bw = torch.arange(s, device=device, dtype=torch.int64)  # [s]

    # 生成所有覆盖位置：
    # hh: [N, r, 1] + [1, r, 1] -> [N, r, 1]
    hh = bi_h[:, None, None] + intra_bh[None, :, None]  # [N, r, 1]
    ww = bi_w[:, None, None] + intra_bw[None, None, :]  # [N, 1, s]

    # broadcast 到 [N, r, s]
    hh = hh.expand(N, r, s)
    ww = ww.expand(N, r, s)

    # 边界过滤
    valid = (hh >= 0) & (hh < h) & (ww >= 0) & (ww < w)
    hh = hh[valid]
    ww = ww[valid]

    # 同步生成对应的写入值
    # block_id：每个 (r,s) 都对应同一个 ib
    block_id = ib[:, None, None].expand(N, r, s)[valid]
    intra_h  = intra_bh[None, :, None].expand(N, r, s)[valid]
    intra_w  = intra_bw[None, None, :].expand(N, r, s)[valid]

    # 写入（一次性）
    scatter_map[hh, ww, 0] = block_id.to(torch.int32)
    scatter_map[hh, ww, 1] = intra_h.to(torch.int32)
    scatter_map[hh, ww, 2] = intra_w.to(torch.int32)

    return scatter_map





# def scatter_gather3d(
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
#     rms_norm_fn = None,   # ✅ 新增：可传 RMS_norm forward

# ) -> torch.Tensor:
#     """
#     3D (T,H,W) extension of `sige/cuda/scatter_gather_kernel.py` (torch reference).

#     - `x`: updated conv outputs as blocks, shape [B*num_active, C, T, rx, sx]
#     - `y`: baseline full tensor, shape [B, C, T, H, W]
#     - returns: gathered blocks for the next conv, shape [B*num_active, C, T, b_size_h, b_size_w]
#     """
#     b, c, t, h, w = y.shape
#     num_active = int(active_indices.size(0))
#     ro, so = int(b_size_h), int(b_size_w)
#     rx, sx = int(x.size(-2)), int(x.size(-1))

#     # output blocks: [B, num_active, C, T, ro, so] -> flatten to [B*num_active, C, T, ro, so]
#     output = torch.zeros((b, num_active, c, t, ro, so), dtype=x.dtype, device=x.device)
#     if num_active == 0:
#         return output.view(b * num_active, c, t, ro, so)

#     x_blocks = x.view(b, num_active, c, t, rx, sx)

#     if scatter_map.device.type != "cpu":
#         scatter_map_cpu = scatter_map.detach().cpu()
#     else:
#         scatter_map_cpu = scatter_map

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
#                     # z: [B, C, T]
#                     z = x_blocks[:, bx, :, :, hx, wx]
#                 else:
#                     z = y[:, :, :, hh, ww]

#                 if rms_norm_fn is not None:
#                     # z: [B, C, T] -> [B, C, T, 1, 1]，让 gamma: [C,1,1] 能正确广播
#                     z = rms_norm_fn(z.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)

#                 if not activation_first:
#                     if scale is not None:
#                         z = z * scale[:, :, :, hh, ww]
#                     if shift is not None:
#                         z = z + shift
#                     z = activation(z, activation_name)
#                 else:
#                     z = activation(z, activation_name)
#                     if scale is not None:
#                         z = z * scale
#                     if shift is not None:
#                         z = z + shift
#                 output[:, ib, :, :, intra_bh, intra_bw] = z

#     return output.view(b * num_active, c, t, ro, so)






def _as_5d_broadcastable(v: torch.Tensor, B: int, C: int, T: int, H: int, W: int,
                        device, dtype) -> torch.Tensor:
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

    if v.dim() == 0:          # []
        v = v.view(1, 1, 1, 1, 1)
    elif v.dim() == 1:        # [C]
        v = v.view(1, C, 1, 1, 1)
    elif v.dim() == 2:        # [B,C] or [1,C]
        v = v.view(v.size(0), v.size(1), 1, 1, 1)
    elif v.dim() == 3:        # [B,C,T]
        v = v.view(v.size(0), v.size(1), v.size(2), 1, 1)
    elif v.dim() == 4:        # [B,C,H,W]
        v = v.unsqueeze(2)    # -> [B,C,1,H,W]
    elif v.dim() == 5:        # [B,C,T,H,W]
        pass
    else:
        raise ValueError(f"scale/shift dim not supported: {v.shape}")

    target = (B, C, T, H, W)
    exp = []
    for cur, tar in zip(v.shape, target):
        if cur == tar:
            exp.append(cur)
        elif cur == 1:
            exp.append(tar)
        else:
            raise ValueError(f"not broadcastable: {v.shape} -> {target}")
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

    # Avoid passing expanded views to the extension (it assumes contiguous indexing).
    return v.contiguous()


def scatter_gather3d(
    x: torch.Tensor,                 # [B*num_active, C, T, Rx, Sx]
    y: torch.Tensor,                 # [B, C, T, H, W]
    b_size_h: int,
    b_size_w: int,
    active_indices: torch.Tensor,    # [num_active, 2]  (bi_h, bi_w)
    scatter_map: torch.Tensor,       # [H, W, 3] (bx, hx, wx)
    scale: torch.Tensor | None = None,
    shift: torch.Tensor | None = None,
    activation_name: str = "identity",
    activation_first: bool = False,
    rms_norm_fn=None,               # callable: rms_norm_fn(tensor)->tensor
) -> torch.Tensor:
    """
    向量化版本（无 Python 循环）:
    - 输出: [B*num_active, C, T, b_size_h, b_size_w]
    """
    B, C, T, H, W = y.shape
    num_active = int(active_indices.size(0))
    Ro, So = int(b_size_h), int(b_size_w)
    Rx, Sx = int(x.size(-2)), int(x.size(-1))

    if num_active == 0:
        return x.new_zeros((0, C, T, Ro, So))

    if x.is_cuda and use_cuda_kernels():
        try:
            from ..cuda_kernels import get_extension

            ext = get_extension()
            device = x.device
            dtype = x.dtype

            active_indices_cuda = active_indices.to(device=device, dtype=torch.int32).contiguous()
            scatter_map_cuda = scatter_map.to(device=device, dtype=torch.int32).contiguous()

            if rms_norm_fn is None:
                scale_cuda = _as_5d_broadcastable_contiguous_for_cuda(scale, B, C, T, H, W, device, dtype)
                shift_cuda = _as_5d_broadcastable_contiguous_for_cuda(shift, B, C, T, H, W, device, dtype)
                return ext.scatter_gather3d(
                    x.contiguous(),
                    y.contiguous(),
                    int(Ro),
                    int(So),
                    active_indices_cuda,
                    scatter_map_cuda,
                    scale_cuda,
                    shift_cuda,
                    activation_name,
                    activation_first,
                )

            # RMSNorm path: fully fused in CUDA (x/y selection + RMSNorm + scale/shift + activation).
            norm_module = getattr(rms_norm_fn, "__self__", None)
            if norm_module is not None and hasattr(ext, "scatter_gather3d_rmsnorm"):
                if not bool(getattr(norm_module, "channel_first", True)):
                    raise RuntimeError("CUDA RMSNorm path requires channel_first=True")
                eps = float(getattr(norm_module, "eps", 1e-6))
                gamma = getattr(norm_module, "gamma", None)
                if gamma is None:
                    raise RuntimeError("rms_norm_fn has no gamma parameter")

                bias = getattr(norm_module, "bias", None)
                bias_tensor = bias if torch.is_tensor(bias) else None

                gamma_cuda = gamma.to(device=device, dtype=dtype).reshape(-1).contiguous()
                bias_cuda = None if bias_tensor is None else bias_tensor.to(device=device, dtype=dtype).reshape(-1).contiguous()

                scale_cuda = _as_5d_broadcastable_contiguous_for_cuda(scale, B, C, T, H, W, device, dtype)
                shift_cuda = _as_5d_broadcastable_contiguous_for_cuda(shift, B, C, T, H, W, device, dtype)

                return ext.scatter_gather3d_rmsnorm(
                    x.contiguous(),
                    y.contiguous(),
                    int(Ro),
                    int(So),
                    active_indices_cuda,
                    scatter_map_cuda,
                    gamma_cuda,
                    bias_cuda,
                    eps,
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
    dh = torch.arange(Ro, device=device, dtype=torch.int64)  # [Ro]
    dw = torch.arange(So, device=device, dtype=torch.int64)  # [So]
    dH, dW = torch.meshgrid(dh, dw, indexing="ij")           # [Ro,So]

    hh = bi_h[:, None, None] + dH[None, :, :]  # [N,Ro,So]
    ww = bi_w[:, None, None] + dW[None, :, :]  # [N,Ro,So]

    valid = (hh >= 0) & (hh < H) & (ww >= 0) & (ww < W)      # [N,Ro,So]
    HW = H * W
    idx_hw = (hh * W + ww)                                   # [N,Ro,So]
    idx_hw_clamped = idx_hw.clamp(0, HW - 1)                 # 防越界 gather

    K = num_active * Ro * So

    # ---------- 2) 查 scatter_map 得到 (bx,hx,wx) ----------
    sm_flat = scatter_map.view(-1, 3).to(torch.int64)        # [HW,3]
    sm = sm_flat[idx_hw_clamped]                             # [N,Ro,So,3]
    bx = sm[..., 0]  # [N,Ro,So]
    hx = sm[..., 1]
    wx = sm[..., 2]
    # invalid 位置强制 bx = -1（后面保证输出为 0）
    bx = torch.where(valid, bx, torch.full_like(bx, -1))

    # ---------- 3) 并行 gather y ----------
    y_flat = y.reshape(B, C, T, HW)                          # [B,C,T,HW]
    idx_y = idx_hw_clamped.reshape(1, 1, 1, K).expand(B, C, T, K)  # 逻辑扩展（不真实拷贝）
    y_vals = torch.gather(y_flat, -1, idx_y).reshape(B, C, T, num_active, Ro, So)
    y_vals = y_vals * valid[None, None, None, :, :, :]       # invalid -> 0

    # ---------- 4) 并行 gather x（把 (bx,hx,wx) 合成一个 source_idx） ----------
    # x_blocks: [B,N,C,T,Rx,Sx]
    x_blocks = x.reshape(B, num_active, C, T, Rx, Sx)
    # x_src: [B,C,T, N*Rx*Sx]
    x_src = (
        x_blocks.reshape(B, num_active, C, T, Rx * Sx)
        .permute(0, 2, 3, 1, 4)
        .reshape(B, C, T, num_active * Rx * Sx)
    )

    idx_rx = (hx * Sx + wx).clamp(0, Rx * Sx - 1)            # [N,Ro,So]
    src_idx = (bx.clamp(min=0) * (Rx * Sx) + idx_rx)         # [N,Ro,So]
    src_idx = src_idx.reshape(1, 1, 1, K).expand(B, C, T, K)

    x_vals = torch.gather(x_src, -1, src_idx).reshape(B, C, T, num_active, Ro, So)

    use_x = (bx >= 0) & valid                                 # [N,Ro,So]
    z = torch.where(use_x[None, None, None, :, :, :], x_vals, y_vals)
    z = z * valid[None, None, None, :, :, :]                  # 再保险：invalid -> 0

    # ---------- 5) 可选 RMSNorm（按你原来做法：z->[B,C,*,1,1] 调 rms_norm_fn） ----------
    if rms_norm_fn is not None:
        z2 = z.reshape(B, C, -1).unsqueeze(-1).unsqueeze(-1)  # [B,C, T*N*Ro*So,1,1]
        z2 = rms_norm_fn(z2)
        z = z2.squeeze(-1).squeeze(-1).reshape(B, C, T, num_active, Ro, So)

    # ---------- 6) scale / shift（按 broadcast 语义，先 expand 到 [B,C,T,H,W] 再 gather） ----------
    scale5 = _as_5d_broadcastable(scale, B, C, T, H, W, device, dtype) if scale is not None else None
    shift5 = _as_5d_broadcastable(shift, B, C, T, H, W, device, dtype) if shift is not None else None

    if scale5 is not None:
        scale_flat = scale5.reshape(B, C, T, HW)
        scale_vals = torch.gather(scale_flat, -1, idx_y).reshape(B, C, T, num_active, Ro, So)
        # invalid 位置让 scale=1（不改变 0）
        scale_vals = scale_vals * valid[None, None, None, :, :, :] + (1.0 - valid[None, None, None, :, :, :].to(dtype))
    else:
        scale_vals = None

    if shift5 is not None:
        shift_flat = shift5.reshape(B, C, T, HW)
        shift_vals = torch.gather(shift_flat, -1, idx_y).reshape(B, C, T, num_active, Ro, So)
        # invalid 位置让 shift=0
        shift_vals = shift_vals * valid[None, None, None, :, :, :]
    else:
        shift_vals = None

    # ---------- 7) activation / scale / shift 顺序 ----------
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

    # ---------- 8) reshape 到 [B*N, C, T, Ro, So] ----------
    out = z.permute(0, 3, 1, 2, 4, 5).reshape(B * num_active, C, T, Ro, So)
    return out
