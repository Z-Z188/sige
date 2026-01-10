import torch

from sige.nn.utils import activation


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
    """PyTorch reference for get_scatter_map in scatter_gather_kernel.cu."""
    scatter_map = torch.full(
        (h, w, 3),
        -1,
        dtype=torch.int32,
        device=active_indices.device,
    )
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


def scatter_gather(
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
    """PyTorch reference for scatter_gather in scatter_gather_kernel.cu."""
    b, c, h, w = y.shape
    num_active = active_indices.size(0)
    ro, so = int(b_size_h), int(b_size_w)
    rx, sx = x.size(2), x.size(3)

    output = torch.zeros((b, num_active, c, ro, so), dtype=x.dtype, device=x.device)
    if num_active == 0:
        return output.reshape(b * num_active, c, ro, so)

    x_blocks = x.reshape(b, num_active, c, rx, sx)
    scale_y = scale.expand_as(y) if scale is not None else None
    shift_y = shift.expand_as(y) if shift is not None else None

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
                    z = x_blocks[:, bx, :, hx, wx]
                else:
                    z = y[:, :, hh, ww]
                if not activation_first:
                    if scale_y is not None:
                        z = z * scale_y[:, :, hh, ww]
                    if shift_y is not None:
                        z = z + shift_y[:, :, hh, ww]
                    z = activation(z, activation_name)
                else:
                    z = activation(z, activation_name)
                    if scale_y is not None:
                        z = z * scale_y[:, :, hh, ww]
                    if shift_y is not None:
                        z = z + shift_y[:, :, hh, ww]
                output[:, ib, :, intra_bh, intra_bw] = z

    return output.reshape(b * num_active, c, ro, so)
