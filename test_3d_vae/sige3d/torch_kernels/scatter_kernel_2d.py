import torch


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
    """PyTorch reference for sige/cuda/scatter_kernel.cu."""
    b, c, h, w = y.shape
    num_active = active_indices.size(0)
    r, s = x.size(2), x.size(3)

    # 不能直接修改y
    # 这张图 必须保持不变, 是full前向算出来的 pre-computed 底图
    output = y.clone()

    if num_active == 0:
        return output

    x_blocks = x.reshape(b, num_active, c, r, s)
    residual_y = residual.expand_as(y) if residual is not None else None

    for ib, (ai_h, ai_w) in enumerate(active_indices.tolist()):
        bi_h = (offset_h + ai_h) // stride_h
        bi_w = (offset_w + ai_w) // stride_w
        h0 = max(bi_h, 0)
        h1 = min(bi_h + r, h)
        w0 = max(bi_w, 0)
        w1 = min(bi_w + s, w)
        if h0 >= h1 or w0 >= w1:
            continue
        dh0 = h0 - bi_h
        dh1 = dh0 + (h1 - h0)
        dw0 = w0 - bi_w
        dw1 = dw0 + (w1 - w0)

        block = x_blocks[:, ib, :, dh0:dh1, dw0:dw1]
        if residual_y is not None:
            block = block + residual_y[:, :, h0:h1, w0:w1]
        output[:, :, h0:h1, w0:w1] = block

    return output


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
    """PyTorch reference for scatter_with_block_residual in scatter_kernel.cu."""
   
    '''
    x0: 主分支稀疏块的输出(main branch blocks)

    y0: 主分支对应的“基底大图”

    x1: shortcut 分支稀疏块输出(shortcut blocks)

    y1: shortcut 分支的“基底大图”

    其实输入的y0 = y0 + y1
    elif self.mode == "full":
        output = x + residual
        self.original_outputs[self.cache_id] = output.contiguous()
        self.original_residuals[self.cache_id] = residual.contiguous()
    
    
    active_indices0: 主分支激活块的位置

    active_indices1: shortcut 分支激活块的位置
    '''
    
    output = scatter(
        x0,
        y0,
        offset_h,
        offset_w,
        stride_h,
        stride_w,
        active_indices0,
        y1,   # scatter x0 + y1
    )

    b, c, h, w = y1.shape
    num_active = active_indices1.size(0)
    if num_active == 0:
        return output

    r, s = x1.size(2), x1.size(3)
    x1_blocks = x1.reshape(b, num_active, c, r, s)

    for ib, (bi_h, bi_w) in enumerate(active_indices1.tolist()):
        h0 = max(bi_h, 0)
        h1 = min(bi_h + r, h)
        w0 = max(bi_w, 0)
        w1 = min(bi_w + s, w)
        if h0 >= h1 or w0 >= w1:
            continue
        dh0 = h0 - bi_h
        dh1 = dh0 + (h1 - h0)
        dw0 = w0 - bi_w
        dw1 = dw0 + (w1 - w0)

        # active_indices1 += x1 - y1
        output[:, :, h0:h1, w0:w1] += (
            x1_blocks[:, ib, :, dh0:dh1, dw0:dw1] - y1[:, :, h0:h1, w0:w1]
        )

    return output
