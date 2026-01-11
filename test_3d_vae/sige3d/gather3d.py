from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple, Union

import torch
from torch import nn

from utils.mask_utils import reduce_mask
from .base import SIGEModule3d
from .torch_kernels import gather3d


class Gather3d(SIGEModule3d):
    def __init__(
        self,
        conv: nn.Conv3d,
        block_size: Union[int, Tuple[int, int]],
        offset: Optional[Union[int, Tuple[int, int]]] = None,
        activation_name: str = "identity",
        activation_first: bool = False,
        verbose: bool = False,
        rms_norm: Optional[nn.Module] = None,   # ✅ 新增
    ):
        super().__init__()

        if isinstance(block_size, int):
            block_size = (block_size, block_size)

        kernel_size = conv.kernel_size if isinstance(conv.kernel_size, tuple) else (conv.kernel_size,) * 3
        stride = conv.stride if isinstance(conv.stride, tuple) else (conv.stride,) * 3

        # Only spatial (H,W) participates in block partitioning.
        k_h, k_w = int(kernel_size[1]), int(kernel_size[2])
        s_h, s_w = int(stride[1]), int(stride[2])

        n0 = max(block_size[0] - k_h, 0) // s_h
        n1 = max(block_size[1] - k_w, 0) // s_w
        b0 = n0 * s_h + k_h
        b1 = n1 * s_w + k_w
        if (b0, b1) != block_size:
            warnings.warn(f"Change the block size from {block_size} to {(b0, b1)}")

        self.model_stride = (s_h, s_w)
        self.kernel_size = (k_h, k_w)
        self.block_size = (b0, b1)
        self.block_stride = ((n0 + 1) * s_h, (n1 + 1) * s_w)

        if offset is None:
            spatial_padding = getattr(conv, "spatial_padding", None)
            if spatial_padding is None:
                pad = conv.padding if isinstance(conv.padding, tuple) else (conv.padding,) * 3
                spatial_padding = (int(pad[1]), int(pad[2]))
            self.offset = (int(spatial_padding[0]), int(spatial_padding[1]))
        else:
            if isinstance(offset, int):
                offset = (offset, offset)
            self.offset = (int(offset[0]), int(offset[1]))

        self.activation_name = activation_name
        self.activation_first = activation_first
        self.verbose = verbose

        self.input_res: Optional[Tuple[int, int]] = None
        self.active_indices: Optional[torch.Tensor] = None

        # rms_norm: Optional[nn.Module]
        if rms_norm is not None:
            self.rms_norm_fn = rms_norm.forward   # ✅ 只存函数
        else:
            self.rms_norm_fn = None


    def forward(
        self,
        x: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        shift: Optional[torch.Tensor] = None,
        is_cache_gather: Optional[bool] = False,
    ) -> torch.Tensor:

        self.check_dtype(x, scale, shift)
        self.check_dim(x, scale, shift)
        b, c, t, _, _ = x.shape

        if self.mode == "profile":
            if self.active_indices is None:
                raise RuntimeError("Active indices are not set for profile mode.")
            output = torch.full(
                (b * self.active_indices.size(0), c, t, *self.block_size),
                fill_value=x[0, 0, 0, 0, 0],
                dtype=x.dtype,
                device=x.device,
            )
            if scale is not None:
                output = output * scale.reshape(scale.size(0), scale.size(1), 1, 1, 1).repeat_interleave(
                    self.active_indices.size(0), dim=0
                )
            if shift is not None:
                output = output + shift.reshape(shift.size(0), shift.size(1), 1, 1, 1).repeat_interleave(
                    self.active_indices.size(0), dim=0
                )
            return output

        if self.mode == "full":
            self.input_res = (int(x.size(3)), int(x.size(4)))
            assert scale is None
            assert shift is None
            return x

        if self.mode == "sparse":
            if self.active_indices is None:
                raise RuntimeError("Active indices are not set for sparse mode.")
            return gather3d(
                x.contiguous(),
                self.block_size[0],
                self.block_size[1],
                self.active_indices.contiguous(),
                None if scale is None else scale.contiguous(),
                None if shift is None else shift.contiguous(),
                self.activation_name,
                self.activation_first,
                rms_norm_fn=self.rms_norm_fn,   # ✅ 改名，传函数
                is_cache_gather=is_cache_gather # 不需要norm和激活  
            )

        raise NotImplementedError(f"Unknown mode: {self.mode}")

    def set_mask(self, masks: Dict, cache: Dict, timestamp: int):
        if self.timestamp == timestamp:
            return
        super().set_mask(masks, cache, timestamp)
        if self.input_res is None:
            raise RuntimeError("Input resolution is not set before set_mask(). Run one full forward first.")

        res = (int(self.input_res[0]), int(self.input_res[1]))
        mask = masks[res]
        self.mask = mask

        key = ("active_indices_3d", *res, *self.block_size, *self.block_stride, *self.offset)
        active_indices = cache.get(key, None)
        if active_indices is None:
            active_indices = reduce_mask(mask, self.block_size, self.block_stride, self.offset, verbose=self.verbose)
            cache[key] = active_indices
        self.active_indices = active_indices

