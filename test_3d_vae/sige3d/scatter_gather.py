from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from .base import SIGEModule3d, SIGEModuleWrapper
from .flow_cache_utils import forward_warp_cache_5d
from .gather import Gather3d
from .torch_kernels import get_scatter_map, scatter3d, scatter_gather3d


class ScatterGather3d(SIGEModule3d):
    def __init__(
        self,
        gather: Gather3d,
        activation_name: str = "identity",
        activation_first: bool = False,
        rms_norm: Optional[nn.Module] = None,   # ✅ 新增
    ):
        super().__init__()
        self.gather = SIGEModuleWrapper(gather)
        self.activation_name = activation_name
        self.activation_first = activation_first

        self.scatter_map: torch.Tensor | None = None
        self.output_res = None
        self.original_outputs = {}

        if rms_norm is not None:
            self.rms_norm_fn = rms_norm.forward
        else:
            self.rms_norm_fn = None

    def clear_cache(self):
        self.original_outputs = {}

    def flow_cache(self, flow):
        if flow is None or not self.original_outputs:
            return
        for cache_id, cached in list(self.original_outputs.items()):
            self.original_outputs[cache_id] = forward_warp_cache_5d(cached, flow).contiguous()

    def forward(
        self, x: torch.Tensor, scale: Optional[torch.Tensor] = None, shift: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        self.check_dtype(x, scale, shift)
        self.check_dim(x, scale, shift)

        active_indices = self.gather.module.active_indices
        block_size = self.gather.module.block_size

        if self.mode == "profile":
            if self.cache_id not in self.original_outputs:
                raise RuntimeError("ScatterGather3d requires a full forward baseline before profile mode.")
            b = int(self.original_outputs[self.cache_id].size(0))
            t = int(self.original_outputs[self.cache_id].size(2))
            _, c, _, _, _ = x.shape
            return torch.full(
                (b * active_indices.size(0), c, t, *block_size),
                fill_value=x[0, 0, 0, 0, 0],
                dtype=x.dtype,
                device=x.device,
            )

        if self.mode == "full":
            output = x
            self.output_res = output.shape[2:]  # (T,H,W)
            self.original_outputs[self.cache_id] = output.contiguous()
            return output

        if self.mode == "sparse":
            if self.cache_id not in self.original_outputs:
                raise RuntimeError("ScatterGather3d requires a full forward baseline before sparse mode.")
            if self.scatter_map is None:
                raise RuntimeError("scatter_map is not set. Call set_masks() first.")
            output = scatter_gather3d(
                x.contiguous(),
                self.original_outputs[self.cache_id].contiguous(),
                block_size[0],
                block_size[1],
                active_indices.contiguous(),
                self.scatter_map.contiguous(),
                None if scale is None else scale.contiguous(),
                None if shift is None else shift.contiguous(),
                self.activation_name,
                self.activation_first,
                rms_norm_fn=self.rms_norm_fn,   # ✅ 改名，传函数
            )
            if self.sparse_update:
                updated = scatter3d(
                    x.contiguous(),
                    self.original_outputs[self.cache_id].contiguous(),
                    self.gather.module.offset[0],
                    self.gather.module.offset[1],
                    self.gather.module.model_stride[0],
                    self.gather.module.model_stride[1],
                    active_indices.contiguous(),
                    None,
                )
                self.original_outputs[self.cache_id].copy_(updated)
            return output

        raise NotImplementedError(f"Unknown mode: {self.mode}")

    def set_mask(self, masks: Dict, cache: Dict, timestamp: int):
        if self.timestamp == timestamp:
            return
        super().set_mask(masks, cache, timestamp)

        self.gather.module.set_mask(masks, cache, timestamp)

        mask = self.gather.module.mask
        if mask is None:
            raise RuntimeError("Gather3d.mask is not set.")
        h, w = int(mask.size(0)), int(mask.size(1))
        block_size = self.gather.module.block_size
        kernel_size = self.gather.module.kernel_size
        offset = self.gather.module.offset
        stride = self.gather.module.model_stride

        key = ("scatter_map_3d", h, w, *block_size, *kernel_size, *offset, *stride)
        scatter_map = cache.get(key, None)
        if scatter_map is None:
            scatter_map = get_scatter_map(
                h,
                w,
                block_size[0],
                block_size[1],
                kernel_size[0],
                kernel_size[1],
                offset[0],
                offset[1],
                stride[0],
                stride[1],
                self.gather.module.active_indices,
            )
            cache[key] = scatter_map
        self.scatter_map = scatter_map
