from __future__ import annotations

from typing import Optional

import torch
from utils.flow_cache_utils import forward_warp_cache_5d

from .base import SIGEModule3d, SIGEModuleWrapper
from .gather3d import Gather3d
from .torch_kernels import scatter3d, scatter_with_block_residual3d


class Scatter3d(SIGEModule3d):
    def __init__(self, gather: Gather3d):
        super().__init__()
        self.gather = SIGEModuleWrapper(gather)
        self.output_res = None
        # [B, C, T, H, W]
        self.original_outputs = {}

    def clear_cache(self):
        self.original_outputs = {}

    # flow: (H, W, 2), 全是 (dx, dy)
    # 传入的flow是反向光流
    def flow_cache(self, flow):
        """
        将 original_outputs 中的缓存特征按 flow 做反向采样（backward warp）+ 双线性插值
        - 输入缓存: [B, C, T, H, W]
        - flow: [H, W, 2]，每个位置是 (dx, dy)（像素位移，允许非整数）
        - 输出：同形状，output[..., y, x] 从 input[..., y+dy, x+dx] 采样（越界按 0 填充）
        - 对每个时间帧 T 做同样的空间采样（flow 仅依赖 H,W）
        """
        if flow is None or not self.original_outputs:
            return

        for cache_id, cached in list(self.original_outputs.items()):
            self.original_outputs[cache_id] = forward_warp_cache_5d(cached, flow).contiguous()


    def forward(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.check_dtype(x, residual)
        self.check_dim(x, residual)

        if self.mode == "profile":
            if self.cache_id not in self.original_outputs:
                raise RuntimeError("Scatter3d requires a full forward baseline before profile mode.")
            _, c, t, _, _ = x.shape
            b = int(self.original_outputs[self.cache_id].size(0))
            output = torch.full(
                (b, c, t, *self.output_res[1:]),
                fill_value=x[0, 0, 0, 0, 0],
                dtype=x.dtype,
                device=x.device,
            )
            if residual is not None:
                output = output + residual
            return output

        if self.mode == "full":
            output = x if residual is None else x + residual
            self.output_res = output.shape[2:]  # (T,H,W)
            self.original_outputs[self.cache_id] = output.contiguous()
            return output

        if self.mode == "sparse":
            if self.cache_id not in self.original_outputs:
                raise RuntimeError("Scatter3d requires a full forward baseline before sparse mode.")
            active_indices = self.gather.module.active_indices
            if active_indices is None:
                raise RuntimeError("Active indices are not set for sparse mode.")

            offset_h, offset_w = self.gather.module.offset
            stride_h, stride_w = self.gather.module.model_stride

            output = scatter3d(
                x.contiguous(),
                self.original_outputs[self.cache_id].contiguous(),
                offset_h,
                offset_w,
                stride_h,
                stride_w,
                active_indices.contiguous(),
                None if residual is None else residual.contiguous(),
            )
            if self.sparse_update:
                self.original_outputs[self.cache_id].copy_(output.contiguous())
            return output

        raise NotImplementedError(f"Unknown mode: {self.mode}")


class ScatterWithBlockResidual3d(SIGEModule3d):
    def __init__(self, main_gather: Gather3d, shortcut_gather: Gather3d):
        super().__init__()
        self.main_gather = SIGEModuleWrapper(main_gather)
        self.shortcut_gather = SIGEModuleWrapper(shortcut_gather)
        self.output_res = None
        self.original_outputs = {}
        self.original_residuals = {}

    def clear_cache(self):
        self.original_outputs = {}
        self.original_residuals = {}

    def flow_cache(self, flow):
        if flow is None:
            return
        if self.original_outputs:
            for cache_id, cached in list(self.original_outputs.items()):
                self.original_outputs[cache_id] = forward_warp_cache_5d(cached, flow).contiguous()
        if self.original_residuals:
            for cache_id, cached in list(self.original_residuals.items()):
                self.original_residuals[cache_id] = forward_warp_cache_5d(cached, flow).contiguous()

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        self.check_dtype(x, residual)
        self.check_dim(x, residual)

        if self.mode == "profile":
            if self.cache_id not in self.original_outputs:
                raise RuntimeError("ScatterWithBlockResidual3d requires a full forward baseline before profile mode.")
            _, c, t, _, _ = x.shape
            b = int(self.original_outputs[self.cache_id].size(0))
            return torch.full(
                (b, c, t, *self.output_res[1:]),
                fill_value=x[0, 0, 0, 0, 0] + residual[0, 0, 0, 0, 0],
                dtype=x.dtype,
                device=x.device,
            )

        if self.mode == "full":
            output = x + residual
            self.output_res = output.shape[2:]  # (T,H,W)
            self.original_outputs[self.cache_id] = output.contiguous()
            self.original_residuals[self.cache_id] = residual.contiguous()
            return output

        if self.mode == "sparse":
            if self.cache_id not in self.original_outputs or self.cache_id not in self.original_residuals:
                raise RuntimeError("ScatterWithBlockResidual3d requires full forward baseline before sparse mode.")

            offset_h, offset_w = self.main_gather.module.offset
            stride_h, stride_w = self.main_gather.module.model_stride

            output = scatter_with_block_residual3d(
                x.contiguous(),
                self.original_outputs[self.cache_id].contiguous(),
                residual.contiguous(),
                self.original_residuals[self.cache_id].contiguous(),
                offset_h,
                offset_w,
                stride_h,
                stride_w,
                self.main_gather.module.active_indices.contiguous(),
                self.shortcut_gather.module.active_indices.contiguous(),
            )

            if self.sparse_update:
                self.original_outputs[self.cache_id].copy_(output.contiguous())
                updated_residual = scatter3d(
                    residual.contiguous(),
                    self.original_residuals[self.cache_id].contiguous(),
                    self.shortcut_gather.module.offset[0],
                    self.shortcut_gather.module.offset[1],
                    self.shortcut_gather.module.model_stride[0],
                    self.shortcut_gather.module.model_stride[1],
                    self.shortcut_gather.module.active_indices.contiguous(),
                    None,
                )
                self.original_residuals[self.cache_id].copy_(updated_residual.contiguous())
            return output

        raise NotImplementedError(f"Unknown mode: {self.mode}")
