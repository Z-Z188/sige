from typing import Optional

import torch

from utils.flow_cache_utils import forward_warp_cache_4d

from .base import SIGEModule3d, SIGEModuleWrapper
from .gather2d import Gather2d



class Scatter2d(SIGEModule3d):
    def __init__(self, gather: Gather2d, backend: Optional[str] = None):
        """
        backend:
          - None: follow `gather.backend`
          - "torch": use `sige.nn.torch_kernels`
          - "ext"/"cuda": use compiled extension (`sige.cpu` / `sige.cuda` / `sige.mps`)
        """
        super(Scatter2d, self).__init__()
        self.gather = SIGEModuleWrapper(gather)

        self.backend = backend if backend is not None else getattr(gather, "backend", "torch")
        self.load_runtime_with_backend("scatter2d", backend=self.backend)
        self.output_res = None
        self.original_outputs = {}

    def clear_cache(self):
        self.original_outputs = {}

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
            self.original_outputs[cache_id] = forward_warp_cache_4d(cached, flow).contiguous()


    def forward(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.check_dtype(x, residual)
        self.check_dim(x, residual)
        if self.mode == "profile":
            _, c, _, _ = x.shape
            output = torch.full(
                (self.original_outputs[self.cache_id].size(0), c, *self.output_res),
                fill_value=x[0, 0, 0, 0],
                dtype=x.dtype,
                device=x.device,
            )  # create a dummy scatter output depending on the input for profiling
            if residual is not None:
                output = output + residual
        elif self.mode == "full":
            if residual is None:
                output = x
            else:
                output = x + residual
            self.output_res = output.shape[2:]
            self.original_outputs[self.cache_id] = output.contiguous()
        elif self.mode == "sparse":
            device = x.device.type
            runtime = self.runtime[device]
            assert runtime is not None

            active_indices = self.gather.module.active_indices
            assert active_indices is not None
            if self.backend.lower() not in {"torch", "pytorch"} and active_indices.device != x.device:
                active_indices = active_indices.to(device=x.device)
            offset = self.gather.module.offset
            stride = self.gather.module.model_stride
            output = runtime(
                x.contiguous(),
                self.original_outputs[self.cache_id].contiguous(),
                offset[0],
                offset[1],
                stride[0],
                stride[1],
                active_indices.contiguous(),
                None if residual is None else residual.contiguous(),
            )
            if self.sparse_update:
                self.original_outputs[self.cache_id].copy_(output.contiguous())
        else:
            raise NotImplementedError("Unknown mode: [%s]!!!" % self.mode)
        return output


class ScatterWithBlockResidual2d(SIGEModule3d):
    def __init__(self, main_gather: Gather2d, shortcut_gather: Gather2d, backend: Optional[str] = None):
        super(ScatterWithBlockResidual2d, self).__init__()
        self.main_gather = SIGEModuleWrapper(main_gather)
        self.shortcut_gather = SIGEModuleWrapper(shortcut_gather)

        self.backend = backend if backend is not None else getattr(main_gather, "backend", "torch")
        self.load_runtime_with_backend("scatter_with_block_residual", backend=self.backend)
        self.scatter_runtime = None
        self.output_res = None
        self.original_outputs = {}
        self.original_residuals = {}

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        self.check_dtype(x, residual)
        self.check_dim(x, residual)
        if self.mode == "profile":
            _, c, _, _ = x.shape
            output = torch.full(
                (self.original_outputs[self.cache_id].size(0), c, *self.output_res),
                fill_value=x[0, 0, 0, 0] + residual[0, 0, 0, 0],
                dtype=x.dtype,
                device=x.device,
            )
        elif self.mode == "full":
            output = x + residual
            self.output_res = output.shape[2:]
            self.original_outputs[self.cache_id] = output.contiguous()
            self.original_residuals[self.cache_id] = residual.contiguous()
        elif self.mode == "sparse":
            device = x.device.type
            runtime = self.runtime[device]
            assert runtime is not None

            offset = self.main_gather.module.offset
            stride = self.main_gather.module.model_stride
            main_active_indices = self.main_gather.module.active_indices
            shortcut_active_indices = self.shortcut_gather.module.active_indices
            assert main_active_indices is not None
            assert shortcut_active_indices is not None
            if self.backend.lower() not in {"torch", "pytorch"}:
                if main_active_indices.device != x.device:
                    main_active_indices = main_active_indices.to(device=x.device)
                if shortcut_active_indices.device != x.device:
                    shortcut_active_indices = shortcut_active_indices.to(device=x.device)

            output = runtime(
                x.contiguous(),
                self.original_outputs[self.cache_id].contiguous(),
                residual.contiguous(),
                self.original_residuals[self.cache_id].contiguous(),
                offset[0],
                offset[1],
                stride[0],
                stride[1],
                main_active_indices.contiguous(),
                shortcut_active_indices.contiguous(),
            )
            if self.sparse_update:
                if self.scatter_runtime is None:
                    self.scatter_runtime = self.load_runtime_with_backend("scatter", {}, backend=self.backend)
                self.original_outputs[self.cache_id].copy_(output.contiguous())
                self.original_residuals[self.cache_id].copy_(
                    self.scatter_runtime[device](
                        residual.contiguous(),
                        self.original_residuals[self.cache_id].contiguous(),
                        self.shortcut_gather.module.offset[0],
                        self.shortcut_gather.module.offset[1],
                        self.shortcut_gather.module.model_stride[0],
                        self.shortcut_gather.module.model_stride[1],
                        shortcut_active_indices.contiguous(),
                        None,
                    )
                )
        else:
            raise NotImplementedError("Unknown mode: [%s]!!!" % self.mode)
        return output

    def clear_cache(self):
        self.original_outputs = {}
        self.original_residuals = {}
