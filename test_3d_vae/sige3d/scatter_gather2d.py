from typing import Dict, Optional

import torch
from utils.flow_cache_utils import forward_warp_cache_4d

from .base import SIGEModule3d, SIGEModuleWrapper
from .gather2d import Gather2d
from .activation import activation


class ScatterGather2d(SIGEModule3d):
    def __init__(
        self,
        gather: Gather2d,
        activation_name: str = "identity",
        activation_first: bool = False,
        backend: Optional[str] = None,
    ):
        """
        backend:
          - None: follow `gather.backend`
          - "torch": use `sige.nn.torch_kernels`
          - "ext"/"cuda": use compiled extension (`sige.cpu` / `sige.cuda` / `sige.mps`)
        """
        super(ScatterGather2d, self).__init__()
        self.gather = SIGEModuleWrapper(gather)
        self.activation_name = activation_name
        self.activation_first = activation_first

        self.backend = backend if backend is not None else getattr(gather, "backend", "torch")
        self.load_runtime_with_backend("scatter_gather2d", backend=self.backend)
        self.scatter_runtime = self.load_runtime_with_backend("scatter", {}, backend=self.backend)
        self.get_scatter_map_runtime = self.load_runtime_with_backend("get_scatter_map", {}, backend=self.backend)

        self.scatter_map = None
        self.output_res = None
        self.original_outputs = None
    
    def flow_cache(self, flow):
        """
        将 original_outputs 中的缓存特征按 flow 做反向采样（backward warp）+ 双线性插值
        - 输入缓存: [B, C, T, H, W]
        - flow: [H, W, 2]，每个位置是 (dx, dy)（像素位移，允许非整数）
        - 输出：同形状，output[..., y, x] 从 input[..., y+dy, x+dx] 采样（越界按 0 填充）
        - 对每个时间帧 T 做同样的空间采样（flow 仅依赖 H,W）
        """

        self.original_outputs = forward_warp_cache_4d(self.original_outputs, flow).contiguous()



    def forward(
        self, x: torch.Tensor, scale: Optional[torch.Tensor] = None, shift: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        self.check_dtype(x, scale, shift)
        self.check_dim(x, scale, shift)
        b, c, h, w = x.shape
        active_indices = self.gather.module.active_indices
        block_size = self.gather.module.block_size
        if self.mode == "profile":
            output = torch.full(
                (self.original_outputs.size(0) * active_indices.size(0), c, *block_size),
                fill_value=x[0, 0, 0, 0],
                dtype=x.dtype,
                device=x.device,
            )  # create a dummy gather output depending on the input for profiling
            if scale is not None:
                output = output * scale[0, 0, 0, 0]
            if shift is not None:
                output = output + shift[0, 0, 0, 0]
            output = activation(output, self.activation_name)
        elif self.mode == "full":
            output = x
            self.output_res = output.shape[2:]
            self.original_outputs = output.contiguous()
        
        elif self.mode == "sparse":
            device = x.device.type
            runtime = self.runtime[device]
            assert runtime is not None
            active_indices = self.gather.module.active_indices
            assert active_indices is not None
            scatter_map = self.scatter_map
            assert scatter_map is not None
            if self.backend.lower() not in {"torch", "pytorch"}:
                if active_indices.device != x.device:
                    active_indices = active_indices.to(device=x.device)
                if scatter_map.device != x.device:
                    scatter_map = scatter_map.to(device=x.device)
            output = runtime(
                x.contiguous(),
                self.original_outputs.contiguous(),
                block_size[0],
                block_size[1],
                active_indices.contiguous(),
                scatter_map.contiguous(),
                None if scale is None else scale.contiguous(),
                None if shift is None else shift.contiguous(),
                self.activation_name,
                self.activation_first,
            )
            if self.sparse_update:
                self.original_outputs.copy_(
                    self.scatter_runtime[device](
                        x.contiguous(),
                        self.original_outputs.contiguous(),
                        self.gather.module.offset[0],
                        self.gather.module.offset[1],
                        self.gather.module.model_stride[0],
                        self.gather.module.model_stride[1],
                        active_indices.contiguous(),
                        None,
                    )
                )
        else:
            raise NotImplementedError("Unknown mode: [%s]!!!" % self.mode)
        return output

    def clear_cache(self):
        self.original_outputs = {}

    def set_mask(self, masks: Dict, cache: Dict, timestamp: int):
        if self.timestamp != timestamp:
            super(ScatterGather2d, self).set_mask(masks, cache, timestamp)
            self.gather.module.set_mask(masks, cache, timestamp)

            mask = self.gather.module.mask
            h, w = mask.shape
            block_size = self.gather.module.block_size
            kernel_size = self.gather.module.kernel_size
            offset = self.gather.module.offset
            stride = self.gather.module.model_stride

            key = ("scatter_map", h, w, *block_size, *kernel_size, *offset, *stride)
            scatter_map = cache.get(key, None)
            if scatter_map is None:
                active_indices = self.gather.module.active_indices
                assert active_indices is not None
                if self.backend.lower() in {"torch", "pytorch"} and active_indices.device.type != "cpu":
                    active_indices = active_indices.detach().cpu()
                device = active_indices.device.type
                runtime = self.get_scatter_map_runtime[device]
                scatter_map = runtime(
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
                    active_indices,
                )
                cache[key] = scatter_map
            self.scatter_map = scatter_map
