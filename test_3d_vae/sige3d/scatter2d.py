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


    def forward(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.check_dtype(x, residual)
        self.check_dim(x, residual)
        if self.mode == "profile":
            _, c, _, _ = x.shape
            output = torch.full(
                (self.original_outputs.size(0), c, *self.output_res),
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
            self.original_outputs = output.contiguous()
        
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

            # torch.cuda.synchronize()
            # start = torch.cuda.Event(enable_timing=True)
            # end   = torch.cuda.Event(enable_timing=True)
            # start.record()

            output = runtime(
                x.contiguous(),
                self.original_outputs.contiguous(),
                offset[0],
                offset[1],
                stride[0],
                stride[1],
                active_indices.contiguous(),
                None if residual is None else residual.contiguous(),
            )

            # end.record()
            # torch.cuda.synchronize()
            # print(f"scatter2d time1111111: {start.elapsed_time(end):.2f} ms")   # ms
            
        else:
            raise NotImplementedError("Unknown mode: [%s]!!!" % self.mode)
        return output
