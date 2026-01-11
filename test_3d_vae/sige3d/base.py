from __future__ import annotations
import importlib
import os

import inspect
from typing import Dict, List, Optional

import torch
from torch import nn

import torch
import torch.nn as nn
import torch.nn.functional as F


class SIGEModule3d(nn.Module):
    def __init__(self, call_super: bool = True):
        if call_super:
            super().__init__()
        # self.devices: List[str] = ["cpu", "cuda", "mps"]
        self.devices: List[str] = ["cuda"]
        self.supported_dtypes = [torch.float32, torch.float16, torch.bfloat16]
        self.mode: str = "full"
        self.runtime: Dict = {}
        self.mask: Optional[torch.Tensor] = None
        self.timestamp: Optional[int] = None
        self.cache_id: int = 0
        self.sparse_update: bool = False

    def set_mask(self, masks: Dict, cache: Dict, timestamp: int):
        self.timestamp = timestamp

    def set_mode(self, mode: str):
        self.mode = mode

    def set_cache_id(self, cache_id: int):
        self.cache_id = cache_id

    def clear_cache(self):
        pass

    def clear_stream_cache(self):
        pass

    def load_runtime_with_backend(self, function_name: str, runtime_dict: Dict = None, backend: str = "ext"):
        """
        backend:
          - "torch": use pure PyTorch reference kernels from `sige.nn.torch_kernels`
          - "ext"/"cuda": use compiled extension modules (`sige.cpu` / `sige.cuda` / `sige.mps`)
          - "auto": prefer extensions, fall back to torch when missing
        """
        if runtime_dict is None:
            runtime_dict = self.runtime
        backend = (backend or "ext").lower()

        if backend in {"torch", "pytorch"}:
            torch_kernels = importlib.import_module("sige3d.torch_kernels")
            try:
                runtime = getattr(torch_kernels, function_name)
            except AttributeError as e:
                raise AttributeError(f"torch_kernels has no function [{function_name}]") from e
            for device in self.devices:
                runtime_dict[device] = runtime
            return runtime_dict


        # "ext" / "cuda" / "native": prefer compiled extension modules (sige.cpu / sige.cuda / sige.mps)
        for device in self.devices:
            name = "sige.%s" % device
            try:
                module = importlib.import_module(name)
                runtime = getattr(module, function_name)
                runtime_dict[device] = runtime
                if device == "mps":
                    os.environ["SIGE_METAL_LIB_PATH"] = os.path.abspath(
                        os.path.join(os.path.dirname(module.__file__), "..", "sige.metallib")
                    )
            except (ModuleNotFoundError, AttributeError):
                runtime_dict[device] = torch_runtime if torch_runtime is not None else None
        return runtime_dict


    def set_sparse_update(self, sparse_update: bool):
        self.sparse_update = sparse_update
            

    def check_dtype(self, *args: Optional[torch.Tensor]):
        for x in args:
            if x is None:
                continue
            if x.dtype not in self.supported_dtypes:
                raise NotImplementedError(
                    f"[{self.__class__.__name__}] does not support dtype [{x.dtype}]. "
                    f"Supported: {self.supported_dtypes}"
                )

    def check_dim(self, *args: Optional[torch.Tensor]):
        for x in args:
            if x is None:
                continue
            if x.dim() != 5 and x.dim() != 4:
                raise NotImplementedError(
                    f"[{self.__class__.__name__}] does not support input with dim [{x.dim()}]."
                )


class SIGEModuleWrapper:
    def __init__(self, module: SIGEModule3d):
        self.module = module


class SIGEModel3d(nn.Module):
    def __init__(self, call_super: bool = True):
        if call_super:
            super().__init__()
        self.mode: str = "full"
        self.timestamp: int = 0

    def set_masks(self, masks: Dict):
        self.timestamp += 1
        cache: Dict = {}
        for module in self.modules():
            if isinstance(module, SIGEModule3d):
                module.set_mask(masks, cache, self.timestamp)
           
    def set_mode(self, mode: str):
        self.mode = mode
        for module in self.modules():
            if isinstance(module, SIGEModule3d):
                module.set_mode(mode)

    def set_sparse_update(self, sparse_update: bool):
        for module in self.modules():
            if isinstance(module, SIGEModule3d):
                module.set_sparse_update(sparse_update)

    def flow_cache(self, flow):
        for module in self.modules():
            # if isinstance(module, SIGEModule3d):
            if isinstance(module, SIGEModule3d) and hasattr(module, "flow_cache"):
                module.flow_cache(flow)


class SIGECausalConv3d(nn.Conv3d, SIGEModule3d):
    """
    full: pad(T causal) + pad(HW) -> conv3d(padding=0)
    sparse/profile: pad(T causal only) -> conv3d(padding=0)
    """

    def __init__(self, *args, **kwargs):
        nn.Conv3d.__init__(self, *args, **kwargs)
        SIGEModule3d.__init__(self, call_super=False)

        # 原始 conv3d 的 padding (T,H,W)
        p_t, p_h, p_w = self.padding
        
        # 供 Gather3d 推断 spatial offset（等价于原始 padding(H,W)）
        self.spatial_padding = (int(p_h), int(p_w))

        # spatial pad for F.pad order: (Wl, Wr, Hl, Hr, Tl, Tr)
        # mode = full的时候才需要
        self._spatial_pad = (p_w, p_w, p_h, p_h, 0, 0)

        # causal temporal pad: only pad "past" (left) side
        self._temporal_pad = (0, 0, 0, 0, 2 * p_t, 0)

        # 关闭 conv3d 内部 padding（我们统一用 F.pad）
        self.padding = (0, 0, 0)

    def _apply_temporal_pad(self, x: torch.Tensor, cache_x: torch.Tensor = None) -> torch.Tensor:
        # temporal pad is (Wl,Wr,Hl,Hr,Tl,Tr) but only Tl/Tr are non-zero
        pad = list(self._temporal_pad)

        if cache_x is not None and pad[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)  # dim=2 is T
            # cache 覆盖掉一部分“过去pad需求”
            pad[4] -= cache_x.shape[2]

        return F.pad(x, pad)

    def forward(self, x: torch.Tensor, cache_x: torch.Tensor = None) -> torch.Tensor:
        if self.mode == "full":
            # full：整图需要 HW padding + 因果 T padding
            x = self._apply_temporal_pad(x, cache_x)
            x = F.pad(x, self._spatial_pad)

            return super(SIGECausalConv3d, self).forward(x)  # self.padding=0

        elif self.mode in ["sparse", "profile"]:
            # sparse：假设 gather 已经提供了 HW halo，所以只做 T 因果 padding
            x = self._apply_temporal_pad(x, cache_x)

            return F.conv3d(    # pylint: disable=not-callable
                x, self.weight, self.bias,
                self.stride, (0, 0, 0),
                self.dilation, self.groups
            )

        else:
            raise NotImplementedError(f"Unknown mode: {self.mode}")


class SIGEConv2d(nn.Conv2d, SIGEModule3d):
    def __init__(self, *args, **kwargs):
        nn.Conv2d.__init__(self, *args, **kwargs)
        SIGEModule3d.__init__(self, call_super=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "full":
            output = super(SIGEConv2d, self).forward(x)
        elif self.mode in ["sparse", "profile"]:
            output = F.conv2d(x, self.weight, self.bias, self.stride, (0, 0), self.dilation, self.groups) # pylint: disable=not-callable
        else:
            raise NotImplementedError("Unknown mode: %s" % self.mode)
        return output