from __future__ import annotations

import inspect
from typing import Dict, List, Optional

import torch
from torch import nn

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from sige.nn import SIGEModule as SIGEModule2d
except Exception:
    SIGEModule2d = None


class SIGEModule3d(nn.Module):
    def __init__(self, call_super: bool = True):
        if call_super:
            super().__init__()
        self.supported_dtypes = [torch.float32, torch.float16, torch.bfloat16]
        self.mode: str = "full"
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
            if x.dim() != 5:
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
            if SIGEModule2d is not None and isinstance(module, SIGEModule2d):
                module.set_mask(masks, cache, self.timestamp)

    def set_mode(self, mode: str):
        self.mode = mode
        for module in self.modules():
            if isinstance(module, SIGEModule3d):
                module.set_mode(mode)
            if SIGEModule2d is not None and isinstance(module, SIGEModule2d):
                module.set_mode(mode)
    

    def clear_cache(self):
        for module in self.modules():
            if isinstance(module, SIGEModule3d):
                module.clear_cache()
            if SIGEModule2d is not None and isinstance(module, SIGEModule2d):
                module.clear_cache()

    def clear_stream_cache(self):
        for module in self.modules():
            if isinstance(module, SIGEModule3d):
                module.clear_stream_cache()

    def set_cache_id(self, cache_id: int):
        for module in self.modules():
            if isinstance(module, SIGEModule3d):
                module.set_cache_id(cache_id)
            if SIGEModule2d is not None and isinstance(module, SIGEModule2d):
                module.set_cache_id(cache_id)

    def set_sparse_update(self, sparse_update: bool):
        for module in self.modules():
            if isinstance(module, SIGEModule3d):
                module.set_sparse_update(sparse_update)
            if SIGEModule2d is not None and isinstance(module, SIGEModule2d):
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

            return F.conv3d(
                x, self.weight, self.bias,
                self.stride, (0, 0, 0),
                self.dilation, self.groups
            )

        else:
            raise NotImplementedError(f"Unknown mode: {self.mode}")
