# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import os
import sys
import warnings
import numpy as np

import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF

from einops import rearrange, repeat
from abc import abstractmethod, ABC

from sige.nn import Gather, Scatter, SIGEConv2d, SIGEModule
from sige.utils import compute_difference_mask, dilate_mask, downsample_mask, reduce_mask

from debugUtil import enable_custom_repr
enable_custom_repr()


repo_root = "/media/cephfs/video/VideoUsers/thu2025/zhurui11/StreamDiffusionV2"


class VAEInterface(ABC, torch.nn.Module):
    @abstractmethod
    def decode_to_pixel(self, latent: torch.Tensor) -> torch.Tensor:
        """
        A method to decode a latent representation to an image or video.
        Input: a tensor with shape [B, F // T, C, H // S, W // S] where T and S are temporal and spatial compression factors.
        Output: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
        """
        pass


__all__ = [
    'WanVAE',
]

CACHE_T = 2


class CausalConv3d(nn.Conv3d):
    """
    Causal 3d convolusion.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 因为 nn.Conv3d.padding 的顺序是 (T, H, W)，
        # 而 F.pad 对 5D Tensor 的 padding 顺序是 (W, H, T)（而且是成对的）
        self._padding = (self.padding[2], self.padding[2],
                         self.padding[1], self.padding[1],
                         2 * self.padding[0], 0)
        # nn.Conv3d内部会自动pad, 要关掉
        self.padding = (0, 0, 0)
        self.time_padding = self._padding[4]
        self.spatial_padding = (self._padding[2], self._padding[0])

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)   # 默认就是零填充

        return super().forward(x)

# Sparse utilities for 3D causal convs (pure PyTorch gather/scatter).
class SIGEModule3d(nn.Module):
    def __init__(self, call_super=True):
        if call_super:
            super().__init__()
        self.mode = "full"
        self.mask = None
        self.timestamp = None
        self.cache_id = 0
        self.sparse_update = False
        self.supported_dtypes = [torch.float32]

    def set_mask(self, masks, cache, timestamp):
        self.timestamp = timestamp

    def set_mode(self, mode):
        self.mode = mode

    def set_cache_id(self, cache_id):
        self.cache_id = cache_id

    def clear_cache(self):
        pass

    def set_sparse_update(self, sparse_update):
        self.sparse_update = sparse_update

    def check_dtype(self, *args):
        for x in args:
            if x is not None:
                assert isinstance(x, torch.Tensor)
                if x.dtype not in self.supported_dtypes:
                    raise NotImplementedError(
                        "[%s] does not support dtype [%s]. Supported: %s"
                        % (self.__class__.__name__, x.dtype, str(self.supported_dtypes))
                    )

    def check_dim(self, *args):
        for x in args:
            if x is not None:
                assert isinstance(x, torch.Tensor)
                if x.dim() != 5:
                    raise NotImplementedError(
                        "[%s] does not support input with dim [%d]!" % (self.__class__.__name__, x.dim())
                    )


class SIGEModel3d(nn.Module):
    def __init__(self):
        super().__init__()
        self.mode = "full"
        self.timestamp = 0

    def set_masks(self, masks):
        self.timestamp += 1
        cache = {}
        for module in self.modules():
            if isinstance(module, (SIGEModule, SIGEModule3d)):
                module.set_mask(masks, cache, self.timestamp)

    def set_mode(self, mode):
        self.mode = mode
        for module in self.modules():
            if isinstance(module, (SIGEModule, SIGEModule3d)):
                module.set_mode(mode)

    def clear_cache(self):
        for module in self.modules():
            if isinstance(module, (SIGEModule, SIGEModule3d)):
                module.clear_cache()

    def set_cache_id(self, cache_id):
        for module in self.modules():
            if isinstance(module, (SIGEModule, SIGEModule3d)):
                module.set_cache_id(cache_id)

    def set_sparse_update(self, sparse_update):
        for module in self.modules():
            if isinstance(module, (SIGEModule, SIGEModule3d)):
                module.set_sparse_update(sparse_update)


class SIGECausalConv3d(CausalConv3d, SIGEModule3d):
    def __init__(self, *args, **kwargs):
        CausalConv3d.__init__(self, *args, **kwargs)
        SIGEModule3d.__init__(self, call_super=False)

    def forward(self, x):
        self.check_dtype(x)
        self.check_dim(x)
        if self.mode == "full":
            return super().forward(x)
        if self.mode in ["sparse", "profile"]:
            return F.conv3d(x, self.weight, self.bias, self.stride, (0, 0, 0), self.dilation, self.groups)
        raise NotImplementedError("Unknown mode: %s" % self.mode)


class Gather3d(SIGEModule3d):
    def __init__(self, conv: CausalConv3d, block_size, offset=None, verbose=False):
        super().__init__()
        if isinstance(block_size, int):
            block_size = (block_size, block_size)

        kernel_size = conv.kernel_size
        stride = conv.stride
        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if not isinstance(stride, tuple):
            stride = (stride, stride, stride)

        n0 = max(block_size[0] - kernel_size[1], 0) // stride[1]
        n1 = max(block_size[1] - kernel_size[2], 0) // stride[2]
        b0 = n0 * stride[1] + kernel_size[1]
        b1 = n1 * stride[2] + kernel_size[2]
        if (b0, b1) != block_size:
            warnings.warn("Change the block size from (%d, %d) to (%d, %d)" % (*block_size, b0, b1))

        self.model_stride = (stride[1], stride[2])
        self.kernel_size = (kernel_size[1], kernel_size[2])
        self.block_size = (b0, b1)
        self.block_stride = ((n0 + 1) * stride[1], (n1 + 1) * stride[2])

        if offset is None:
            self.offset = conv.spatial_padding
        else:
            if isinstance(offset, int):
                offset = (offset, offset)
            self.offset = offset
        self.time_padding = conv.time_padding
        self.verbose = verbose

        self.input_res = None
        self.active_indices = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.check_dtype(x)
        self.check_dim(x)
        b, c, t, _, _ = x.shape
        if self.mode == "profile":
            if self.active_indices is None:
                raise RuntimeError("Active indices are not set for profile mode.")
            output = torch.full(
                (b * self.active_indices.size(0), c, t + self.time_padding, *self.block_size),
                fill_value=x[0, 0, 0, 0, 0],
                dtype=x.dtype,
                device=x.device,
            )
        elif self.mode == "full":
            self.input_res = x.shape[3:]
            output = x
        elif self.mode == "sparse":
            if self.active_indices is None:
                raise RuntimeError("Active indices are not set for sparse mode.")
            pad_h, pad_w = self.offset
            pad_t = self.time_padding
            padded = F.pad(
                x, (pad_w, self.block_size[1], pad_h, self.block_size[0], pad_t, 0)
            )
            num_active = self.active_indices.size(0)
            if num_active == 0:
                return x.new_empty((0, c, t + pad_t, self.block_size[0], self.block_size[1]))
            blocks = []
            for idx in self.active_indices:
                y0 = int(idx[0].item()) + pad_h
                x0 = int(idx[1].item()) + pad_w
                blocks.append(padded[:, :, :, y0:y0 + self.block_size[0], x0:x0 + self.block_size[1]])
            blocks = torch.stack(blocks, dim=1).contiguous()
            output = blocks.view(b * num_active, c, t + pad_t, self.block_size[0], self.block_size[1])
        else:
            raise NotImplementedError("Unknown mode: %s" % self.mode)
        return output

    def set_mask(self, masks, cache, timestamp):
        if self.timestamp != timestamp:
            super().set_mask(masks, cache, timestamp)
            if self.input_res is None:
                raise RuntimeError("Input resolution is not set before set_mask.")
            res = tuple(int(r) for r in self.input_res)
            mask = masks[res]
            self.mask = mask
            key = ("active_indices_3d", *res, *self.block_size, *self.block_stride, *self.offset)
            active_indices = cache.get(key, None)
            if active_indices is None:
                active_indices = reduce_mask(
                    mask, self.block_size, self.block_stride, self.offset, verbose=self.verbose
                )
                cache[key] = active_indices
            self.active_indices = active_indices


class Scatter3d(SIGEModule3d):
    def __init__(self, gather: Gather3d):
        super().__init__()
        self.gather = gather
        self.output_res = None
        self.original_outputs = {}

    def clear_cache(self):
        self.original_outputs = {}

    def forward(self, x: torch.Tensor, residual: torch.Tensor = None) -> torch.Tensor:
        self.check_dtype(x, residual)
        self.check_dim(x, residual)
        if self.mode == "profile":
            output = torch.full(
                (self.original_outputs[self.cache_id].size(0), x.size(1), *self.output_res),
                fill_value=x[0, 0, 0, 0, 0],
                dtype=x.dtype,
                device=x.device,
            )
            if residual is not None:
                output = output + residual
            return output

        if self.mode == "full":
            output = x if residual is None else x + residual
            self.output_res = output.shape[2:]
            self.original_outputs[self.cache_id] = output.contiguous()
            return output

        if self.mode != "sparse":
            raise NotImplementedError("Unknown mode: %s" % self.mode)

        active_indices = self.gather.active_indices
        if active_indices is None:
            raise RuntimeError("Active indices are not set for sparse mode.")
        num_active = active_indices.size(0)
        if num_active == 0:
            return self.original_outputs[self.cache_id].clone()

        offset_h, offset_w = self.gather.offset
        stride_h, stride_w = self.gather.model_stride
        output = self.original_outputs[self.cache_id].clone()

        b, c, t, h, w = output.shape
        out_block_h = x.size(3)
        out_block_w = x.size(4)
        blocks = x.view(b, num_active, c, t, out_block_h, out_block_w)
        for ib in range(num_active):
            bi_h = int((offset_h + active_indices[ib, 0].item()) // stride_h)
            bi_w = int((offset_w + active_indices[ib, 1].item()) // stride_w)
            h_end = min(bi_h + out_block_h, h)
            w_end = min(bi_w + out_block_w, w)
            block = blocks[:, ib, :, :, :h_end - bi_h, :w_end - bi_w]
            if residual is not None:
                block = block + residual[:, :, :, bi_h:h_end, bi_w:w_end]
            output[:, :, :, bi_h:h_end, bi_w:w_end] = block

        if self.sparse_update:
            self.original_outputs[self.cache_id] = output.contiguous()
        return output


class SIGEConv3dBlock(SIGEModule3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, block_size=6):
        super().__init__()
        self.conv = SIGECausalConv3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.gather = Gather3d(self.conv, block_size=block_size)
        self.scatter = Scatter3d(self.gather)

    def forward(self, x):
        x = self.gather(x)
        x = self.conv(x)
        x = self.scatter(x)
        return x


class SIGEResidualBlock3d(SIGEModule3d):
    def __init__(self, in_dim, out_dim, dropout=0.0, main_block_size=6, shortcut_block_size=4):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.norm1 = RMS_norm(in_dim, images=False)
        self.conv1 = SIGECausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = RMS_norm(out_dim, images=False)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = SIGECausalConv3d(out_dim, out_dim, 3, padding=1)

        self.gather1 = Gather3d(self.conv1, block_size=main_block_size)
        self.scatter1 = Scatter3d(self.gather1)
        self.gather2 = Gather3d(self.conv2, block_size=main_block_size)
        self.scatter2 = Scatter3d(self.gather2)

        if in_dim != out_dim:
            self.shortcut = SIGECausalConv3d(in_dim, out_dim, 1)
            self.shortcut_gather = Gather3d(self.shortcut, block_size=shortcut_block_size)
            self.shortcut_scatter = Scatter3d(self.shortcut_gather)
        else:
            self.shortcut = None

    def forward(self, x):
        if self.shortcut is None:
            shortcut = x
        else:
            shortcut = self.shortcut_gather(x)
            shortcut = self.shortcut(shortcut)
            shortcut = self.shortcut_scatter(shortcut)

        h = self.norm1(x)
        h = F.silu(h)
        h = self.gather1(h)
        h = self.conv1(h)
        h = self.scatter1(h)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.gather2(h)
        h = self.conv2(h)
        h = self.scatter2(h, shortcut)
        return h


class SIGEAttentionBlock3d(SIGEModule3d):
    def __init__(self, dim, block_size=4):
        super().__init__()
        self.dim = dim
        self.block_size = block_size

        self.norm = RMS_norm(dim)
        self.to_qkv = SIGEConv2d(dim, dim * 3, 1)
        self.proj = SIGEConv2d(dim, dim, 1)

        self.gather = Gather(self.to_qkv, block_size=block_size)
        self.k_scatter = Scatter(self.gather)
        self.v_scatter = Scatter(self.gather)
        self.out_scatter = Scatter(self.gather)

        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        b, c, t, h, w = x.size()
        x2d = rearrange(x, 'b c t h w -> (b t) c h w')
        x_in = x2d

        x2d = self.gather(x2d)
        x2d = self.norm(x2d)
        q, k, v = self.to_qkv(x2d).chunk(3, dim=1)

        k = self.k_scatter(k)
        v = self.v_scatter(v)

        if self.mode == "full":
            bt, cc, hh, ww = q.shape
            q = q.reshape(bt, cc, hh * ww)
            q = q.permute(0, 2, 1)
        elif self.mode in ("sparse", "profile"):
            bt = b * t
            _, cc, bh, bw = q.shape
            nb = q.shape[0] // bt
            q = q.reshape(bt, nb, cc, bh * bw)
            q = q.permute(0, 1, 3, 2).reshape(bt, -1, cc)
        else:
            raise NotImplementedError("Unknown mode: %s" % self.mode)

        bt, cc, hh, ww = k.shape
        k = k.reshape(bt, cc, hh * ww)
        w_ = torch.bmm(q, k)
        w_ = w_ * (int(cc) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        v = v.reshape(bt, cc, hh * ww)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)

        if self.mode == "full":
            h_ = h_.reshape(bt, cc, hh, ww)
            h_ = self.proj(h_)
            h_ = self.out_scatter(h_, x_in)
        elif self.mode in ("sparse", "profile"):
            bh, bw = self.gather.block_size
            h_ = h_.reshape(bt, cc, -1, bh, bw)
            h_ = h_.permute(0, 2, 1, 3, 4).reshape(-1, cc, bh, bw)
            h_ = self.proj(h_)
            h_ = self.out_scatter(h_, x_in)
        else:
            raise NotImplementedError("Unknown mode: %s" % self.mode)

        h_ = rearrange(h_, '(b t) c h w -> b c t h w', b=b, t=t)
        return h_


class SIGEResample3d(SIGEModule3d):
    def __init__(self, dim, mode, block_size=6):
        assert mode in ('none', 'upsample2d', 'upsample3d', 'downsample2d', 'downsample3d')
        super().__init__()
        self.dim = dim
        self.resample_mode = mode

        if mode == 'upsample2d':
            self.upsample = Upsample(scale_factor=(2., 2.), mode='nearest-exact')
            self.spatial_conv = SIGEConv2d(dim, dim // 2, 3, padding=1)
            self.spatial_gather = Gather(self.spatial_conv, block_size=block_size)
            self.spatial_scatter = Scatter(self.spatial_gather)
        elif mode == 'upsample3d':
            self.upsample = Upsample(scale_factor=(2., 2.), mode='nearest-exact')
            self.spatial_conv = SIGEConv2d(dim, dim // 2, 3, padding=1)
            self.spatial_gather = Gather(self.spatial_conv, block_size=block_size)
            self.spatial_scatter = Scatter(self.spatial_gather)
            self.time_conv = SIGECausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
            self.time_gather = Gather3d(self.time_conv, block_size=block_size)
            self.time_scatter = Scatter3d(self.time_gather)
        elif mode == 'downsample2d':
            self.pad = nn.ZeroPad2d((0, 1, 0, 1))
            self.spatial_conv = SIGEConv2d(dim, dim, 3, stride=2, padding=0)
            self.spatial_gather = Gather(self.spatial_conv, block_size=block_size)
            self.spatial_scatter = Scatter(self.spatial_gather)
        elif mode == 'downsample3d':
            self.pad = nn.ZeroPad2d((0, 1, 0, 1))
            self.spatial_conv = SIGEConv2d(dim, dim, 3, stride=2, padding=0)
            self.spatial_gather = Gather(self.spatial_conv, block_size=block_size)
            self.spatial_scatter = Scatter(self.spatial_gather)
            self.time_conv = SIGECausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
            self.time_gather = Gather3d(self.time_conv, block_size=block_size)
            self.time_scatter = Scatter3d(self.time_gather)
        else:
            self.spatial_conv = None

    def forward(self, x):
        b, c, t, h, w = x.size()
        if self.resample_mode == 'none':
            return x

        if self.resample_mode == 'upsample3d':
            x = self.time_gather(x)
            x = self.time_conv(x)
            x = self.time_scatter(x)
            x = x.reshape(b, 2, c, t, h, w)
            x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), dim=3)
            x = x.reshape(b, c, t * 2, h, w)

        t = x.shape[2]
        x2d = rearrange(x, 'b c t h w -> (b t) c h w')
        if self.resample_mode in ('upsample2d', 'upsample3d'):
            x2d = self.upsample(x2d)
            x2d = self.spatial_gather(x2d)
            x2d = self.spatial_conv(x2d)
            x2d = self.spatial_scatter(x2d)
        else:
            x2d = self.spatial_gather(x2d)
            if self.mode == "full":
                x2d = self.pad(x2d)
            x2d = self.spatial_conv(x2d)
            x2d = self.spatial_scatter(x2d)
        x = rearrange(x2d, '(b t) c h w -> b c t h w', t=t)

        if self.resample_mode == 'downsample3d':
            if x.size(2) < self.time_conv.kernel_size[0]:
                pad_t = self.time_conv.kernel_size[0] - x.size(2)
                x = F.pad(x, (0, 0, 0, 0, pad_t, 0))
            x = self.time_gather(x)
            x = self.time_conv(x)
            x = self.time_scatter(x)

        return x


class SIGEEncoder3d(SIGEModel3d):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0,
        block_size=6,
        attn_block_size=4,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        self.conv1 = SIGEConv3dBlock(3, dims[0], 3, padding=1, block_size=block_size)

        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                downsamples.append(SIGEResidualBlock3d(in_dim, out_dim, dropout, block_size, block_size))
                if scale in attn_scales:
                    downsamples.append(SIGEAttentionBlock3d(out_dim, block_size=attn_block_size))
                in_dim = out_dim

            if i != len(dim_mult) - 1:
                mode = 'downsample3d' if temperal_downsample[i] else 'downsample2d'
                downsamples.append(SIGEResample3d(out_dim, mode=mode, block_size=block_size))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        self.middle = nn.Sequential(
            SIGEResidualBlock3d(out_dim, out_dim, dropout, block_size, block_size),
            SIGEAttentionBlock3d(out_dim, block_size=attn_block_size),
            SIGEResidualBlock3d(out_dim, out_dim, dropout, block_size, block_size),
        )

        self.norm_out = RMS_norm(out_dim, images=False)
        self.conv_out = SIGEConv3dBlock(out_dim, z_dim * 2, 3, padding=1, block_size=block_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.downsamples(x)
        x = self.middle(x)
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        return x


class SIGEDecoder3d(SIGEModel3d):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_upsample=[False, True, True],
        dropout=0.0,
        block_size=6,
        attn_block_size=4,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2**(len(dim_mult) - 2)

        self.conv1 = SIGEConv3dBlock(z_dim, dims[0], 3, padding=1, block_size=block_size)

        self.middle = nn.Sequential(
            SIGEResidualBlock3d(dims[0], dims[0], dropout, block_size, block_size),
            SIGEAttentionBlock3d(dims[0], block_size=attn_block_size),
            SIGEResidualBlock3d(dims[0], dims[0], dropout, block_size, block_size),
        )

        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(SIGEResidualBlock3d(in_dim, out_dim, dropout, block_size, block_size))
                if scale in attn_scales:
                    upsamples.append(SIGEAttentionBlock3d(out_dim, block_size=attn_block_size))
                in_dim = out_dim

            if i != len(dim_mult) - 1:
                mode = 'upsample3d' if temperal_upsample[i] else 'upsample2d'
                upsamples.append(SIGEResample3d(out_dim, mode=mode, block_size=block_size))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        self.norm_out = RMS_norm(out_dim, images=False)
        self.conv_out = SIGEConv3dBlock(out_dim, 3, 3, padding=1, block_size=block_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.middle(x)
        x = self.upsamples(x)
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        return x


class SIGEWanVAE3d(SIGEModel3d):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0,
        block_size=6,
        attn_block_size=4,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = SIGEEncoder3d(
            dim=dim,
            z_dim=z_dim,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            temperal_downsample=temperal_downsample,
            dropout=dropout,
            block_size=block_size,
            attn_block_size=attn_block_size,
        )
        self.conv1 = SIGEConv3dBlock(z_dim * 2, z_dim * 2, 1, block_size=block_size)
        self.conv2 = SIGEConv3dBlock(z_dim, z_dim, 1, block_size=block_size)
        self.decoder = SIGEDecoder3d(
            dim=dim,
            z_dim=z_dim,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            temperal_upsample=temperal_downsample[::-1],
            dropout=dropout,
            block_size=block_size,
            attn_block_size=attn_block_size,
        )

    def encode(self, x):
        h = self.encoder(x)
        moments = self.conv1(h)
        return moments.chunk(2, dim=1)

    def decode(self, z):
        z = self.conv2(z)
        return self.decoder(z)

    def forward(self, x, deterministic=True):
        mu, log_var = self.encode(x)
        if deterministic:
            z = mu
        else:
            std = torch.exp(0.5 * log_var)
            z = mu + std * torch.randn_like(std)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

# 归一化层用RMSNorm
class RMS_norm(nn.Module):
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.

    def forward(self, x):
        return F.normalize(
            x, dim=(1 if self.channel_first else
                    -1)) * self.scale * self.gamma + self.bias


class Upsample(nn.Upsample):

    def forward(self, x):
        """
        Fix bfloat16 support for nearest neighbor interpolation.
        """
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):

    def __init__(self, dim, mode):
        assert mode in ('none', 'upsample2d', 'upsample3d', 'downsample2d',
                        'downsample3d')
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == 'upsample2d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
        elif mode == 'upsample3d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
            self.time_conv = CausalConv3d(
                dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

        elif mode == 'downsample2d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), # 右边、下边补 1，保证 stride=2 时尺寸整齐（常见 trick
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == 'downsample3d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == 'upsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = 'Rep'
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < CACHE_T and feat_cache[
                            idx] is not None and feat_cache[idx] != 'Rep':
                        # cache last frame of last two chunk
                        cache_x = torch.cat([
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                                cache_x.device), cache_x
                        ],
                            dim=2)
                    if cache_x.shape[2] < CACHE_T and feat_cache[
                            idx] is not None and feat_cache[idx] == 'Rep':
                        cache_x = torch.cat([
                            torch.zeros_like(cache_x).to(cache_x.device),
                            cache_x
                        ],
                            dim=2)
                    if feat_cache[idx] == 'Rep':
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
                    
                    # 把时间维度扩展为2倍, 用通道维翻倍（2C）换来时间维翻倍（2T）
                    # 和空间上采样直接用Upsample(scale_factor=(2., 2.)不一样
                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]),dim=3)
                    x = x.reshape(b, c, t * 2, h, w)

        # 统一做空间 2D resample
        t = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.resample(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)

        if self.mode == 'downsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:

                    cache_x = x[:, :, -1:, :, :].clone()
                    # if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None and feat_cache[idx]!='Rep':
                    #     # cache last frame of last two chunk
                    #     cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)

                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x

    def init_weight(self, conv):
        conv_weight = conv.weight
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        one_matrix = torch.eye(c1, c2)
        init_matrix = one_matrix
        nn.init.zeros_(conv_weight)
        # conv_weight.data[:,:,-1,1,1] = init_matrix * 0.5
        conv_weight.data[:, :, 1, 0, 0] = init_matrix  # * 0.5
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def init_weight2(self, conv):
        conv_weight = conv.weight.data
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        init_matrix = torch.eye(c1 // 2, c2)
        # init_matrix = repeat(init_matrix, 'o ... -> (o 2) ...').permute(1,0,2).contiguous().reshape(c1,c2)
        conv_weight[:c1 // 2, :, -1, 0, 0] = init_matrix
        conv_weight[c1 // 2:, :, -1, 0, 0] = init_matrix
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False), nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False), nn.SiLU(), nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1))
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) \
            if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h


class AttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.norm(x)
        # compute query, key, value
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3,
                                         -1).permute(0, 1, 3,
                                                     2).contiguous().chunk(
                                                         3, dim=-1)

        # apply attention
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
        )
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)

        # output
        x = self.proj(x)
        x = rearrange(x, '(b t) c h w-> b c t h w', t=t)
        return x + identity


class Encoder3d(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                mode = 'downsample3d' if temperal_downsample[
                    i] else 'downsample2d'
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout), AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout))

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                    dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        # downsamples
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        # middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        # head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


class Decoder3d(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_upsample=[False, True, True],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2**(len(dim_mult) - 2)

        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout), AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout))

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # upsample block
            if i != len(dim_mult) - 1:
                mode = 'upsample3d' if temperal_upsample[i] else 'upsample2d'
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, 3, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        # conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                    dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        # middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        # upsamples
        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        # head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


def count_conv3d(model):
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count


class WanVAE_(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        # modules
        self.encoder = Encoder3d(dim, z_dim * 2, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_downsample, dropout)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dim, z_dim, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_upsample, dropout)
        self.first_encode = True
        self.first_decode = True

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def encode(self, x, scale):
        self.clear_cache()
        # cache
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        # 对encode输入的x，按时间拆分为1、4、4、4....
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(
                    x[:, :, :1, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx)
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx)
                out = torch.cat([out, out_], 2)

        mu, log_var = self.conv1(out).chunk(2, dim=1)

        if isinstance(scale[0], torch.Tensor):
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                1, self.z_dim, 1, 1, 1)
        else:
            mu = (mu - scale[0]) * scale[1]
        self.clear_cache()
        return mu

    def stream_encode(self, x, scale):
        # self.clear_cache()
        # cache
        t = x.shape[2]
        if self.first_encode:
            self.first_encode = False
            self.clear_cache_encode()
            self._enc_conv_idx = [0]
            out = self.encoder(
                x[:, :, :1, :, :],
                feat_cache=self._enc_feat_map,
                feat_idx=self._enc_conv_idx,
                )
            self._enc_conv_idx = [0]
            out_ = self.encoder(
                x[:, :, 1:, :, :],
                feat_cache=self._enc_feat_map,
                feat_idx=self._enc_conv_idx,
                )
            out = torch.cat([out, out_], 2)
        else:
            out=[]
            for i in range(t//4):
                self._enc_conv_idx = [0]
                out.append(self.encoder(
                    x[:, :, i*4:(i+1)*4, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                    ))
            out = torch.cat(out, 2)
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if scale is not None:
            if isinstance(scale[0], torch.Tensor):
                mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                    1, self.z_dim, 1, 1, 1)
            else:
                mu = (mu - scale[0]) * scale[1]
        # self.clear_cache()
        return mu

    def decode(self, z, scale):
        self.clear_cache()
        # z: [b,c,t,h,w]
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1)
        else:
            z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        x = self.conv2(z)
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx)
            else:
                out_ = self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx)
                out = torch.cat([out, out_], 2)
        self.clear_cache()
        return out

    def stream_decode(self, z, scale):
        # z: [b,c,t,h,w]
        t = z.shape[2]
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1)
        else:
            z = z / scale[1] + scale[0]
        x = self.conv2(z)
        if self.first_decode:
            self.first_decode = False
            self.clear_cache_decode()
            self.first_batch = False
            self._conv_idx = [0]
            out = self.decoder(
                x[:, :, :1, :, :],
                feat_cache=self._feat_map,
                feat_idx=self._conv_idx,
                )
            self._conv_idx = [0]
            out_ = self.decoder(
                x[:, :, 1:, :, :],
                feat_cache=self._feat_map,
                feat_idx=self._conv_idx,
                )
            out = torch.cat([out, out_], 2)
        else:
            out = []
            for i in range(t):
                self._conv_idx = [0]
                out.append(self.decoder(
                    x[:, :, i:(i+1), :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                    ))
            out = torch.cat(out, 2)
        # self.clear_cache()
        return out

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, imgs, deterministic=False):
        mu, log_var = self.encode(imgs)
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        return mu + std * torch.randn_like(std)

    def clear_cache(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        # cache encode
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num

    def clear_cache_decode(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
    
    def clear_cache_encode(self):
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num
    


def _video_vae(pretrained_path=None, z_dim=None, device='cpu', **kwargs):
    """
    Autoencoder3d adapted from Stable Diffusion 1.x, 2.x and XL.
    """
    # params
    cfg = dict(
        dim=96,
        z_dim=z_dim,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0)
    cfg.update(**kwargs)

    # init model
    with torch.device('meta'):
        model = WanVAE_(**cfg)

    # load checkpoint
    logging.info(f'loading {pretrained_path}')

    # 用mmap加速
    model.load_state_dict(
        torch.load(pretrained_path, map_location=device, mmap=True), assign=True)

    return model


class WanVAE:

    def __init__(self,
                 z_dim=16,
                 vae_pth='cache/vae_step_411000.pth',
                 dtype=torch.float,
                 device="cuda"):
        self.dtype = dtype
        self.device = device

        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=dtype, device=device)
        self.std = torch.tensor(std, dtype=dtype, device=device)
        self.scale = [self.mean, 1.0 / self.std]

        # init model
        self.model = _video_vae(
            pretrained_path=vae_pth,
            z_dim=z_dim,
        ).eval().requires_grad_(False).to(device)

    def encode(self, videos):
        """
        videos: A list of videos each with shape [C, T, H, W].
        """
        with amp.autocast(dtype=self.dtype):
            return [
                self.model.encode(u.unsqueeze(0), self.scale).float().squeeze(0)
                for u in videos
            ]

    def decode(self, zs):
        with amp.autocast(dtype=self.dtype):
            return [
                self.model.decode(u.unsqueeze(0),
                                  self.scale).float().clamp_(-1, 1).squeeze(0)
                for u in zs
            ]

class WanVAEWrapper(VAEInterface):
    def __init__(self, model_type="T2V-1.3B"):
        super().__init__()
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        # init model
        self.model = _video_vae(
            pretrained_path=os.path.join(repo_root, "wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"),
            z_dim=16,
        ).eval().requires_grad_(False)

    def decode_to_pixel(self, latent: torch.Tensor) -> torch.Tensor:
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        output = [
            self.model.decode(u.unsqueeze(0),
                              scale).float().clamp_(-1, 1).squeeze(0)
            for u in zs
        ]
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        output = self.model.decode(zs, scale).clamp_(-1, 1)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        # output = output.permute(0, 2, 1, 3, 4)
        return output
    
    # 调用入口
    def stream_encode(self, video: torch.Tensor, is_scale=True) -> torch.Tensor:
        if is_scale:
            device, dtype = video.device, video.dtype
            scale = [self.mean.to(device=device, dtype=dtype),
                    1.0 / self.std.to(device=device, dtype=dtype)]
        else:
            scale = None
        return self.model.stream_encode(video, scale)
    
    def stream_decode_to_pixel(self, latent: torch.Tensor) -> torch.Tensor:
        zs = latent.permute(0, 2, 1, 3, 4)
        zs = zs.to(torch.bfloat16).to('cuda')
        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]
        output = self.model.stream_decode(zs, scale).float().clamp_(-1, 1)
        output = output.permute(0, 2, 1, 3, 4)
        return output

def load_mp4_as_tensor(
    video_path: str,
    max_frames: int = None,
    resize_hw: tuple[int, int] = None,
    normalize: bool = True,
) -> tuple[torch.Tensor, int]:
    assert os.path.exists(video_path), f"Video file not found: {video_path}"

    # 捕获第三个返回值 info，其中包含元数据
    video, _, info = torchvision.io.read_video(video_path, output_format="TCHW", pts_unit="sec")
    
    # 从元数据中获取视频的FPS，如果获取不到则提供一个默认值
    original_fps = info.get('video_fps', 16)

    if max_frames is not None:
        video = video[:max_frames]
    video = rearrange(video, "t c h w -> c t h w")
    if resize_hw is not None:
        c, t, h0, w0 = video.shape
        video = torch.stack([TF.resize(video[:, i], resize_hw, antialias=True) for i in range(t)], dim=1)
    if video.dtype != torch.float32:
        video = video.float()
    if normalize:
        video = video / 127.5 - 1.0
        
    return video, original_fps # 返回视频张量和原始FPS


def iter_5_4_4_chunks(x: torch.Tensor):
    # x: [B, C, T, H, W]
    T = x.shape[2]
    s = 0
    first = True
    while s < T:
        step = 5 if first else 4
        first = False
        e = min(s + step, T)
        yield x[:, :, s:e]  # [B,C,chunkT,H,W]
        s = e

def psnr_np(x, y, max_val=255.0):
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    mse = np.mean((x - y) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10((max_val ** 2) / mse)


def load_img_sd_style(path: str, size: tuple[int, int] | None = None) -> torch.Tensor:
    from PIL import Image

    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    if size is None:
        w, h = w - w % 32, h - h % 32
    else:
        w, h = size
        w, h = w - w % 32, h - h % 32
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def video_tensor_to_numpy_uint8(video: torch.Tensor) -> np.ndarray:
    video = (video * 0.5 + 0.5).clamp(0, 1)
    video = video[0].permute(1, 2, 3, 0).contiguous()
    video = (video * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    return video


def save_video_uint8(video: np.ndarray, out_path: str, fps: int = 16):
    torchvision.io.write_video(out_path, video, fps=fps)


def copy_sige_weights_from_base(base_model: nn.Module, sige_model: nn.Module):
    module_types = (nn.Conv2d, nn.Conv3d, RMS_norm)
    base_modules = [m for m in base_model.modules() if isinstance(m, module_types)]
    sige_modules = [m for m in sige_model.modules() if isinstance(m, module_types)]
    if len(base_modules) != len(sige_modules):
        raise RuntimeError(
            f"Module count mismatch: base={len(base_modules)} sige={len(sige_modules)}"
        )
    for src, dst in zip(base_modules, sige_modules):
        if any(a.shape != b.shape for a, b in zip(src.state_dict().values(), dst.state_dict().values())):
            raise RuntimeError("Parameter shape mismatch during weight transfer.")
        dst.load_state_dict(src.state_dict(), strict=True)


def get_sige_vae_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_img", type=str, required=True, help="path to the original image")
    parser.add_argument("--edited_img", type=str, required=True, help="path to the edited image")
    parser.add_argument("--out", type=str, default="sige_vae_recon.mp4")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"])
    parser.add_argument("--vae_pth", type=str, default=None)
    parser.add_argument("--z_dim", type=int, default=16)
    parser.add_argument("--size", type=int, default=None, help="square resize (multiple of 32)")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    return parser.parse_args()


@torch.no_grad()
def run_sige_vae3d_test():
    from torchprofile import profile_macs

    args = get_sige_vae_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    torch.manual_seed(0)

    if args.size is not None:
        target_hw = (args.size, args.size)
    elif args.height is not None and args.width is not None:
        target_hw = (args.width, args.height)
    else:
        target_hw = None

    edited_img = load_img_sd_style(args.edited_img, size=target_hw)
    if target_hw is None:
        target_hw = (edited_img.shape[-1], edited_img.shape[-2])
    init_img = load_img_sd_style(args.init_img, size=target_hw)

    edited_img = edited_img.to(device=device, dtype=torch.float32)
    init_img = init_img.to(device=device, dtype=torch.float32)

    edited_img = repeat(edited_img, "1 ... -> b ...", b=1)
    init_img = repeat(init_img, "1 ... -> b ...", b=1)

    difference_mask = compute_difference_mask(init_img, edited_img)
    print("Edit Ratio: %.2f%%" % (difference_mask.sum() / difference_mask.numel() * 100))
    difference_mask = dilate_mask(difference_mask, 5)
    masks = downsample_mask(difference_mask, min_res=(4, 4), dilation=1)

    b, c, t = 1, 3, args.frames
    edited_input = edited_img.unsqueeze(2).repeat(1, 1, t, 1, 1)
    original_input = init_img.unsqueeze(2).repeat(1, 1, t, 1, 1)

    cfg = dict(
        dim=96,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
    )

    if args.vae_pth is None:
        args.vae_pth = os.path.join(
            repo_root, "wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
        )
    if not os.path.exists(args.vae_pth):
        raise FileNotFoundError(f"VAE checkpoint not found: {args.vae_pth}")

    base_model = _video_vae(pretrained_path=args.vae_pth, z_dim=args.z_dim, device=device, **cfg)
    base_model.eval()

    model = SIGEWanVAE3d(z_dim=args.z_dim, **cfg).to(device)
    copy_sige_weights_from_base(base_model, model)
    model.eval()
    del base_model

    class ReconWrapper(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return self.inner(x, deterministic=True)[0]

    wrapper = ReconWrapper(model)

    # Full pass on original input to populate caches (same as sdedit_runner flow).
    model.set_mode("full")
    full_macs = profile_macs(wrapper, (original_input,))
    model.clear_cache()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model.set_mode("full")
    full_output = wrapper(original_input)

    # Sparse pass on edited input using the difference mask.
    model.set_mode("sparse")
    model.set_masks(masks)
    sparse_output = wrapper(edited_input)

    model.set_mode("profile")
    sparse_macs = profile_macs(wrapper, (edited_input,))

    print("Full MACs: %.2fG" % (full_macs / 1e9))
    print("Sparse MACs: %.2fG" % (sparse_macs / 1e9))

    full_video = video_tensor_to_numpy_uint8(full_output)
    sparse_video = video_tensor_to_numpy_uint8(sparse_output)

    save_video_uint8(sparse_video, args.out, fps=args.fps)
    base_out, ext = os.path.splitext(args.out)
    save_video_uint8(full_video, f"{base_out}_full{ext}", fps=args.fps)
    print("saved sparse to", args.out)
    print("saved full to", f"{base_out}_full{ext}")


@torch.no_grad()
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    vae = WanVAEWrapper().to(device=device, dtype=dtype)

    video_path = "original.mp4"
    out_path = "recon.mp4"

    input_video_original, original_fps = load_mp4_as_tensor(video_path, resize_hw=(480, 832))
    input_video_original = input_video_original.unsqueeze(0).to(device=device, dtype=dtype)

    recon_video_list = []

    # ---- 5,4,4,... 分块 encode ----
    for i, chunk in enumerate(iter_5_4_4_chunks(input_video_original)):
        lat = vae.stream_encode(chunk)          # 逐块喂进去
        lat = lat.permute(0, 2, 1, 3, 4)
        video = vae.stream_decode_to_pixel(lat)
        recon_video_list.append(video)
        print(f"[Chunk {i}] done!!!")


    video = torch.cat(recon_video_list, dim=1)

    video = (video * 0.5 + 0.5).clamp(0, 1)
    video = video[0].permute(0, 2, 3, 1).contiguous()
    video = (video * 255).round().clamp(0,255).to(torch.uint8).cpu().numpy()

    torchvision.io.write_video(out_path, video, fps=int(original_fps))
    print("saved to", out_path)

    # cal PSNR
    input_video_original = (input_video_original * 0.5 + 0.5).clamp(0, 1)
    input_video_original = input_video_original[0].permute(1, 2, 3, 0).contiguous()
    input_video_original = (input_video_original * 255).round().clamp(0,255).to(torch.uint8).cpu().numpy()

    psnr_video = psnr_np(input_video_original, video)
    print(f"[PSNR] Full video: {psnr_video:.2f} dB")



if __name__ == "__main__":
    if "--sige-vae-test" in sys.argv:
        sys.argv.remove("--sige-vae-test")
        run_sige_vae3d_test()
    else:
        main()
