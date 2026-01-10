# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
'''
实现的功能:
每次下一帧(下一个chunk)处理之前, 把scatter中的cache即original_outputs,
按照光流映射到下一帧的位置,新露出的区域即mask,通过调用set_masks函数进行设置

sparse_update设置为True
这样新生成的mask区域会写回original_outputs,方便下一帧用
'''

import argparse
import logging
import os
import numpy as np

import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF

from einops import rearrange
from abc import abstractmethod, ABC
import re
from collections import OrderedDict
from PIL import Image

from torchprofile import profile_macs

# Optional profiling helpers (torch >= 1.8, recommended torch >= 2.0)
try:
    from torch.profiler import profile as torch_profile, ProfilerActivity
except Exception:
    torch_profile = None
    ProfilerActivity = None

# TODO: 有2D的gather，scatter需要处理, flow_cache
from sige.nn import Gather, Scatter, SIGEConv2d
from sige3d import SIGECausalConv3d, Gather3d, Scatter3d, ScatterWithBlockResidual3d, ScatterGather3d, SIGEModel3d, SIGEModule3d
from sige.utils import dilate_mask, downsample_mask


from debugUtil import enable_custom_repr
enable_custom_repr()


repo_root = "/media/cephfs/video/VideoUsers/thu2025/zhurui11/StreamDiffusionV2"
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(THIS_DIR, "assets")


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

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)   # 默认就是零填充

        return super().forward(x)

# TODO: 看需不需要继承SIGEModule
# 归一化层用RMSNorm
class RMS_norm(nn.Module):
    def __init__(self, dim, channel_first=True, images=True, bias=False, eps=1e-6):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.eps = eps

        # 这些 nn.Parameter 在推理（inference）时，正常情况下就是从训练好的权重里加载进来的
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.

    def forward(self, x, return_inv_norm: bool = False):
            return F.normalize(
                x, dim=(1 if self.channel_first else
                        -1)) * self.scale * self.gamma + self.bias

        # F.normalize等价的两步写法:
        # norm = torch.norm(x, p=2, dim=1, keepdim=True)  # 计算L2范数
        # normalized_x = x / norm

        # dim = 1 if self.channel_first else -1
        # norm = x.norm(2, dim=dim, keepdim=True) + self.eps
        # inv_norm = 1.0 / norm
        # y = x * inv_norm * self.scale * self.gamma + self.bias
        # if return_inv_norm:
        #     return y, inv_norm
        # return y


class Upsample(nn.Upsample):

    def forward(self, x):
        """
        Fix bfloat16 support for nearest neighbor interpolation.
        """
        return super().forward(x.float()).type_as(x)


class Resample(SIGEModule3d):
    def __init__(self, dim, resample_mode, block_size=6):
        assert resample_mode in ('none', 'upsample2d', 'upsample3d', 'downsample2d',
                        'downsample3d')
        super().__init__()
        self.dim = dim
        self.resample_mode = resample_mode

        # layers
        if resample_mode == 'upsample2d':
            # self.resample = nn.Sequential(
            #     Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
            #     nn.Conv2d(dim, dim // 2, 3, padding=1))
            
            self.spatupsample = Upsample(scale_factor=(2., 2.), mode='nearest-exact')
            self.conv = SIGEConv2d(dim, dim // 2, 3, padding=1)
            # self.conv = nn.Conv2d(dim, dim // 2, 3, padding=1)

            self.gather2d = Gather(self.conv, block_size=block_size)
            self.scatter2d = Scatter(self.gather2d)

        elif resample_mode == 'upsample3d':
            # self.resample = nn.Sequential(
            #     Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
            #     nn.Conv2d(dim, dim // 2, 3, padding=1))

            self.spatupsample = Upsample(scale_factor=(2., 2.), mode='nearest-exact')
            self.conv = SIGEConv2d(dim, dim // 2, 3, padding=1)
            # self.conv = nn.Conv2d(dim, dim // 2, 3, padding=1)

            self.gather2d = Gather(self.conv, block_size=block_size)
            self.scatter2d = Scatter(self.gather2d)

            # TODO: 和ResnetBlck一样，也是两次卷积操作，看看能不能用scatter_gather来合并中间的一次scatter和gather
            # 先用普通的两次代替
            self.time_conv = SIGECausalConv3d(
                dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
            self.gather3d = Gather3d(self.time_conv, block_size=block_size)
            self.scatter3d = Scatter3d(self.gather3d)


        elif resample_mode == 'downsample2d':
            # self.resample = nn.Sequential(
            #     nn.ZeroPad2d((0, 1, 0, 1)), # 右边、下边补 1，保证 stride=2 时尺寸整齐（常见 trick
            #     nn.Conv2d(dim, dim, 3, stride=(2, 2)))

            self.downpad = nn.ZeroPad2d((0, 1, 0, 1))
            self.conv = SIGEConv2d(dim, dim, 3, stride=(2, 2))
            # self.conv = nn.Conv2d(dim, dim, 3, stride=(2, 2))
            self.gather2d = Gather(self.conv, block_size=block_size)
            self.scatter2d = Scatter(self.gather2d)
            
        elif resample_mode == 'downsample3d':
            # self.resample = nn.Sequential(
            #     nn.ZeroPad2d((0, 1, 0, 1)),
            #     nn.Conv2d(dim, dim, 3, stride=(2, 2)))

            self.downpad = nn.ZeroPad2d((0, 1, 0, 1))
            self.conv = SIGEConv2d(dim, dim, 3, stride=(2, 2))
            self.conv = nn.Conv2d(dim, dim, 3, stride=(2, 2))
            self.gather2d = Gather(self.conv, block_size=block_size)
            self.scatter2d = Scatter(self.gather2d)

            self.time_conv = SIGECausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
            self.gather3d = Gather3d(self.time_conv, block_size=block_size)
            self.scatter3d = Scatter3d(self.gather3d)


    def first_forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.resample_mode == 'upsample3d':
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

        if self.resample_mode == 'upsample3d' or self.resample_mode == 'upsample2d':
            x = self.spatupsample(x)
            x = self.conv(x)    # 2D卷积
            
        if self.resample_mode == 'downsample2d' or self.resample_mode == 'downsample3d':
            # TODO: 稀疏计算的时候是不是不需要padding
            # DONE: 不需要
            x = self.downpad(x)
            x = self.conv(x)    # 2D卷积，通过stride=2下采样

        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)


        if self.resample_mode == 'downsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:

                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x



    def sparse_forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.resample_mode == 'upsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = 'Rep'
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()

                    x = self.gather3d(x)

                    if cache_x.shape[2] < CACHE_T and feat_cache[
                            idx] is not None and feat_cache[idx] != 'Rep':
                        # cache last frame of last two chunk
                        cache_x = torch.cat([
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                                cache_x.device), cache_x
                        ],
                            dim=2)
            
                    if feat_cache[idx] == 'Rep':
                        x = self.time_conv(x)
                    else:
                        cache = feat_cache[idx]
                        cache = self.gather3d(cache)
                        x = self.time_conv(x, cache)
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
                  
                    # 把时间维度扩展为2倍, 用通道维翻倍（2C）换来时间维翻倍（2T）
                    N, C2, T, bh, bw = x.shape
                    assert C2 % 2 == 0
                    C = C2 // 2

                    # 拆成 2 份通道
                    x = x.view(N, 2, C, T, bh, bw)          # [N, 2, C, T, bh, bw]

                    # 把“2”插到时间维，变成 2T
                    x = x.permute(0, 2, 3, 1, 4, 5)         # [N, C, T, 2, bh, bw]
                    x = x.reshape(N, C, T * 2, bh, bw)      # [N, C, 2T, bh, bw]
                    
                    x = self.scatter3d(x)
            # else:
            #     x = self.gather3d(x)
            #     x = self.time_conv(x)

            #     N, C2, T, bh, bw = x.shape
            #     assert C2 % 2 == 0
            #     C = C2 // 2
            #     x = x.view(N, 2, C, T, bh, bw)
            #     x = x.permute(0, 2, 3, 1, 4, 5)
            #     x = x.reshape(N, C, T * 2, bh, bw)
            #     x = self.scatter3d(x)
                    

        # 统一做空间 2D resample
        t = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')

        if self.resample_mode == 'upsample3d' or self.resample_mode == 'upsample2d':
            x = self.spatupsample(x)
            x = self.gather2d(x)
            x = self.conv(x)    # 2D卷积
            x = self.scatter2d(x)
            
        if self.resample_mode == 'downsample2d' or self.resample_mode == 'downsample3d':
            # TODO: 稀疏计算的时候是不是不需要padding
            # DONE: 不需要
            x = self.gather2d(x)
            if self.mode == "full":
                x = self.downpad(x)
            x = self.conv(x)    # 2D卷积，通过stride=2下采样
            x = self.scatter2d(x)
        
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)


        if self.resample_mode == 'downsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    # 先因果缓存，再gather，因为这个因果缓存要缓存整张图，而不是稀疏的部分
                    cache_x = x[:, :, -1:, :, :].clone()

                    x = self.gather3d(x)
                    
                    cache = feat_cache[idx][:, :, -1:, :, :]
                    cache = self.gather3d(cache)
                    # x = self.time_conv(
                    #     torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    x = self.time_conv(torch.cat([cache, x], 2))
                
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
                    x = self.scatter3d(x)
            # else:
            #     x = self.gather3d(x)
            #     cache = torch.zeros_like(x[:, :, :1])
            #     x = self.time_conv(torch.cat([cache, x], 2))
            #     x = self.scatter3d(x)
        return x
    
    def forward(self, x, feat_cache=None, feat_idx=[0], is_first_frame=False):
        if is_first_frame:
            return self.first_forward(x, feat_cache, feat_idx)
        else:
            return self.sparse_forward(x, feat_cache, feat_idx)

class ResidualBlock(SIGEModule3d):
    def __init__(self, in_dim, out_dim, dropout=0.0, main_block_size=6, shortcut_block_size=4):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # 各个模块得分开写
        self.norm1 = RMS_norm(in_dim, images=False)
        self.nonlinearity = nn.SiLU()
        self.conv1 = SIGECausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = RMS_norm(out_dim, images=False)
        self.conv2 = SIGECausalConv3d(out_dim, out_dim, 3, padding=1)

        self.main_gather = Gather3d(self.conv1, main_block_size, activation_name="silu", rms_norm=self.norm1)
        self.scatter_gather = ScatterGather3d(self.main_gather, activation_name="silu", rms_norm=self.norm2)
        
        if self.in_dim != self.out_dim:
            self.shortcut = SIGECausalConv3d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)
            self.shortcut_gather = Gather3d(self.shortcut, shortcut_block_size)
            self.scatter = ScatterWithBlockResidual3d(self.main_gather, self.shortcut_gather)
        else:
            self.scatter = Scatter3d(self.main_gather)

        # 兼容wan_vae在conv之前的cache
        self.scatter_for_time1 = Scatter3d(self.main_gather)
        self.scatter_for_time2 = Scatter3d(self.main_gather)
      
    def first_forward(self, x, feat_cache=None, feat_idx=[0]):
        h = x
        if self.in_dim != self.out_dim:
            h = self.shortcut(h)
        
        x = self.norm1(x)
        x = self.nonlinearity(x)

        cache_x = x[:, :, -CACHE_T:, :, :].clone()

        # 对conv特殊处理
        idx = feat_idx[0]
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

        x = self.norm2(x)
        x = self.nonlinearity(x)

        cache_x = x[:, :, -CACHE_T:, :, :].clone()

        idx = feat_idx[0]
        if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
            # cache last frame of last two chunk
            cache_x = torch.cat([
                feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                    cache_x.device), cache_x
            ],
                dim=2)

        x = self.conv2(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
        
        return x + h
    
    def sparse_forward(self, x, feat_cache=None, feat_idx=[0]):
        # 注意：Gather3d / ScatterGather3d 在 mode=="full" 时是 identity（只记录分辨率/缓存），
        # 不会执行 RMSNorm/激活融合逻辑；因此 full 模式下需要走“dense 计算 + 写 baseline cache”的路径，
        # 否则输出和 cache（feat_cache / scatter original_outputs）都会错，后续 sparse 也会基于错误 baseline。
        if self.mode == "full":
            h = x
            if self.in_dim != self.out_dim:
                h = self.shortcut_gather(h)
                h = self.shortcut(h)

            x = self.norm1(x)
            x = self.nonlinearity(x)

            # 记录 input_res，供后续 set_masks 使用
            x = self.main_gather(x)
            cache_x = self.scatter_for_time1(x)[:, :, -CACHE_T:, :, :].clone()

            idx = feat_idx[0]
            if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                    cache_x,
                ], dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
           

            # ScatterGather3d 的 baseline 需要是 conv1 的输出（pre-norm2）
            x = self.scatter_gather(x)

            x = self.norm2(x)
            x = self.nonlinearity(x)

            cache_x = self.scatter_for_time2(x)[:, :, -CACHE_T:, :, :].clone()

            idx = feat_idx[0]
            if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                    cache_x,
                ], dim=2)
            x = self.conv2(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
            
            x = self.scatter(x, h)
            
            return x

        h = x
        if self.in_dim != self.out_dim:
            h = self.shortcut_gather(h)
            h = self.shortcut(h)
                 
        # gather自带rms_norm, 激活函数silu
        x = self.main_gather(x)

        cache_x = self.scatter_for_time1(x)[:, :, -CACHE_T:, :, :].clone()

        # 对conv特殊处理
        idx = feat_idx[0]
        if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
            # cache last frame of last two chunk
            cache_x = torch.cat([
                feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                cache_x,
            ], dim=2)

        cache = self.main_gather(feat_cache[idx], is_cache_gather=True)
        x = self.conv1(x, cache)
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
       

        # gather自带激活函数silu
        x = self.scatter_gather(x)

        cache_x = self.scatter_for_time2(x)[:, :, -CACHE_T:, :, :].clone()

        idx = feat_idx[0]
        if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
            # cache last frame of last two chunk
            cache_x = torch.cat([
                feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                cache_x,
            ], dim=2)

        # 这个gather不需要norm和激活，只是简单的gather即可
        cache = self.main_gather(feat_cache[idx], is_cache_gather=True)
        x = self.conv2(x, cache)
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
    

        # no sparse: x:[1, 128, 512, 1024], self.scatter = Scatter3d
        # sparse: x:[405, 256, 4, 4]), self.scatter = ScatterWithBlockResidual3d
        x = self.scatter(x, h)
        
        return x
    
    def forward(self, x, feat_cache=None, feat_idx=[0], is_first_frame=False):
        if is_first_frame:
            return self.first_forward(x, feat_cache, feat_idx)
        else:
            return self.sparse_forward(x, feat_cache, feat_idx)
        

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


class Encoder3d(SIGEModel3d):
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
        # 不进行稀疏计算
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
                mode = 'downsample3d' if temperal_downsample[i] else 'downsample2d'
                downsamples.append(Resample(out_dim, resample_mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout))

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            # 不进行稀疏计算
            CausalConv3d(out_dim, z_dim, 3, padding=1))


    def forward(self, x, feat_cache=None, feat_idx=[0], is_first_frame=False):
        # 针对conv1的, 不进行稀疏计算
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
                x = layer(x, feat_cache, feat_idx, is_first_frame)
            else:
                x = layer(x)

        # middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx, is_first_frame)
            else:
                x = layer(x)

        # head
        # 针对head中的conv, 不进行稀疏计算
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


class Decoder3d(SIGEModel3d):
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
        # 不进行稀疏计算
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout), 
            AttentionBlock(dims[0]),
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
                upsamples.append(Resample(out_dim, resample_mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            # 不进行稀疏计算
            CausalConv3d(out_dim, 3, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0], is_first_frame=False):
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
                x = layer(x, feat_cache, feat_idx, is_first_frame)
            else:
                x = layer(x)

        # upsamples
        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx, is_first_frame)
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
        if isinstance(m, (CausalConv3d, SIGECausalConv3d)):
            count += 1
    return count

def dump_scatter_cache(model, tag=""):
    total = 0.0
    print(f"\n==== Scatter original_outputs {tag} ====")
    for name, m in model.named_modules():
        if hasattr(m, "original_outputs"):
            d = getattr(m, "original_outputs", {})
            mb = 0.0
            n = 0
            for v in d.values():
                if torch.is_tensor(v):
                    n += 1
                    mb += v.element_size() * v.nelement() / 1024**2
            if n > 0:
                print(f"{name}: entries={n}, {mb:.1f} MB")
            total += mb
    print(f"TOTAL Scatter cache: {total:.1f} MB")

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
        # 这两个也不稀疏计算
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

    def stream_encode(self, x, scale, mask=None, flow=None):
        # self.clear_cache()
        # cache
        # TODO: 看看如何使用mask和flow
        t = x.shape[2]
        if self.first_encode:
            self.first_encode = False
            self.clear_cache_encode()
            self._enc_conv_idx = [0]
            out = self.encoder(
                x[:, :, :1, :, :],
                feat_cache=self._enc_feat_map,
                feat_idx=self._enc_conv_idx,
                is_first_frame=True,
                )
            self._enc_conv_idx = [0]

            # 第一次只cache，没有flow的映射和mask
            self.encoder.set_mode("full")
            # dump_scatter_cache(self.encoder, "after t=1")  

            out_ = self.encoder(
                x[:, :, 1:, :, :],
                feat_cache=self._enc_feat_map,
                feat_idx=self._enc_conv_idx,
                )
            out = torch.cat([out, out_], 2)
            # dump_scatter_cache(self.encoder, "after t=4")   

        else:
            # set_maks
            # 通过光流/运动向量映射来改变scatter's的original_outputs
            self.encoder.set_mode("sparse")

            out=[]
            for i in range(t//4):
                # TODO: 下面传入的mask，是当前chunk相对于上一个chunk，计算得到的，flow也是
                # 测试的时候mask, flow取得固定值
                self.encoder.set_masks(mask)
                self.encoder.flow_cache(flow)
                # 每次稀疏计算之后，把结果写回原图进行覆盖
                self.encoder.set_sparse_update(True)

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

    def stream_decode(self, z, scale, mask=None, flow=None):
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
                is_first_frame=True,
                )
            self._conv_idx = [0]

            # 第一次只cache，没有flow的映射和mask
            self.decoder.set_mode("full")
            
            out_ = self.decoder(
                x[:, :, 1:, :, :],
                feat_cache=self._feat_map,
                feat_idx=self._conv_idx,
                )
            out = torch.cat([out, out_], 2)
        else:
            # set_maks
            # change scatter's original_outputs through flow
            self.decoder.set_mode("sparse")

            out = []
            for i in range(t):
                # TODO: 下面传入的mask，是当前chunk相对于上一个chunk，计算得到的，flow也是
                # 测试的时候mask, flow取得固定值
                self.decoder.set_masks(mask)
                self.decoder.flow_cache(flow)
                # 每次稀疏计算之后，把结果写回原图进行覆盖
                self.decoder.set_sparse_update(True)

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


def map_wanvae_ckpt_keys(sd: dict) -> dict:
    """
    1) ResidualBlock:  ...residual.{0,2,3,6}.{gamma,weight,bias}  -> ...{norm1,conv1,norm2,conv2}.{...}
    2) Resample:      ...resample.1.{weight,bias}                -> ...conv.{weight,bias}

    不改原 sd，返回新的 OrderedDict，可用于 strict=True。
    """
    out = OrderedDict()

    # ---- 1) residual Sequential -> flat modules ----
    residual_map = {
        0: "norm1",  # RMS_norm(in_dim)
        2: "conv1",  # CausalConv3d(in_dim -> out_dim)
        3: "norm2",  # RMS_norm(out_dim)
        6: "conv2",  # CausalConv3d(out_dim -> out_dim)
        # 1/4/5 (SiLU/SiLU/Dropout) 理论上无参数
    }
    pat_residual = re.compile(r"^(.*)\.residual\.(\d+)\.(weight|bias|gamma)$")

    # ---- 2) resample Sequential conv -> your conv ----
    # checkpoint: encoder.downsamples.2.resample.1.weight
    # your model: encoder.downsamples.2.conv.weight

    pat_resample = re.compile(r"^(.*)\.resample\.1\.(weight|bias)$")

    for k, v in sd.items():
        # resample first (更具体)
        m2 = pat_resample.match(k)
        if m2:
            prefix, param = m2.group(1), m2.group(2)
            new_k = f"{prefix}.conv.{param}"
            out[new_k] = v
            continue

        # residual
        m1 = pat_residual.match(k)
        if m1:
            prefix, idx_str, param = m1.group(1), m1.group(2), m1.group(3)
            idx = int(idx_str)
            if idx in residual_map:
                new_k = f"{prefix}.{residual_map[idx]}.{param}"
                out[new_k] = v
            else:
                # 理论上不会出现（SiLU/Dropout无参数），保留原样以便排查
                out[k] = v
            continue

        # default passthrough
        out[k] = v

    return out


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
    raw_sd = torch.load(pretrained_path, map_location=device, mmap=True)

    mapped_sd = map_wanvae_ckpt_keys(raw_sd)

    model.load_state_dict(mapped_sd, assign=True)
    
    print("*" * 40)
    print("Load Model Weight Successfully!!!")
    print("*" * 40)


    # 用mmap加速
    # model.load_state_dict(
    #     torch.load(pretrained_path, map_location=device, mmap=True), assign=True)

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
    def stream_encode(self, video: torch.Tensor, mask=None, flow=None) -> torch.Tensor:
        device, dtype = video.device, video.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                    1.0 / self.std.to(device=device, dtype=dtype)]

        return self.model.stream_encode(video, scale, mask, flow)
    
    def stream_decode_to_pixel(self, latent: torch.Tensor, mask=None, flow=None) -> torch.Tensor:
        zs = latent.permute(0, 2, 1, 3, 4)
        device, dtype = latent.device, latent.dtype
        zs = zs.to(device=device, dtype=torch.bfloat16)
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]
        output = self.model.stream_decode(zs, scale, mask, flow).float().clamp_(-1, 1)
        output = output.permute(0, 2, 1, 3, 4)
        return output

def load_mp4_as_tensor(video_path: str, max_frames: int = None, resize_hw: tuple[int, int] = None, normalize: bool = True) -> tuple[torch.Tensor, int]:
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

    time_axis = 0
    tx = x.shape[time_axis]
    ty = y.shape[time_axis]
    t = min(tx, ty)
    if tx != ty:
        x = np.take(x, indices=np.arange(t), axis=time_axis)
        y = np.take(y, indices=np.arange(t), axis=time_axis)

    mse = np.mean((x - y) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10((max_val ** 2) / mse)

def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--video",
        default=os.path.join(ASSETS_DIR, "input_red.mp4"),
        help="Input video path (mp4).",
    )
    parser.add_argument(
        "-o",
        "--out",
        default=os.path.join(ASSETS_DIR, "output_red.mp4"),
        help="Output video path (mp4).",
    )
    parser.add_argument(
        "--resize-hw",
        nargs=2,
        type=int,
        default=(480, 832),
        metavar=("H", "W"),
        help="Resize input frames to H W before encoding/decoding.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optionally truncate to first N frames.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Use torch.profiler to profile one chunk and print a summary table.",
    )
    parser.add_argument(
        "--profile-chunk",
        type=int,
        default=1,
        help="Which chunk index (0-based) to profile. Use 1+ to hit sparse mode.",
    )
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="Exit after profiling (skip saving output video).",
    )
    parser.add_argument(
        "--profile-row-limit",
        type=int,
        default=80,
        help="Row limit for torch.profiler table output.",
    )
    parser.add_argument(
        "--profile-trace",
        type=str,
        default=None,
        help="Optional path to export Chrome trace json (e.g. trace.json).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='Torch device string, e.g. "cuda:0" or "cpu" (default: auto).',
    )
    return parser.parse_args(argv)


def _print_profiler_focus(prof, *, row_limit: int = 80):
    events = prof.key_averages()

    has_cuda_events = any(getattr(e, "self_cuda_time_total", 0.0) > 0 for e in events)
    sort_by = "self_cuda_time_total" if has_cuda_events else "self_cpu_time_total"
    print(events.table(sort_by=sort_by, row_limit=row_limit))

    def _sum_by_key_substr(substr: str):
        matched = [e for e in events if substr in e.key]
        if not matched:
            return None
        return {
            "count": int(sum(getattr(e, "count", 0) for e in matched)),
            "self_cpu_ms": float(sum(getattr(e, "self_cpu_time_total", 0.0) for e in matched)) / 1000.0,
            "cpu_ms": float(sum(getattr(e, "cpu_time_total", 0.0) for e in matched)) / 1000.0,
            "self_cuda_ms": float(sum(getattr(e, "self_cuda_time_total", 0.0) for e in matched)) / 1000.0,
            "cuda_ms": float(sum(getattr(e, "cuda_time_total", 0.0) for e in matched)) / 1000.0,
        }

    focus = [
        "sige3d::forward_warp_cache_5d",
        "sige3d::forward_warp_cache_5d::grid_sample",
        "sige3d::gather3d",
        "sige3d::scatter_gather3d",
        "aten::grid_sampler_2d",
    ]
    print("\n---- Focus (sum by substring) ----")
    for k in focus:
        s = _sum_by_key_substr(k)
        if s is None:
            print(f"{k}: not found")
        else:
            print(
                f"{k}: count={s['count']} "
                f"self_cpu={s['self_cpu_ms']:.3f}ms cpu={s['cpu_ms']:.3f}ms "
                f"self_cuda={s['self_cuda_ms']:.3f}ms cuda={s['cuda_ms']:.3f}ms"
            )


@torch.no_grad()
def main(argv=None):
    args = _parse_args(argv)

    device_str = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    dtype = torch.bfloat16
    vae = WanVAEWrapper().to(device=device, dtype=dtype)

    video_path = args.video
    out_path = args.out

    input_video_original, original_fps = load_mp4_as_tensor(
        video_path, max_frames=args.max_frames, resize_hw=tuple(args.resize_hw)
    )
    input_video_original = input_video_original.unsqueeze(0).to(device=device, dtype=dtype)


    flow_np = np.load(os.path.join(ASSETS_DIR, "flow.npy"))    # (H, W, 2), float32
    mask_np = np.load(os.path.join(ASSETS_DIR, "mask.npy"))    # (H, W), uint8

    flow = torch.from_numpy(flow_np).to(device=device)   # torch.float32
    mask = torch.from_numpy(mask_np).to(device=device)   # torch.uint8

    # Encoder masks
    mask_enc = dilate_mask(mask, 5)
    masks_enc = downsample_mask(mask_enc, min_res=(4, 4), dilation=1)

    # Decoder masks
    mask_dec = dilate_mask(mask, 40)
    masks_dec = downsample_mask(mask_dec, min_res=(4, 4), dilation=0)


    recon_video_list = []
    # ---- 5,4,4,... 分块 encode ----
    for i, chunk in enumerate(iter_5_4_4_chunks(input_video_original)):
        do_profile = bool(args.profile and (i == args.profile_chunk))
        if do_profile:
            use_cuda = device.type == "cuda"
            if use_cuda:
                torch.cuda.synchronize()

            if torch_profile is not None and ProfilerActivity is not None:
                activities = [ProfilerActivity.CPU]
                if use_cuda:
                    activities.append(ProfilerActivity.CUDA)

                with torch_profile(
                    activities=activities,
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=False,
                ) as prof:
                    lat = vae.stream_encode(chunk, mask=masks_enc, flow=flow)   # 逐块喂进去
                    lat = lat.permute(0, 2, 1, 3, 4)
                    video = vae.stream_decode_to_pixel(lat, mask=masks_dec, flow=flow)
                    if hasattr(prof, "step"):
                        prof.step()
                    if use_cuda:
                        torch.cuda.synchronize()
            else:
                from torch.autograd.profiler import profile as autograd_profile

                with autograd_profile(
                    use_cuda=use_cuda,
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=False,
                ) as prof:
                    lat = vae.stream_encode(chunk, mask=masks_enc, flow=flow)   # 逐块喂进去
                    lat = lat.permute(0, 2, 1, 3, 4)
                    video = vae.stream_decode_to_pixel(lat, mask=masks_dec, flow=flow)
                    if use_cuda:
                        torch.cuda.synchronize()

            if args.profile_trace is not None:
                prof.export_chrome_trace(args.profile_trace)
                print("exported trace to", args.profile_trace)

            _print_profiler_focus(prof, row_limit=args.profile_row_limit)
            if args.profile_only:
                # MACs (similar to `sige/example.py`):
                # - "full"  : dense inference (no sparsity)
                # - "profile": symbolic sparse inference (counts only active blocks)
                enc = vae.model.encoder
                dec = vae.model.decoder

                with torch.no_grad():
                    # Encoder MACs
                    enc.set_mode("full")
                    full_macs_enc = profile_macs(enc, (chunk,))
                    enc.set_mode("profile")
                    enc.set_masks(masks_enc)
                    sige_macs_enc = profile_macs(enc, (chunk,))

                    # Decoder MACs (profile 1 latent frame, then scale by latent T)
                    z = lat.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
                    t_lat = int(z.size(2))
                    z0 = z[:, :, :1, :, :]

                    dec.set_mode("full")
                    full_macs_dec_frame = profile_macs(dec, (z0,))
                    dec.set_mode("profile")
                    dec.set_masks(masks_dec)
                    sige_macs_dec_frame = profile_macs(dec, (z0,))

                full_macs_total = full_macs_enc + full_macs_dec_frame * max(t_lat, 1)
                sige_macs_total = sige_macs_enc + sige_macs_dec_frame * max(t_lat, 1)

                print("\n---- MACs ----")
                print(f"[MACs] Encoder full: {full_macs_enc/1e9:.3f} G, SIGE(profile): {sige_macs_enc/1e9:.3f} G")
                print(
                    f"[MACs] Decoder full(per-frame): {full_macs_dec_frame/1e9:.3f} G, "
                    f"SIGE(profile)(per-frame): {sige_macs_dec_frame/1e9:.3f} G, "
                    f"T_lat={t_lat}"
                )
                print(f"[MACs] Total(full): {full_macs_total/1e9:.3f} G, Total(SIGE): {sige_macs_total/1e9:.3f} G")
                return
        else:
            lat = vae.stream_encode(chunk, mask=masks_enc, flow=flow)   # 逐块喂进去
            lat = lat.permute(0, 2, 1, 3, 4)
            video = vae.stream_decode_to_pixel(lat, mask=masks_dec, flow=flow)

        recon_video_list.append(video)
        print(f"[Chunk {i}] done!!!")
        # break     


    video = torch.cat(recon_video_list, dim=1)

    video = (video * 0.5 + 0.5).clamp(0, 1)
    video = video[0].permute(0, 2, 3, 1).contiguous()
    video = (video * 255).round().clamp(0,255).to(torch.uint8).cpu().numpy()

    torchvision.io.write_video(out_path, video, fps=int(original_fps))
    print("saved to", out_path)

    save_dir = out_path.replace(".mp4", "_frames")
    os.makedirs(save_dir, exist_ok=True)

    for t, frame in enumerate(video):
        # frame: (H, W, 3), uint8
        img = Image.fromarray(frame, mode="RGB")
        img.save(os.path.join(save_dir, f"frame_{t:03d}.png"))

    print(f"saved {len(video)} frames to {save_dir}")


    # cal PSNR
    input_video_original = (input_video_original * 0.5 + 0.5).clamp(0, 1)
    input_video_original = input_video_original[0].permute(1, 2, 3, 0).contiguous()
    input_video_original = (input_video_original * 255).round().clamp(0,255).to(torch.uint8).cpu().numpy()

    psnr_video = psnr_np(input_video_original, video)
    print(f"[PSNR] Full video: {psnr_video:.2f} dB")



if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    main()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.3f} GB")
