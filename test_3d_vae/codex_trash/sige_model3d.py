from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from sige3d import (
    Gather3d,
    Scatter3d,
    ScatterGather3d,
    ScatterWithBlockResidual3d,
    SIGECausalConv3d,
    SIGEModel3d,
    SIGEModule3d,
)


class RMSNorm3d(nn.Module):
    """
    Wan-VAE style RMSNorm for causal streaming.

    - Normalizes across channel dim only, so it does NOT mix future frames.
    - Works for both dense tensors [B,C,T,H,W] and sparse blocks [B*N,C,T,h,w].
    """

    def __init__(self, dim: int, bias: bool = False):
        super().__init__()
        self.dim = int(dim)
        self.scale = self.dim**0.5
        self.gamma = nn.Parameter(torch.ones(self.dim, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(self.dim, 1, 1, 1)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W] or [B*N, C, T, h, w]
        x = F.normalize(x, dim=1) * self.scale
        x = x * self.gamma
        if self.bias is not None:
            x = x + self.bias
        return x


def nonlinearity(x: torch.Tensor) -> torch.Tensor:
    # swish (== SiLU)
    return F.silu(x)


class AlwaysDenseCausalConv3d(SIGECausalConv3d):
    """
    A causal Conv3d that stays dense even when the model is in sparse mode.

    Used to mirror the 2D SIGE VAE behavior where some layers (e.g. conv_in/conv_out)
    remain dense for simplicity.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Force dense semantics: input is always [B,C,T,H,W] and needs spatial padding.
        self.check_dtype(x)
        self.check_dim(x)
        x = self._pad_and_concat_time(x, pad_spatial=True)
        return F.conv3d(x, self.weight, self.bias, self.stride, (0, 0, 0), self.dilation, self.groups)


class SIGEResnetBlock3d(SIGEModule3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        temb_channels: int = 0,
        main_block_size: int = 6,
        shortcut_block_size: int = 4,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        out_channels = in_channels if out_channels is None else int(out_channels)
        self.out_channels = out_channels

        if main_block_size is None:
            assert shortcut_block_size is None

        main_support_sparse = main_block_size is not None
        MainConv3d = SIGECausalConv3d if main_support_sparse else AlwaysDenseCausalConv3d

        self.norm1 = RMSNorm3d(in_channels)
        self.conv1 = MainConv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = RMSNorm3d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = MainConv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if main_support_sparse:
            self.main_gather = Gather3d(self.conv1, main_block_size, activation_name="identity")
            self.scatter_gather = ScatterGather3d(self.main_gather, activation_name="identity")

        if self.in_channels != self.out_channels:
            shortcut_support_sparse = shortcut_block_size is not None
            ShortcutConv3d = SIGECausalConv3d if shortcut_support_sparse else AlwaysDenseCausalConv3d
            self.nin_shortcut = ShortcutConv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            if shortcut_support_sparse:
                self.shortcut_gather = Gather3d(self.nin_shortcut, shortcut_block_size)
                self.scatter = ScatterWithBlockResidual3d(self.main_gather, self.shortcut_gather)
            elif main_support_sparse:
                self.scatter = Scatter3d(self.main_gather)
        else:
            shortcut_support_sparse = False
            if main_support_sparse:
                self.scatter = Scatter3d(self.main_gather)

        self.main_support_sparse = main_support_sparse
        self.shortcut_support_sparse = shortcut_support_sparse

    def forward(self, x: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.mode == "full":
            return self.full_forward(x, temb)
        if self.mode in ("sparse", "profile"):
            return self.sparse_forward(x, temb)
        raise NotImplementedError(f"Unknown mode: {self.mode}")

    def full_forward(self, x: torch.Tensor, temb: Optional[torch.Tensor]) -> torch.Tensor:
        h = x
        if self.in_channels != self.out_channels:
            if self.shortcut_support_sparse:
                x = self.shortcut_gather(x)
            x = self.nin_shortcut(x)

        if self.main_support_sparse:
            h = self.main_gather(h)

        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if self.main_support_sparse:
            h = self.scatter_gather(h)

        if temb is not None:
            temb = self.temb_proj(nonlinearity(temb))
            temb = temb.view(*temb.shape, 1, 1, 1)
            h = h + temb
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.main_support_sparse:
            return self.scatter(h, x)
        return h + x

    def sparse_forward(self, x: torch.Tensor, temb: Optional[torch.Tensor]) -> torch.Tensor:
        h = x
        if self.in_channels != self.out_channels:
            if self.shortcut_support_sparse:
                x = self.shortcut_gather(x)
            x = self.nin_shortcut(x)

        if self.main_support_sparse:
            # Dense -> blocks (spatial gather):
            #   [B, C, T, H, W] -> [B*num_active, C, T, bH, bW]
            h = self.main_gather(h)
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if self.main_support_sparse:
            # Scatter updates into full baseline, then gather blocks for the next conv:
            #   conv1 out blocks [B*num_active, C', T, rH, rW] -> blocks [B*num_active, C', T, bH, bW]
            h = self.scatter_gather(h)
        if temb is not None:
            temb = self.temb_proj(nonlinearity(temb))
            temb = temb.view(*temb.shape, 1, 1, 1)
            h = h + temb
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.main_support_sparse:
            return self.scatter(h, x)
        return h + x


class SIGEDownsample3d(SIGEModule3d):
    def __init__(self, in_channels: int, block_size: int = 6):
        super().__init__()
        # No asymmetric padding in conv3d, do it ourselves (only spatial).
        self.conv = SIGECausalConv3d(in_channels, in_channels, kernel_size=3, stride=(1, 2, 2), padding=(1, 0, 0))
        self.gather = Gather3d(self.conv, block_size=block_size)
        self.scatter = Scatter3d(self.gather)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gather(x)
        if self.mode == "full":
            # pad (right, bottom) by 1: (W_left,W_right,H_left,H_right,T_left,T_right)
            x = F.pad(x, (0, 1, 0, 1, 0, 0), mode="constant", value=0)
        x = self.conv(x)
        x = self.scatter(x)
        return x


class SIGEUpsample3d(SIGEModule3d):
    def __init__(self, in_channels: int, block_size: int = 6):
        super().__init__()
        self.conv = SIGECausalConv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.gather = Gather3d(self.conv, block_size=block_size)
        self.scatter = Scatter3d(self.gather)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Only spatial upsample; keep time axis unchanged.
        x = F.interpolate(x, scale_factor=(1.0, 2.0, 2.0), mode="nearest")
        x = self.gather(x)
        x = self.conv(x)
        x = self.scatter(x)
        return x


class SIGEEncoder3d(SIGEModel3d):
    def __init__(
        self,
        *,
        ch: int = 64,
        in_channels: int = 3,
        z_channels: int = 4,
        ch_mult: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ch = int(ch)
        self.in_channels = int(in_channels)
        self.z_channels = int(z_channels)
        self.ch_mult = tuple(int(x) for x in ch_mult)
        self.num_resolutions = len(self.ch_mult)
        self.num_res_blocks = int(num_res_blocks)

        # conv_in stays dense (like the 2D SIGE VAE).
        self.conv_in = AlwaysDenseCausalConv3d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_ch = self.ch
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            out_ch = self.ch * self.ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(SIGEResnetBlock3d(curr_ch, out_ch, dropout=dropout, temb_channels=0))
                curr_ch = out_ch
            down = nn.Module()
            down.block = block
            if i_level != self.num_resolutions - 1:
                down.downsample = SIGEDownsample3d(curr_ch)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = SIGEResnetBlock3d(curr_ch, curr_ch, dropout=dropout, temb_channels=0)
        self.mid.block_2 = SIGEResnetBlock3d(curr_ch, curr_ch, dropout=dropout, temb_channels=0)

        # end (dense norm/conv, like the 2D SIGE VAE)
        self.norm_out = RMSNorm3d(curr_ch)
        self.conv_out = AlwaysDenseCausalConv3d(curr_ch, z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temb = None

        # downsampling
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h, temb)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class SIGEDecoder3d(SIGEModel3d):
    def __init__(
        self,
        *,
        ch: int = 64,
        out_channels: int = 3,
        z_channels: int = 4,
        ch_mult: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ch = int(ch)
        self.out_channels = int(out_channels)
        self.z_channels = int(z_channels)
        self.ch_mult = tuple(int(x) for x in ch_mult)
        self.num_resolutions = len(self.ch_mult)
        self.num_res_blocks = int(num_res_blocks)

        block_in = self.ch * self.ch_mult[self.num_resolutions - 1]

        # z to block_in (dense)
        self.conv_in = AlwaysDenseCausalConv3d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = SIGEResnetBlock3d(block_in, block_in, dropout=dropout, temb_channels=0)
        self.mid.block_2 = SIGEResnetBlock3d(block_in, block_in, dropout=dropout, temb_channels=0)

        # upsampling
        self.up = nn.ModuleList()
        curr_ch = block_in
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            out_ch = self.ch * self.ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(SIGEResnetBlock3d(curr_ch, out_ch, dropout=dropout, temb_channels=0))
                curr_ch = out_ch
            up = nn.Module()
            up.block = block
            if i_level != 0:
                up.upsample = SIGEUpsample3d(curr_ch)
            self.up.insert(0, up)

        # end (dense)
        self.norm_out = RMSNorm3d(curr_ch)
        self.conv_out = AlwaysDenseCausalConv3d(curr_ch, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        temb = None

        h = self.conv_in(z)
        h = self.mid.block_1(h, temb)
        h = self.mid.block_2(h, temb)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class SIGEAutoencoder3d(nn.Module):
    def __init__(
        self,
        *,
        ch: int = 64,
        in_channels: int = 3,
        out_channels: int = 3,
        z_channels: int = 4,
        ch_mult: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = SIGEEncoder3d(
            ch=ch,
            in_channels=in_channels,
            z_channels=z_channels,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
        )
        self.decoder = SIGEDecoder3d(
            ch=ch,
            out_channels=out_channels,
            z_channels=z_channels,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
        )

    def clear_cache(self):
        self.encoder.clear_cache()
        self.decoder.clear_cache()

    def clear_stream_cache(self):
        self.encoder.clear_stream_cache()
        self.decoder.clear_stream_cache()

    @torch.no_grad()
    def stream_encode(self, video: torch.Tensor, *, cache_per_chunk: bool = False) -> torch.Tensor:
        """
        Full-mode streaming encode for long videos.

        - Keeps causal Conv3d temporal caches across chunks.
        - By default overwrites `cache_id=0` to avoid growing SIGE full-baseline caches.

        Input/Output:
          - video: [B, 3, T, H, W]
          - latent: [B, z_channels, T, H', W']  (time length unchanged in this toy SIGE-3D VAE)
        """
        from chunk_utils import iter_5_4_4_chunks
        from streaming_sige import stream_forward_in_chunks

        self.encoder.clear_cache()
        self.encoder.clear_stream_cache()
        self.encoder.set_mode("full")
        return stream_forward_in_chunks(
            self.encoder,
            video,
            chunker=iter_5_4_4_chunks,
            cache_per_chunk=cache_per_chunk,
        )

    @torch.no_grad()
    def stream_decode(self, latent: torch.Tensor, *, cache_per_chunk: bool = False) -> torch.Tensor:
        """
        Full-mode streaming decode for long latent sequences.

        Input/Output:
          - latent: [B, z_channels, T, H, W]
          - video:  [B, 3, T, H', W']
        """
        from chunk_utils import iter_5_4_4_chunks
        from streaming_sige import stream_forward_in_chunks

        self.decoder.clear_cache()
        self.decoder.clear_stream_cache()
        self.decoder.set_mode("full")
        return stream_forward_in_chunks(
            self.decoder,
            latent,
            chunker=iter_5_4_4_chunks,
            cache_per_chunk=cache_per_chunk,
        )
