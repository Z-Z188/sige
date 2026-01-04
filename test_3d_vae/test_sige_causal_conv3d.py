import argparse
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchprofile import profile_macs

from sige.utils import reduce_mask


class CausalConv3d(nn.Conv3d):
    """
    Causal 3D convolution with left-only padding on time.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # nn.Conv3d padding order is (T, H, W), but F.pad uses (W, H, T).
        self._padding = (self.padding[2], self.padding[2],
                         self.padding[1], self.padding[1],
                         2 * self.padding[0], 0)
        # Disable internal padding; we do explicit padding.
        self.padding = (0, 0, 0)
        # Cached padding info for sparse gather/scatter.
        self.time_padding = self._padding[4]
        self.spatial_padding = (self._padding[2], self._padding[0])

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return super().forward(x)


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
            if isinstance(module, SIGEModule3d):
                module.set_mask(masks, cache, self.timestamp)

    def set_mode(self, mode):
        self.mode = mode
        for module in self.modules():
            if isinstance(module, SIGEModule3d):
                module.set_mode(mode)

    def clear_cache(self):
        for module in self.modules():
            if isinstance(module, SIGEModule3d):
                module.clear_cache()

    def set_cache_id(self, cache_id):
        for module in self.modules():
            if isinstance(module, SIGEModule3d):
                module.set_cache_id(cache_id)

    def set_sparse_update(self, sparse_update):
        for module in self.modules():
            if isinstance(module, SIGEModule3d):
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
            res = tuple(self.input_res)
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

class ExampleCausalConv3dModule(nn.Module):
    def __init__(
        self,
        in_channels=16,
        out_channels=32,
        kernel_size=(3, 3, 3),
        stride=(1, 1, 1),
        padding=(1, 1, 1),
        block_size=6,
    ):
        super().__init__()
        self.conv = SIGECausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        self.gather = Gather3d(self.conv, block_size=block_size)
        self.scatter = Scatter3d(self.gather)

    def forward(self, x):
        x = self.gather(x)
        x = self.conv(x)
        x = self.scatter(x)
        return x


class ExampleCausalConv3dModel(SIGEModel3d):
    def __init__(self, **kwargs):
        super().__init__()
        self.example_module = ExampleCausalConv3dModule(**kwargs)

    def forward(self, x):
        return self.example_module(x)


@torch.no_grad()
def run_sparse_causal_conv3d_test():
    device = torch.device("cuda")
    torch.manual_seed(0)

    mask_path = os.path.join(os.path.dirname(__file__), "..", "assets", "mask.npy")
    mask = torch.from_numpy(np.load(mask_path)).to(device=device, dtype=torch.float32)
    h, w = mask.shape

    b, c, t = 1, 16, 100
    frame = torch.randn((b, c, h, w), device=device)
    original_input = frame.unsqueeze(2).repeat(1, 1, t, 1, 1)
    edited_input = original_input + torch.randn_like(original_input) * mask[None, None, None]

    model = ExampleCausalConv3dModel(in_channels=c, out_channels=32, block_size=6).to(device)
    model.eval()

    model.set_mode("full")
    full_output = model(edited_input)
    full_macs = profile_macs(model, (edited_input,))

    model.set_mode("full")
    _ = model(original_input)

    model.set_mode("sparse")
    model.set_masks({(h, w): mask})
    sparse_output = model(edited_input)

    model.set_mode("profile")
    sparse_macs = profile_macs(model, (edited_input,))

    max_error = (full_output - sparse_output).abs().max().item()
    masked_pct = (mask > 0).float().mean().item() * 100.0
    print("Max Error: %.6f" % max_error)
    print("Masked Region: %.2f%%" % masked_pct)
    print("Full MACs: %.2fM" % (full_macs / 1e6))
    print("Sparse MACs: %.2fM" % (sparse_macs / 1e6))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"])
    return parser.parse_args()


if __name__ == "__main__":
    run_sparse_causal_conv3d_test()
