from __future__ import annotations

import os
from typing import Tuple

import torch
import torchvision
import torchvision.transforms.functional as TF
from einops import repeat


def load_image_as_tensor(
    path: str,
    *,
    resize_hw: Tuple[int, int] | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Returns: image tensor with shape [1, 3, H, W], value range [-1, 1] if normalize=True.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    img = torchvision.io.read_image(path)  # [C,H,W], uint8
    img = img[:3]  # enforce RGB if input has alpha
    img = img.to(dtype=torch.float32) / 255.0
    if resize_hw is not None:
        img = TF.resize(img, resize_hw, antialias=True)
    if normalize:
        img = img * 2.0 - 1.0

    img = img.unsqueeze(0)  # [1,3,H,W]
    if device is not None:
        img = img.to(device=device)
    img = img.to(dtype=dtype)
    return img


def repeat_image_to_video(img: torch.Tensor, num_frames: int) -> torch.Tensor:
    """
    img: [1, 3, H, W] -> video: [1, 3, T, H, W]
    """
    if img.dim() != 4:
        raise ValueError(f"Expected img with dim=4, got {img.shape}")
    return repeat(img, "b c h w -> b c t h w", t=int(num_frames))


def save_video_tensor(
    out_path: str,
    video: torch.Tensor,
    *,
    fps: int = 16,
    from_range: str = "minus1_1",
) -> None:
    """
    video: [1, 3, T, H, W] (or [B,3,T,H,W] with B==1) saved as mp4.
    """
    if video.dim() != 5:
        raise ValueError(f"Expected video with dim=5, got {video.shape}")
    if video.size(0) != 1 or video.size(1) != 3:
        raise ValueError(f"Expected video shape [1,3,T,H,W], got {video.shape}")

    x = video.detach().cpu()
    if from_range == "minus1_1":
        x = (x * 0.5 + 0.5).clamp(0, 1)
    elif from_range == "0_1":
        x = x.clamp(0, 1)
    else:
        raise ValueError(f"Unknown from_range: {from_range}")

    # [1,3,T,H,W] -> [T,H,W,3] uint8
    x = x[0].permute(2, 3, 4, 1).contiguous()
    x = (x * 255.0).round().clamp(0, 255).to(torch.uint8)
    torchvision.io.write_video(out_path, x, fps=fps)

