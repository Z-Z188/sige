from __future__ import annotations

from typing import Dict, Tuple

import torch

from sige.utils import compute_difference_mask, dilate_mask, downsample_mask


def compute_sdedit_masks(
    init_img: torch.Tensor,
    edited_img: torch.Tensor,
    *,
    min_res: Tuple[int, int] = (4, 4),
) -> tuple[Dict[Tuple[int, int], torch.Tensor], Dict[Tuple[int, int], torch.Tensor], torch.Tensor]:
    """
    Replicates `stable-diffusion/runners/sdedit_runner.py` mask logic.

    Inputs:
      - init_img / edited_img: [1, 3, H, W] in [-1, 1]

    Returns:
      - masks_enc: dict[(h,w)] -> bool mask, encoder setting
      - masks_dec: dict[(h,w)] -> bool mask, decoder setting
      - diff_mask: [H, W] bool
    """
    diff_mask = compute_difference_mask(init_img, edited_img)  # [H,W]

    # Encoder masks
    diff_enc = dilate_mask(diff_mask, 5)
    masks_enc = downsample_mask(diff_enc, min_res=(4, 4), dilation=1)

    # Decoder masks
    diff_dec = dilate_mask(diff_mask, 40)
    masks_dec = downsample_mask(diff_dec, min_res=min_res, dilation=0)

    return masks_enc, masks_dec, diff_mask

