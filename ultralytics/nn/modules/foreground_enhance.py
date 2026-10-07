# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Training-only OBB-aware foreground detail enhancement for Race_Sea."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


@torch.no_grad()
def exact_obb_foreground_mask(
    feature: torch.Tensor,
    batch: dict[str, torch.Tensor],
    ensure_one_cell: bool = True,
) -> torch.Tensor:
    """Rasterize normalized GT OBBs onto a feature map using feature-cell centers."""
    if feature.ndim != 4:
        raise ValueError(f"Expected BCHW feature tensor, got shape={tuple(feature.shape)}")

    bs, _, height, width = feature.shape
    mask = torch.zeros((bs, 1, height, width), device=feature.device, dtype=torch.bool)
    boxes = batch["bboxes"].to(device=feature.device, dtype=feature.dtype)
    batch_idx = batch["batch_idx"].to(device=feature.device, dtype=torch.long).view(-1)

    if boxes.numel() == 0:
        return mask
    if boxes.ndim != 2 or boxes.shape[1] < 4:
        raise ValueError(f"Expected bboxes with >=4 columns, got shape={tuple(boxes.shape)}")
    if boxes.shape[0] != batch_idx.numel():
        raise ValueError(f"bboxes/batch_idx mismatch: {boxes.shape[0]} vs {batch_idx.numel()}")

    xs = (torch.arange(width, device=feature.device, dtype=feature.dtype) + 0.5) / width
    ys = (torch.arange(height, device=feature.device, dtype=feature.dtype) + 0.5) / height
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    for image_index in range(bs):
        image_boxes = boxes[batch_idx == image_index]
        if image_boxes.numel() == 0:
            continue

        centers = image_boxes[:, 0:2]
        sizes = image_boxes[:, 2:4].clamp_min(0)
        angle = image_boxes[:, 4] if image_boxes.shape[1] >= 5 else torch.zeros_like(image_boxes[:, 0])
        cos_a = angle.cos()[:, None, None]
        sin_a = angle.sin()[:, None, None]

        dx = xx[None] - centers[:, 0, None, None]
        dy = yy[None] - centers[:, 1, None, None]
        local_x = dx * cos_a + dy * sin_a
        local_y = -dx * sin_a + dy * cos_a
        inside = (
            (local_x.abs() <= sizes[:, 0, None, None] * 0.5)
            & (local_y.abs() <= sizes[:, 1, None, None] * 0.5)
        )

        if ensure_one_cell:
            missing = ~inside.flatten(1).any(1)
            for box_id in torch.nonzero(missing, as_tuple=False).flatten().tolist():
                cx = centers[box_id, 0].clamp(0, 1 - torch.finfo(feature.dtype).eps)
                cy = centers[box_id, 1].clamp(0, 1 - torch.finfo(feature.dtype).eps)
                ix = int(torch.clamp((cx * width).floor(), 0, width - 1).item())
                iy = int(torch.clamp((cy * height).floor(), 0, height - 1).item())
                inside[box_id, iy, ix] = True

        mask[image_index, 0] = inside.any(dim=0)

    return mask


class ForegroundDetailEnhancer(nn.Module):
    """Enhance local detail residuals only inside exact GT OBBs."""

    def __init__(self, gain: float = 0.25, kernel_size: int = 3) -> None:
        super().__init__()
        if gain < 0:
            raise ValueError(f"gain must be >= 0, got {gain}")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
        self.gain = float(gain)
        self.kernel_size = int(kernel_size)

    def forward(
        self,
        feature: torch.Tensor,
        batch: dict[str, torch.Tensor],
        base_feature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Add P3 detail from the original shared feature onto an optional HBS base feature."""
        base = feature if base_feature is None else base_feature
        if base.shape != feature.shape:
            raise ValueError(f"feature/base mismatch: {tuple(feature.shape)} vs {tuple(base.shape)}")
        if self.gain == 0:
            return base

        mask = exact_obb_foreground_mask(feature, batch).to(dtype=feature.dtype)
        if not mask.any():
            return base

        pad = self.kernel_size // 2
        padded = F.pad(feature, (pad, pad, pad, pad), mode="replicate") if pad else feature
        low = F.avg_pool2d(padded, kernel_size=self.kernel_size, stride=1) if pad else feature
        detail = feature - low
        return base + self.gain * mask * detail
