# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""OBB-aware masks for Race_Sea HBS.

HBS protects a 1.15x dilated oriented bounding box (OBB) region and smooths
the remaining background. The mask uses exact overlap between each dilated OBB
and each axis-aligned feature-map cell via the Separating Axis Theorem (SAT).

This module is training-only and adds no inference cost.
"""

from __future__ import annotations

import torch


HBS_OBB_DILATION = 1.15


@torch.no_grad()
def dilated_obb_cell_overlap_mask(
    feature: torch.Tensor,
    batch: dict[str, torch.Tensor],
    scale: float = HBS_OBB_DILATION,
) -> torch.Tensor:
    """Rasterize GT boxes as feature cells overlapping a dilated OBB."""
    if feature.ndim != 4:
        raise ValueError(f"Expected BCHW feature tensor, got shape={tuple(feature.shape)}")
    if scale <= 0:
        raise ValueError(f"OBB dilation scale must be positive, got {scale}")

    bs, _, height, width = feature.shape
    device = feature.device
    mask = torch.zeros((bs, 1, height, width), device=device, dtype=torch.bool)

    boxes = batch["bboxes"].to(device=device, dtype=torch.float32)
    batch_idx = batch["batch_idx"].to(device=device, dtype=torch.long).view(-1)

    if boxes.numel() == 0:
        return mask
    if boxes.ndim != 2 or boxes.shape[1] < 4:
        raise ValueError(f"Expected bboxes with >=4 columns, got shape={tuple(boxes.shape)}")
    if boxes.shape[0] != batch_idx.numel():
        raise ValueError(
            f"bboxes and batch_idx size mismatch: {boxes.shape[0]} boxes vs {batch_idx.numel()} indices"
        )

    xs = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
    ys = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    cell_half_x = 0.5 / width
    cell_half_y = 0.5 / height

    for image_index in range(bs):
        image_boxes = boxes[batch_idx == image_index]
        if image_boxes.numel() == 0:
            continue

        centers = image_boxes[:, :2]
        half_w = image_boxes[:, 2].clamp_min(0) * (0.5 * scale)
        half_h = image_boxes[:, 3].clamp_min(0) * (0.5 * scale)

        angle = (
            image_boxes[:, 4]
            if image_boxes.shape[1] >= 5
            else torch.zeros_like(image_boxes[:, 0])
        )

        cos_a = angle.cos()
        sin_a = angle.sin()
        abs_cos = cos_a.abs()
        abs_sin = sin_a.abs()

        dx = xx[None] - centers[:, 0, None, None]
        dy = yy[None] - centers[:, 1, None, None]

        proj_x = (
            half_w * abs_cos + half_h * abs_sin + cell_half_x
        )[:, None, None]
        proj_y = (
            half_w * abs_sin + half_h * abs_cos + cell_half_y
        )[:, None, None]

        overlap_x = dx.abs() <= proj_x
        overlap_y = dy.abs() <= proj_y

        local_u = dx * cos_a[:, None, None] + dy * sin_a[:, None, None]
        local_v = -dx * sin_a[:, None, None] + dy * cos_a[:, None, None]

        cell_radius_u = (
            cell_half_x * abs_cos + cell_half_y * abs_sin
        )[:, None, None]
        cell_radius_v = (
            cell_half_x * abs_sin + cell_half_y * abs_cos
        )[:, None, None]

        overlap_u = local_u.abs() <= half_w[:, None, None] + cell_radius_u
        overlap_v = local_v.abs() <= half_h[:, None, None] + cell_radius_v

        overlaps = overlap_x & overlap_y & overlap_u & overlap_v
        mask[image_index, 0] = overlaps.any(dim=0)

    return mask
