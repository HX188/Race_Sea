# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Mask utilities for Race_Sea HBS experiments.

This file intentionally keeps the exact-OBB rasterization separate from
``head.py`` so HBS mask experiments can be enabled/disabled with minimal
changes to the baseline head implementation.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def exact_obb_cell_overlap_mask(feature: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Rasterize GT OBBs using exact OBB-vs-feature-cell overlap semantics.

    This function mirrors the *overlap* semantics of the original HBS AABB mask:
    a feature cell is foreground when its rectangular area has positive-area
    intersection with at least one GT box. The only intended variable change is
    the box geometry itself: axis-aligned enclosing rectangle -> true OBB.

    The rectangle intersection test uses the Separating Axis Theorem (SAT) on
    four axes (image x/y and the two OBB local axes), vectorized over all cells.

    Args:
        feature: BCHW feature tensor.
        batch: Ultralytics training batch containing normalized ``bboxes`` in
            ``(cx, cy, w, h, angle)`` form and ``batch_idx``.

    Returns:
        Boolean tensor shaped ``(B, 1, H, W)``.
    """
    if feature.ndim != 4:
        raise ValueError(f"Expected BCHW feature tensor, got shape={tuple(feature.shape)}")

    bs, _, height, width = feature.shape
    mask = torch.zeros((bs, 1, height, width), device=feature.device, dtype=torch.bool)

    boxes = batch["bboxes"].to(device=feature.device, dtype=torch.float32)
    batch_idx = batch["batch_idx"].to(device=feature.device, dtype=torch.long).view(-1)

    if boxes.numel() == 0:
        return mask
    if boxes.ndim != 2 or boxes.shape[1] < 4:
        raise ValueError(f"Expected bboxes with at least 4 columns, got shape={tuple(boxes.shape)}")
    if boxes.shape[0] != batch_idx.numel():
        raise ValueError(
            f"bboxes and batch_idx size mismatch: {boxes.shape[0]} boxes vs {batch_idx.numel()} indices"
        )

    # Feature-cell centers and half extents in normalized image coordinates.
    cell_half_x = 0.5 / width
    cell_half_y = 0.5 / height
    xs = (torch.arange(width, device=feature.device, dtype=torch.float32) + 0.5) / width
    ys = (torch.arange(height, device=feature.device, dtype=torch.float32) + 0.5) / height
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    # Small tolerance prevents floating-point roundoff from turning pure boundary
    # touching into a positive-area overlap. The baseline HBS uses strict overlap
    # tests, so this keeps the semantics aligned.
    eps = 1e-7

    for image_index in range(bs):
        image_boxes = boxes[batch_idx == image_index]
        if image_boxes.numel() == 0:
            continue

        centers = image_boxes[:, 0:2]
        half_w = image_boxes[:, 2].clamp_min(0.0) * 0.5
        half_h = image_boxes[:, 3].clamp_min(0.0) * 0.5
        angle = image_boxes[:, 4] if image_boxes.shape[1] >= 5 else torch.zeros_like(half_w)

        cos_a = angle.cos()
        sin_a = angle.sin()
        abs_cos = cos_a.abs()
        abs_sin = sin_a.abs()

        dx = xx[None] - centers[:, 0, None, None]
        dy = yy[None] - centers[:, 1, None, None]

        # SAT axis 1/2: image x and image y.
        overlap_x = dx.abs() < (
            cell_half_x + half_w[:, None, None] * abs_cos[:, None, None] + half_h[:, None, None] * abs_sin[:, None, None] - eps
        )
        overlap_y = dy.abs() < (
            cell_half_y + half_w[:, None, None] * abs_sin[:, None, None] + half_h[:, None, None] * abs_cos[:, None, None] - eps
        )

        # SAT axis 3/4: the two local axes of each OBB.
        proj_u = (dx * cos_a[:, None, None] + dy * sin_a[:, None, None]).abs()
        proj_v = (-dx * sin_a[:, None, None] + dy * cos_a[:, None, None]).abs()

        overlap_u = proj_u < (
            half_w[:, None, None]
            + cell_half_x * abs_cos[:, None, None]
            + cell_half_y * abs_sin[:, None, None]
            - eps
        )
        overlap_v = proj_v < (
            half_h[:, None, None]
            + cell_half_x * abs_sin[:, None, None]
            + cell_half_y * abs_cos[:, None, None]
            - eps
        )

        per_box = overlap_x & overlap_y & overlap_u & overlap_v
        mask[image_index, 0] = per_box.any(dim=0)

    return mask
