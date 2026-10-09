# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Fog Feature Contrast (FFC) helpers for Race_Sea.

FFC is training-only and operates on the original shared P3 feature before HBS
or foreground enhancement.

For each GT object in a fogged training image:
    1) invariance:
       fog foreground feature should stay close to the corresponding clean
       foreground feature (clean branch is stop-gradient);
    2) separation:
       fog foreground feature should stay dissimilar from its local fog
       background ring.

Foreground:
    1.0x exact OBB.

Local background:
    2.0x OBB minus every 1.0x GT OBB in the same image.

All masks are rasterized at P3 using feature-cell centers. Tiny foreground OBBs
are guaranteed at least one feature cell.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F


@contextmanager
def _temporary_feature_eval(model: torch.nn.Module):
    """Temporarily put all pre-head modules in eval mode, then restore states.

    This prevents the clean stop-gradient branch from updating BatchNorm running
    statistics a second time in the same training iteration and also removes any
    stochastic train/eval behavior from the clean target branch.
    """
    modules = []
    states = []
    seen = set()

    for layer in list(model.model)[:-1]:
        for module in layer.modules():
            key = id(module)
            if key in seen:
                continue
            seen.add(key)
            modules.append(module)
            states.append(module.training)
            module.training = False

    try:
        yield
    finally:
        for module, training in zip(modules, states):
            module.training = training


@torch.no_grad()
def extract_p3_feature(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Run only backbone+neck and return the first feature consumed by the head.

    The first head input is P3 for YOLO26-OBB. The detection head itself is not
    executed, so the clean branch is cheaper than a full second forward.
    """
    layers = model.model
    head = layers[-1]

    x = images
    y = []

    with _temporary_feature_eval(model):
        for module in layers[:-1]:
            if module.f != -1:
                x = (
                    y[module.f]
                    if isinstance(module.f, int)
                    else [x if j == -1 else y[j] for j in module.f]
                )
            x = module(x)
            y.append(x if module.i in model.save else None)

        if isinstance(head.f, int):
            head_inputs = [x if head.f == -1 else y[head.f]]
        else:
            head_inputs = [x if j == -1 else y[j] for j in head.f]

    if not head_inputs or not isinstance(head_inputs[0], torch.Tensor):
        raise RuntimeError("Could not extract P3 feature from the clean FFC branch.")

    return head_inputs[0].detach()


def _obb_center_masks(
    boxes: torch.Tensor,
    height: int,
    width: int,
    scale: float = 1.0,
    ensure_one_cell: bool = False,
) -> torch.Tensor:
    """Rasterize normalized OBBs by feature-cell centers.

    Args:
        boxes: (N, 5) normalized (cx, cy, w, h, angle[radian]).
        height: Feature-map height.
        width: Feature-map width.
        scale: Multiplicative scale applied to width and height.
        ensure_one_cell: Ensure each valid OBB owns at least one cell.

    Returns:
        Boolean masks shaped (N, H, W).
    """
    n = boxes.shape[0]
    device = boxes.device

    if n == 0:
        return torch.zeros((0, height, width), dtype=torch.bool, device=device)

    work = boxes.float()
    xs = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
    ys = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    centers = work[:, :2]
    sizes = work[:, 2:4].clamp_min(0) * float(scale)
    angle = work[:, 4] if work.shape[1] >= 5 else torch.zeros(n, device=device)

    cos_a = angle.cos()[:, None, None]
    sin_a = angle.sin()[:, None, None]

    dx = xx[None] - centers[:, 0, None, None]
    dy = yy[None] - centers[:, 1, None, None]

    local_x = dx * cos_a + dy * sin_a
    local_y = -dx * sin_a + dy * cos_a

    masks = (
        (local_x.abs() <= sizes[:, 0, None, None] * 0.5)
        & (local_y.abs() <= sizes[:, 1, None, None] * 0.5)
    )

    if ensure_one_cell:
        missing = ~masks.flatten(1).any(1)
        if missing.any():
            for box_id in torch.nonzero(missing, as_tuple=False).flatten().tolist():
                cx = centers[box_id, 0].clamp(0.0, 1.0 - 1e-7)
                cy = centers[box_id, 1].clamp(0.0, 1.0 - 1e-7)
                ix = int(torch.clamp((cx * width).floor(), 0, width - 1).item())
                iy = int(torch.clamp((cy * height).floor(), 0, height - 1).item())
                masks[box_id, iy, ix] = True

    return masks


def _masked_mean(feature: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average one CHW feature inside one HW boolean mask."""
    weight = mask.to(dtype=feature.dtype)
    denom = weight.sum().clamp_min(1.0)
    return (feature * weight.unsqueeze(0)).sum(dim=(-2, -1)) / denom


class FogFeatureContrastLoss:
    """Compute FFC invariance and local foreground/background separation losses."""

    def __init__(self, margin: float = 0.20, bg_scale: float = 2.0) -> None:
        if not -1.0 <= margin <= 1.0:
            raise ValueError(f"ffc_margin must be in [-1, 1], got {margin}.")
        if bg_scale <= 1.0:
            raise ValueError(f"ffc_bg_scale must be > 1.0, got {bg_scale}.")

        self.margin = float(margin)
        self.bg_scale = float(bg_scale)

    def __call__(
        self,
        fog_p3: torch.Tensor,
        clean_p3: torch.Tensor,
        batch: dict[str, torch.Tensor],
        pair_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (invariance_loss, separation_loss), each mean-reduced.

        ``clean_p3`` is expected to be stop-gradient and ordered consistently
        with ``pair_idx``.
        """
        if fog_p3.ndim != 4 or clean_p3.ndim != 4:
            raise ValueError("FFC expects BCHW P3 feature tensors.")
        if pair_idx.numel() != clean_p3.shape[0]:
            raise ValueError(
                f"pair_idx/clean batch mismatch: {pair_idx.numel()} vs {clean_p3.shape[0]}"
            )
        if fog_p3.shape[-2:] != clean_p3.shape[-2:]:
            raise ValueError(
                f"fog/clean P3 spatial mismatch: {fog_p3.shape[-2:]} vs {clean_p3.shape[-2:]}"
            )

        boxes = batch["bboxes"].to(device=fog_p3.device, dtype=torch.float32)
        batch_idx = batch["batch_idx"].to(device=fog_p3.device, dtype=torch.long).view(-1)
        pair_idx = pair_idx.to(device=fog_p3.device, dtype=torch.long).view(-1)

        height, width = fog_p3.shape[-2:]
        zero = fog_p3.sum() * 0.0

        inv_terms = []
        sep_terms = []

        for clean_i, fog_image_i_tensor in enumerate(pair_idx):
            fog_image_i = int(fog_image_i_tensor.item())
            image_boxes = boxes[batch_idx == fog_image_i]
            if image_boxes.numel() == 0:
                continue

            fg_masks = _obb_center_masks(
                image_boxes,
                height,
                width,
                scale=1.0,
                ensure_one_cell=True,
            )
            outer_masks = _obb_center_masks(
                image_boxes,
                height,
                width,
                scale=self.bg_scale,
                ensure_one_cell=True,
            )
            outer_masks = outer_masks | fg_masks

            # Remove every GT object from every local background ring.
            all_gt = fg_masks.any(dim=0)
            bg_masks = outer_masks & (~all_gt.unsqueeze(0))

            fog_feature = fog_p3[fog_image_i].float()
            clean_feature = clean_p3[clean_i].float().detach()

            for target_i in range(image_boxes.shape[0]):
                fg_mask = fg_masks[target_i]

                fog_fg = _masked_mean(fog_feature, fg_mask)
                clean_fg = _masked_mean(clean_feature, fg_mask)

                inv = 1.0 - F.cosine_similarity(
                    fog_fg.unsqueeze(0),
                    clean_fg.unsqueeze(0),
                    dim=1,
                    eps=1e-6,
                ).squeeze(0)
                inv_terms.append(inv)

                bg_mask = bg_masks[target_i]
                if bg_mask.any():
                    fog_bg = _masked_mean(fog_feature, bg_mask)
                    similarity = F.cosine_similarity(
                        fog_fg.unsqueeze(0),
                        fog_bg.unsqueeze(0),
                        dim=1,
                        eps=1e-6,
                    ).squeeze(0)
                    sep_terms.append(F.relu(similarity - self.margin))

        inv_loss = torch.stack(inv_terms).mean() if inv_terms else zero
        sep_loss = torch.stack(sep_terms).mean() if sep_terms else zero
        return inv_loss, sep_loss
