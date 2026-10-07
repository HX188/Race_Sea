# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Training-time synthetic sea-fog augmentation for Race_Sea.

This module is intentionally independent from ``augment.py`` so that the feature
can be enabled/disabled with minimal changes to the baseline.

The augmentation is photometric only:
    I_fog(x, y) = I(x, y) * t(x, y) + A * (1 - t(x, y))

where ``t(x, y)`` is a smooth, spatially varying transmission map and ``A`` is
a near-neutral atmospheric-light color. Bounding boxes / OBB labels are not
modified.

No third-party dependency beyond packages already used by Ultralytics is added.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FogProfile:
    """Parameter ranges for one synthetic-fog severity level."""

    base_t: tuple[float, float]
    variation: tuple[float, float]
    airlight: tuple[float, float]


# Conservative first-stage settings.
# Heavy fog is intentionally not extreme because tiny OBB targets must remain
# visually learnable instead of becoming impossible positive samples.
FOG_PROFILES: dict[str, FogProfile] = {
    "light": FogProfile(base_t=(0.84, 0.93), variation=(0.04, 0.08), airlight=(220.0, 248.0)),
    "medium": FogProfile(base_t=(0.68, 0.84), variation=(0.07, 0.13), airlight=(224.0, 252.0)),
    "heavy": FogProfile(base_t=(0.50, 0.70), variation=(0.10, 0.18), airlight=(228.0, 255.0)),
}


def _smooth_random_field(
    height: int,
    width: int,
    rng,
) -> np.ndarray:
    """Return a smooth non-uniform field in approximately [-1, 1].

    Two low-resolution random fields are bicubically upsampled and blended,
    followed by Gaussian smoothing. This gives spatially coherent haze without
    introducing a Perlin-noise dependency.
    """
    # Coarse and fine grids scale with image dimensions while remaining valid
    # for small images.
    coarse_h, coarse_w = max(2, round(height / 160)), max(2, round(width / 160))
    fine_h, fine_w = max(3, round(height / 80)), max(3, round(width / 80))

    coarse = rng.uniform(0.0, 1.0, size=(coarse_h, coarse_w)).astype(np.float32)
    fine = rng.uniform(0.0, 1.0, size=(fine_h, fine_w)).astype(np.float32)

    coarse = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    fine = cv2.resize(fine, (width, height), interpolation=cv2.INTER_CUBIC)
    field = 0.72 * coarse + 0.28 * fine

    sigma = max(1.0, min(height, width) / 64.0)
    field = cv2.GaussianBlur(field, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT101)

    field = field - float(field.mean())
    denom = float(np.max(np.abs(field)))
    if denom < 1e-6:
        return np.zeros((height, width), dtype=np.float32)
    return np.clip(field / denom, -1.0, 1.0).astype(np.float32)


def synthesize_fog(
    image: np.ndarray,
    severity: str = "medium",
    rng=None,
) -> np.ndarray:
    """Apply spatially non-uniform atmospheric fog to an image.

    Args:
        image: HWC image. Race_Sea/Ultralytics training images are normally uint8 BGR.
        severity: One of ``light``, ``medium`` or ``heavy``.
        rng: NumPy RNG-like object exposing ``uniform``. If None, ``np.random``
            is used so Ultralytics worker seeding remains effective.

    Returns:
        Fogged image with the same shape and dtype as ``image``.
    """
    if severity not in FOG_PROFILES:
        raise ValueError(f"Unknown fog severity '{severity}', expected one of {tuple(FOG_PROFILES)}.")
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Fog augmentation expects an HWC 3-channel image, got shape={getattr(image, 'shape', None)}.")

    rng = np.random if rng is None else rng
    profile = FOG_PROFILES[severity]
    height, width = image.shape[:2]

    base_t = float(rng.uniform(*profile.base_t))
    variation = float(rng.uniform(*profile.variation))
    field = _smooth_random_field(height, width, rng)

    # Lower t means denser fog. Keep a safety floor so tiny targets are not
    # systematically erased by augmentation.
    transmission = np.clip(base_t + variation * field, 0.30, 0.98)[..., None]

    # Near-neutral atmospheric light with very small per-channel variation.
    air_base = float(rng.uniform(*profile.airlight))
    air = np.array(
        [air_base + float(rng.uniform(-3.0, 3.0)) for _ in range(3)],
        dtype=np.float32,
    )

    src = image.astype(np.float32)
    if np.issubdtype(image.dtype, np.integer):
        value_max = float(np.iinfo(image.dtype).max)
    else:
        value_max = 1.0 if float(np.nanmax(src)) <= 1.5 else 255.0

    air *= value_max / 255.0
    fogged = src * transmission + air.reshape(1, 1, 3) * (1.0 - transmission)
    fogged = np.clip(fogged, 0.0, value_max)

    if np.issubdtype(image.dtype, np.integer):
        fogged = np.rint(fogged)
    return fogged.astype(image.dtype, copy=False)


class RandomFog:
    """Apply non-uniform fog to a training sample with configurable probability.

    When ``enabled=False`` this transform returns immediately *without consuming
    RNG state*, which preserves baseline behavior/reproducibility as closely as
    possible.
    """

    def __init__(
        self,
        enabled: bool = False,
        p: float = 0.25,
        light_prob: float = 0.55,
        medium_prob: float = 0.35,
        heavy_prob: float = 0.10,
    ) -> None:
        self.enabled = bool(enabled)
        self.p = float(p)
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(f"fog_p must be in [0, 1], got {self.p}.")

        weights = np.asarray([light_prob, medium_prob, heavy_prob], dtype=np.float64)
        if np.any(weights < 0.0) or float(weights.sum()) <= 0.0:
            raise ValueError(f"Fog severity probabilities must be non-negative and sum to > 0, got {weights.tolist()}.")
        self.weights = (weights / weights.sum()).tolist()
        self.levels = ("light", "medium", "heavy")

    def __call__(self, labels: dict) -> dict:
        """Apply fog to ``labels['img']`` without changing geometry/annotations."""
        if not self.enabled or self.p <= 0.0:
            return labels
        if random.random() >= self.p:
            return labels

        severity = random.choices(self.levels, weights=self.weights, k=1)[0]
        labels["img"] = synthesize_fog(labels["img"], severity=severity)
        return labels

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(enabled={self.enabled}, p={self.p}, "
            f"weights={dict(zip(self.levels, self.weights))})"
        )
