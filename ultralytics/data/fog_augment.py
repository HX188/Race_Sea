# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Training-time synthetic sea-fog augmentation for Race_Sea.

The same ``synthesize_fog()`` function is used by training and
``tools/prepare_fog_val.py``.

When FFC is enabled, ``RandomFog`` stores the clean image only for samples that
are actually selected for fog augmentation. The clean image is captured after
all preceding geometry/color/flip augmentation and immediately before fog, so
clean/fog pairs have identical OBB geometry.
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
    blur_sigma_ref: tuple[float, float]
    blur_mix: tuple[float, float]
    min_t: float


FOG_PROFILES: dict[str, FogProfile] = {
    "light": FogProfile(
        base_t=(0.47, 0.65),
        variation=(0.10, 0.19),
        airlight=(220.0, 248.0),
        blur_sigma_ref=(2.10, 3.15),
        blur_mix=(0.62, 0.88),
        min_t=0.23,
    ),
    "medium": FogProfile(
        base_t=(0.425, 0.61),
        variation=(0.108, 0.203),
        airlight=(224.0, 252.0),
        blur_sigma_ref=(2.35, 3.50),
        blur_mix=(0.665, 0.935),
        min_t=0.20,
    ),
    "heavy": FogProfile(
        base_t=(0.39, 0.58),
        variation=(0.115, 0.212),
        airlight=(228.0, 255.0),
        blur_sigma_ref=(2.55, 3.75),
        blur_mix=(0.69, 0.97),
        min_t=0.18,
    ),
}


def _smooth_random_field(height: int, width: int, rng) -> np.ndarray:
    """Generate a smooth non-uniform field in approximately [-1, 1]."""
    coarse_h, coarse_w = max(2, round(height / 160)), max(2, round(width / 160))
    fine_h, fine_w = max(3, round(height / 80)), max(3, round(width / 80))

    coarse = rng.uniform(0.0, 1.0, size=(coarse_h, coarse_w)).astype(np.float32)
    fine = rng.uniform(0.0, 1.0, size=(fine_h, fine_w)).astype(np.float32)

    coarse = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    fine = cv2.resize(fine, (width, height), interpolation=cv2.INTER_CUBIC)

    field = 0.72 * coarse + 0.28 * fine
    sigma = max(1.0, min(height, width) / 64.0)
    field = cv2.GaussianBlur(
        field,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT101,
    )

    field -= float(field.mean())
    denom = float(np.max(np.abs(field)))
    if denom < 1e-6:
        return np.zeros((height, width), dtype=np.float32)

    return np.clip(field / denom, -1.0, 1.0).astype(np.float32)


def _density_aware_blur(
    image: np.ndarray,
    transmission: np.ndarray,
    sigma_ref: float,
    blur_mix: float,
) -> np.ndarray:
    """Blur denser-fog regions more strongly while preserving thinner-fog regions."""
    height, width = image.shape[:2]

    resolution_scale = max(height, width) / 640.0
    sigma = max(0.01, float(sigma_ref) * resolution_scale)

    src = image.astype(np.float32)
    blurred = cv2.GaussianBlur(
        src,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT101,
    )

    density = 1.0 - transmission
    local_strength = np.clip(density / 0.55, 0.0, 1.0)
    local_strength = np.power(local_strength, 0.85)
    local_strength = (local_strength * float(blur_mix))[..., None]

    return src * (1.0 - local_strength) + blurred * local_strength


def synthesize_fog(
    image: np.ndarray,
    severity: str = "medium",
    rng=None,
) -> np.ndarray:
    """Apply spatially non-uniform sea fog with density-aware edge attenuation."""
    if severity not in FOG_PROFILES:
        raise ValueError(
            f"Unknown fog severity '{severity}', expected one of {tuple(FOG_PROFILES)}."
        )
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "Fog augmentation expects an HWC 3-channel image, "
            f"got shape={getattr(image, 'shape', None)}."
        )

    rng = np.random if rng is None else rng
    profile = FOG_PROFILES[severity]
    height, width = image.shape[:2]

    base_t = float(rng.uniform(*profile.base_t))
    variation = float(rng.uniform(*profile.variation))
    field = _smooth_random_field(height, width, rng)

    transmission = np.clip(
        base_t + variation * field,
        profile.min_t,
        0.98,
    ).astype(np.float32)

    sigma_ref = float(rng.uniform(*profile.blur_sigma_ref))
    blur_mix = float(rng.uniform(*profile.blur_mix))
    scene = _density_aware_blur(
        image=image,
        transmission=transmission,
        sigma_ref=sigma_ref,
        blur_mix=blur_mix,
    )

    air_base = float(rng.uniform(*profile.airlight))
    air = np.array(
        [air_base + float(rng.uniform(-3.0, 3.0)) for _ in range(3)],
        dtype=np.float32,
    )

    if np.issubdtype(image.dtype, np.integer):
        value_max = float(np.iinfo(image.dtype).max)
    else:
        value_max = 1.0 if float(np.nanmax(scene)) <= 1.5 else 255.0

    air *= value_max / 255.0

    fogged = (
        scene * transmission[..., None]
        + air.reshape(1, 1, 3) * (1.0 - transmission[..., None])
    )
    fogged = np.clip(fogged, 0.0, value_max)

    if np.issubdtype(image.dtype, np.integer):
        fogged = np.rint(fogged)

    return fogged.astype(image.dtype, copy=False)


class RandomFog:
    """Apply non-uniform synthetic fog to a training sample."""

    def __init__(
        self,
        enabled: bool = False,
        p: float = 0.25,
        light_prob: float = 0.20,
        medium_prob: float = 0.30,
        heavy_prob: float = 0.50,
        save_clean: bool = False,
    ) -> None:
        self.enabled = bool(enabled)
        self.p = float(p)
        self.save_clean = bool(save_clean)

        if not 0.0 <= self.p <= 1.0:
            raise ValueError(f"fog_p must be in [0, 1], got {self.p}.")

        weights = np.asarray(
            [light_prob, medium_prob, heavy_prob],
            dtype=np.float64,
        )
        if np.any(weights < 0.0) or float(weights.sum()) <= 0.0:
            raise ValueError(
                "Fog severity probabilities must be non-negative and sum to > 0, "
                f"got {weights.tolist()}."
            )

        self.weights = (weights / weights.sum()).tolist()
        self.levels = ("light", "medium", "heavy")

    def __call__(self, labels: dict) -> dict:
        """Apply fog without changing OBB geometry."""
        if not self.enabled or self.p <= 0.0:
            return labels

        if random.random() >= self.p:
            return labels

        severity = random.choices(
            self.levels,
            weights=self.weights,
            k=1,
        )[0]

        if self.save_clean:
            labels["fog_clean_img"] = labels["img"].copy()

        labels["img"] = synthesize_fog(
            labels["img"],
            severity=severity,
        )
        return labels

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(enabled={self.enabled}, p={self.p}, "
            f"save_clean={self.save_clean}, weights={dict(zip(self.levels, self.weights))})"
        )
