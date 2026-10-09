"""Frequency-domain image augmentation for Sea Race training.

This module contains the validated V1 strategy used by the team: randomly jitter
amplitude inside a centered low-frequency disk while preserving the original
phase and all detection annotations.

The training configuration is intentionally kept outside this module. Recommended
starting values:
    freq_aug=True
    freq_aug_prob=0.5
    freq_strength=0.1
    freq_radius=0.1
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np


class RandomFrequencyStyle:
    """Randomize low-frequency amplitude while keeping image geometry unchanged.

    The input image is transformed with a 2D FFT over the spatial dimensions.
    Inside a centered low-frequency disk, each amplitude bin is multiplied by a
    random factor sampled from ``Uniform(1 - strength, 1 + strength)``. The
    original phase is reused for the inverse FFT.

    This is an image-only augmentation: OBBs/classes/instances are not modified.
    """

    def __init__(self, p: float = 0.5, radius: float = 0.1, strength: float = 0.1) -> None:
        """Initialize V1 frequency-style augmentation.

        Args:
            p: Probability of applying the transform, in [0, 1].
            radius: Low-frequency disk radius as a fraction of the Nyquist radius,
                in [0, 1].
            strength: Multiplicative amplitude jitter half-width, in [0, 1].
                For example, 0.1 samples factors from [0.9, 1.1].
        """
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"freq_aug_prob={p} must be in [0, 1].")
        if not 0.0 <= radius <= 1.0:
            raise ValueError(f"freq_radius={radius} must be in [0, 1].")
        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"freq_strength={strength} must be in [0, 1].")

        self.p = float(p)
        self.radius = float(radius)
        self.strength = float(strength)

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        """Apply the image-only transform to an Ultralytics labels dictionary."""
        if self.p <= 0.0 or self.radius <= 0.0 or self.strength <= 0.0 or random.random() > self.p:
            return labels

        img = labels.get("img")
        if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] != 3:
            return labels

        labels["img"] = self._randomize_amplitude(img)
        return labels

    def _randomize_amplitude(self, img: np.ndarray) -> np.ndarray:
        """Return an image with V1 low-frequency amplitude jitter applied."""
        original_dtype = img.dtype
        spatial = img.astype(np.float32, copy=False)

        spectrum = np.fft.fftshift(np.fft.fft2(spatial, axes=(0, 1)), axes=(0, 1))
        amplitude = np.abs(spectrum)
        phase = np.angle(spectrum)

        height, width = spatial.shape[:2]
        center_y, center_x = height // 2, width // 2
        yy, xx = np.ogrid[:height, :width]
        nyquist = 0.5 * min(height, width)
        disk = (yy - center_y) ** 2 + (xx - center_x) ** 2 <= (self.radius * nyquist) ** 2

        scale = np.random.uniform(
            1.0 - self.strength,
            1.0 + self.strength,
            size=amplitude.shape,
        ).astype(np.float32)
        amplitude = np.where(disk[..., None], amplitude * scale, amplitude)

        randomized = np.fft.ifft2(
            np.fft.ifftshift(amplitude * np.exp(1j * phase), axes=(0, 1)),
            axes=(0, 1),
        ).real

        # Numerical safety only; under normal finite inputs this branch is never taken.
        if not np.isfinite(randomized).all():
            return img

        randomized = np.clip(randomized, 0, 255)
        if original_dtype == np.uint8:
            return np.rint(randomized).astype(np.uint8)
        return randomized.astype(original_dtype, copy=False)
