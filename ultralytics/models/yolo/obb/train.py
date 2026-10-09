# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from pathlib import Path

from ultralytics.models import yolo
from ultralytics.nn.tasks import OBBModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK


class OBBTrainer(yolo.detect.DetectionTrainer):
    """A class extending the DetectionTrainer class for training based on an Oriented Bounding Box (OBB) model.

    This trainer specializes in training YOLO models that detect oriented bounding boxes, which are useful for detecting
    objects at arbitrary angles rather than just axis-aligned rectangles.

    Attributes:
        loss_names (tuple): Names of the loss components, derived from the loss dict returned by the criterion.

    Methods:
        get_model: Return OBBModel initialized with specified config and weights.
        get_validator: Return an instance of OBBValidator for validation of YOLO model.

    Examples:
        >>> from ultralytics.models.yolo.obb import OBBTrainer
        >>> args = dict(model="yolo26n-obb.pt", data="dota8.yaml", epochs=3)
        >>> trainer = OBBTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks: dict | None = None):
        """Initialize an OBBTrainer object for training Oriented Bounding Box (OBB) models.

        Args:
            cfg (dict, optional): Configuration dictionary for the trainer. Contains training parameters and model
                configuration.
            overrides (dict, optional): Dictionary of parameter overrides for the configuration. Any values here will
                take precedence over those in cfg.
            _callbacks (dict, optional): Dictionary of callback functions to be invoked during training.
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "obb"
        super().__init__(cfg, overrides, _callbacks)

    def preprocess_batch(self, batch: dict) -> dict:
        """Preprocess OBB batch and keep FFC clean pairs aligned with fog inputs."""
        batch = super().preprocess_batch(batch)
        if "fog_clean_img" in batch:
            clean = batch["fog_clean_img"].float() / 255.0
            if clean.shape[-2:] != batch["img"].shape[-2:]:
                clean = F.interpolate(
                    clean,
                    size=batch["img"].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            batch["fog_clean_img"] = clean
        return batch

    def get_model(
        self, cfg: str | dict | None = None, weights: str | Path | None = None, verbose: bool = True
    ) -> OBBModel:
        """Return OBBModel initialized with specified config and weights.

        Args:
            cfg (str | dict, optional): Model configuration. Can be a path to a YAML config file, a dictionary
                containing configuration parameters, or None to use default configuration.
            weights (str | Path, optional): Path to pretrained weights file. If None, random initialization is used.
            verbose (bool): Whether to display model information during initialization.

        Returns:
            (OBBModel): Initialized OBBModel with the specified configuration and weights.

        Examples:
            >>> trainer = OBBTrainer()
            >>> model = trainer.get_model(cfg="yolo26n-obb.yaml", weights="yolo26n-obb.pt")
        """
        model = self.set_model_names_for_load(
            OBBModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        )
        source_model = getattr(weights, "model", None)
        source_head = source_model[-1] if source_model is not None else None
        source_has_strip = bool(getattr(source_head, "strip_reg", False))
        source_has_hbs = bool(getattr(source_head, "hbs_enabled", False))
        source_hbs_all_levels = bool(getattr(source_head, "hbs_all_levels", False))
        if source_has_strip:
            model.model[-1].enable_reg_strip()
        if source_has_hbs:
            model.model[-1].enable_hbs(all_levels=source_hbs_all_levels)
        if weights:
            model.load(weights)
        if self.args.strip_reg and not source_has_strip:
            model.model[-1].enable_reg_strip()
        if self.args.hbs and (not source_has_hbs or self.args.hbs_all_levels != source_hbs_all_levels):
            model.model[-1].enable_hbs(all_levels=self.args.hbs_all_levels)
        elif not self.args.hbs and source_has_hbs:
            model.model[-1].hbs_enabled = False
            model.model[-1].hbs = None
        if self.args.fg_enhance and not self.args.hbs:
            raise ValueError(
                "fg_enhance=True requires hbs=True because it belongs "
                "to the HBS auxiliary path."
            )
        if self.args.fg_enhance:
            model.model[-1].enable_fg_enhance(
                gain=self.args.fg_enhance_gain,
                kernel_size=self.args.fg_enhance_kernel,
            )
        else:
            model.model[-1].disable_fg_enhance()
        if self.args.ffc and (
                not self.args.fog_aug
                or self.args.fog_p <= 0
        ):
            raise ValueError(
                "ffc=True requires fog_aug=True and fog_p > 0."
            )
        if self.args.strip_reg:
            LOGGER.info("Strip regression enabled for the OBB regression towers.")
        if self.args.hbs:
            head = model.model[-1]
            levels = "all detection levels" if head.hbs_all_levels else "P3 only"
            LOGGER.info(
                f"HBS enabled for OBB: training-only background smoothing on {levels} with kernels "
                f"{head.hbs_kernel_sizes} and an auxiliary one-to-many loss."
            )
        if self.args.fg_enhance:
            LOGGER.info(
                "Foreground detail enhancement enabled for OBB: "
                "training-only exact-OBB P3 enhancement "
                f"(gain={self.args.fg_enhance_gain}, "
                f"kernel={self.args.fg_enhance_kernel})."
            )
        if self.args.ffc:
            LOGGER.info(
                "FFC enabled for OBB: P3-only clean/fog invariance + "
                "local foreground/background separation "
                f"(gain={self.args.ffc_gain}, "
                f"margin={self.args.ffc_margin}, "
                f"bg_scale={self.args.ffc_bg_scale}, "
                f"warmup={self.args.ffc_warmup_epochs} epochs)."
            )

        return model

    def get_validator(self):
        """Return an instance of OBBValidator for validation of YOLO model."""
        return yolo.obb.OBBValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
