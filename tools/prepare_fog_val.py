#!/usr/bin/env python3
"""Generate deterministic light/medium/heavy fog validation sets for Race_Sea.

Read source data from:
    --source-root /mnt/data/wangzijian/train_data_sea

Write generated fog validation data to:
    --output-root /mnt/data/huangxin/datasets/train_data_sea

The source dataset is never modified.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics.data.fog_augment import FOG_PROFILES, synthesize_fog  # noqa: E402


DEFAULT_SOURCE_ROOT = Path("/mnt/data/wangzijian/train_data_sea")
DEFAULT_OUTPUT_ROOT = Path("/mnt/data/huangxin/datasets/train_data_sea")
LEVELS = ("light", "medium", "heavy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Existing prepared Race_Sea dataset. Read-only access is sufficient.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory for generated fog validation data.",
    )
    parser.add_argument("--val-list", type=str, default="sea_val20.txt")
    parser.add_argument("--base-yaml", type=str, default="race_dataset.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--levels", nargs="+", choices=LEVELS, default=list(LEVELS))
    parser.add_argument("--limit", type=int, default=0, help="Debug: only process first N validation images.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def read_image_list(path: Path) -> list[str]:
    items = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not items:
        raise RuntimeError(f"Empty validation list: {path}")
    return items


def resolve_source_image(source_root: Path, listed_path: str) -> Path:
    p = Path(listed_path)
    return p if p.is_absolute() else (source_root / p).resolve()


def source_encoding(path: Path) -> str:
    """Detect actual encoding because Race_Sea filenames may use misleading .jpg suffixes."""
    with path.open("rb") as f:
        header = f.read(16)
    if header.startswith(b"\xff\xd8"):
        return "jpeg"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if header[:4] in {b"II*\x00", b"MM\x00*"}:
        return "tiff"
    return "jpeg"


def write_encoded_image(path: Path, image: np.ndarray, encoding: str, jpeg_quality: int) -> None:
    if encoding == "jpeg":
        ext, params = ".jpg", [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    elif encoding == "png":
        ext, params = ".png", [cv2.IMWRITE_PNG_COMPRESSION, 3]
    elif encoding == "tiff":
        ext, params = ".tiff", []
    else:
        raise ValueError(encoding)

    ok, encoded = cv2.imencode(ext, image, params)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    path.write_bytes(encoded.tobytes())


def ensure_dir_symlink(link: Path, target: Path) -> None:
    """Create/update a directory symlink without touching a real directory."""
    target = target.resolve()

    if link.is_symlink():
        current = (link.parent / os.readlink(link)).resolve()
        if current == target:
            return
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"Refusing to replace non-symlink path: {link}")

    link.parent.mkdir(parents=True, exist_ok=True)
    relative_target = os.path.relpath(target, start=link.parent)
    link.symlink_to(relative_target, target_is_directory=True)


def make_source_entry_absolute(source_root: Path, value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [make_source_entry_absolute(source_root, x) for x in value]

    p = Path(str(value))
    return str(p if p.is_absolute() else (source_root / p).resolve())


def load_base_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "names" not in data:
        raise RuntimeError(f"Invalid dataset YAML: {path}")
    return data


def write_variant_yaml(
    base_data: dict,
    source_root: Path,
    output_root: Path,
    output_yaml: Path,
    val_list_name: str,
) -> None:
    """Generate a YAML that is valid even though source/output roots differ."""
    data = dict(base_data)
    data.pop("path", None)

    # train/test continue to refer to the original prepared dataset.
    if "train" in data:
        data["train"] = make_source_entry_absolute(source_root, data["train"])
    if "test" in data:
        data["test"] = make_source_entry_absolute(source_root, data["test"])

    # val points to the generated fog list in the user's own directory.
    data["val"] = str((output_root / val_list_name).resolve())

    output_yaml.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def profile_json(level: str) -> dict:
    p = FOG_PROFILES[level]
    return {"base_t": list(p.base_t), "variation": list(p.variation), "airlight": list(p.airlight)}


def main() -> None:
    args = parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    source_val_list = source_root / args.val_list
    source_yaml = source_root / args.base_yaml
    source_labels = source_root / "labels" / "val"

    for p in (source_val_list, source_yaml, source_labels):
        if not p.exists():
            raise FileNotFoundError(p)

    listed_images = read_image_list(source_val_list)
    if args.limit > 0:
        listed_images = listed_images[: args.limit]

    base_data = load_base_yaml(source_yaml)

    manifest = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "seed": args.seed,
        "source_val_list": args.val_list,
        "image_count": len(listed_images),
        "levels": {},
    }

    for level_index, level in enumerate(args.levels):
        image_dir = output_root / "images" / f"val_fog_{level}"
        label_link = output_root / "labels" / f"val_fog_{level}"
        image_dir.mkdir(parents=True, exist_ok=True)

        # Fog is photometric only, so reuse source OBB labels through a symlink.
        ensure_dir_symlink(label_link, source_labels)

        output_list = []
        for image_index, listed_path in enumerate(listed_images):
            src = resolve_source_image(source_root, listed_path)
            if not src.exists():
                raise FileNotFoundError(src)

            dst = image_dir / src.name
            if args.overwrite or not dst.exists():
                img = cv2.imread(str(src), cv2.IMREAD_COLOR)
                if img is None:
                    raise RuntimeError(f"OpenCV could not read: {src}")

                # Stable per (level, image) random state.
                seed_seq = np.random.SeedSequence([args.seed, level_index, image_index])
                rng = np.random.default_rng(seed_seq)
                fogged = synthesize_fog(img, severity=level, rng=rng)
                write_encoded_image(dst, fogged, source_encoding(src), args.jpeg_quality)

            output_list.append(f"./images/val_fog_{level}/{src.name}")

        list_name = f"sea_val_fog_{level}.txt"
        (output_root / list_name).write_text("\n".join(output_list) + "\n", encoding="utf-8")

        yaml_name = f"race_dataset_fog_{level}.yaml"
        write_variant_yaml(
            base_data=base_data,
            source_root=source_root,
            output_root=output_root,
            output_yaml=output_root / yaml_name,
            val_list_name=list_name,
        )

        manifest["levels"][level] = {
            "profile": profile_json(level),
            "image_dir": f"images/val_fog_{level}",
            "label_link": f"labels/val_fog_{level} -> {source_labels}",
            "list": list_name,
            "yaml": yaml_name,
        }

        print(f"[{level}] {len(output_list)} images -> {image_dir}")
        print(f"  yaml: {output_root / yaml_name}")

    manifest_path = output_root / "fog_val_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
