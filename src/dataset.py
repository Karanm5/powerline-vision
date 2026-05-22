"""
dataset.py
----------
Dataset loading, validation, and augmentation utilities for the
PowerLine Vision detection pipeline.

Handles YOLO-format datasets with optional OpenCV-based preprocessing
(CLAHE contrast enhancement) and Albumentations augmentation.
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class DatasetStats:
    """Summary statistics for a YOLO-format dataset split."""
    split: str
    num_images: int
    num_labels: int
    class_counts: dict
    images_without_labels: int
    mean_boxes_per_image: float


@dataclass
class BoundingBox:
    """Normalised YOLO-format bounding box."""
    class_id: int
    x_centre: float
    y_centre: float
    width: float
    height: float

    def to_pixel(self, img_w: int, img_h: int) -> tuple[int, int, int, int]:
        """Convert to pixel-space (x1, y1, x2, y2)."""
        x1 = int((self.x_centre - self.width / 2) * img_w)
        y1 = int((self.y_centre - self.height / 2) * img_h)
        x2 = int((self.x_centre + self.width / 2) * img_w)
        y2 = int((self.y_centre + self.height / 2) * img_h)
        return x1, y1, x2, y2


# ── Dataset Validation ───────────────────────────────────────────────────────

class YOLODatasetValidator:
    """
    Validates a YOLO-format dataset directory and returns statistics.
    Checks for missing label files, empty annotations, and malformed entries.
    """

    def __init__(self, data_yaml: str, class_names: Optional[list[str]] = None):
        with open(data_yaml) as f:
            cfg = yaml.safe_load(f)
        self.root = Path(cfg.get("path", "."))
        self.splits = {
            "train": self.root / cfg["train"],
            "val": self.root / cfg["val"],
            "test": self.root / cfg.get("test", cfg["val"]),
        }
        self.class_names = class_names or cfg.get("names", [])
        self.nc = cfg.get("nc", len(self.class_names))

    def validate_split(self, split: str) -> DatasetStats:
        img_dir = self.splits[split]
        label_dir = img_dir.parent.parent / "labels" / img_dir.name

        if not img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

        image_paths = sorted(
            p for p in img_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )

        class_counts = {i: 0 for i in range(self.nc)}
        num_labels = 0
        images_without_labels = 0

        for img_path in image_paths:
            label_path = label_dir / (img_path.stem + ".txt")
            if not label_path.exists() or label_path.stat().st_size == 0:
                images_without_labels += 1
                continue
            with open(label_path) as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            num_labels += len(lines)
            for line in lines:
                parts = line.split()
                if len(parts) == 5:
                    cls = int(parts[0])
                    class_counts[cls] = class_counts.get(cls, 0) + 1

        mean_boxes = num_labels / max(len(image_paths) - images_without_labels, 1)

        return DatasetStats(
            split=split,
            num_images=len(image_paths),
            num_labels=num_labels,
            class_counts={
                self.class_names[k] if k < len(self.class_names) else str(k): v
                for k, v in class_counts.items()
            },
            images_without_labels=images_without_labels,
            mean_boxes_per_image=round(mean_boxes, 2),
        )

    def validate_all(self) -> dict[str, DatasetStats]:
        results = {}
        for split in self.splits:
            try:
                results[split] = self.validate_split(split)
                s = results[split]
                print(
                    f"[{split:5s}] images={s.num_images:4d}  "
                    f"labels={s.num_labels:5d}  "
                    f"no_label={s.images_without_labels:3d}  "
                    f"mean_boxes={s.mean_boxes_per_image:.2f}"
                )
            except FileNotFoundError as e:
                print(f"[{split}] SKIPPED — {e}")
        return results


# ── Preprocessing ─────────────────────────────────────────────────────────────

def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalisation (CLAHE).

    Particularly useful for aerial imagery where lighting is uneven or
    conductors are low-contrast against sky or vegetation backgrounds.

    Args:
        image:      BGR image as numpy array (H, W, 3)
        clip_limit: Threshold for contrast limiting (higher = more contrast)
        tile_size:  Size of grid tiles for local equalisation

    Returns:
        CLAHE-enhanced BGR image
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge([l_channel, a_channel, b_channel])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preprocess_image(image: np.ndarray, target_size: int = 640, clahe: bool = True) -> np.ndarray:
    """
    Full preprocessing pipeline for a single aerial image.

    Steps:
        1. Optional CLAHE contrast enhancement
        2. Resize to target_size x target_size (letterbox-style via OpenCV)

    Args:
        image:       BGR image as numpy array
        target_size: Output spatial dimension (square)
        clahe:       Whether to apply CLAHE enhancement

    Returns:
        Preprocessed BGR image
    """
    if clahe:
        image = apply_clahe(image)
    image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return image


# ── Augmentation Pipeline ────────────────────────────────────────────────────

def get_train_transforms(image_size: int = 640) -> A.Compose:
    """
    Albumentations augmentation pipeline for training.
    Designed for aerial imagery with power line targets.
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Rotate(limit=15, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.MotionBlur(blur_limit=5, p=0.1),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.1),
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )


def get_val_transforms(image_size: int = 640) -> A.Compose:
    """Minimal augmentation pipeline for validation and test sets."""
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )


# ── Label Loading ─────────────────────────────────────────────────────────────

def load_yolo_labels(label_path: Path) -> list[BoundingBox]:
    """
    Parse a YOLO .txt label file into a list of BoundingBox objects.

    Args:
        label_path: Path to .txt label file

    Returns:
        List of BoundingBox; empty list if file is missing or empty
    """
    if not label_path.exists():
        return []
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                boxes.append(BoundingBox(
                    class_id=int(parts[0]),
                    x_centre=float(parts[1]),
                    y_centre=float(parts[2]),
                    width=float(parts[3]),
                    height=float(parts[4]),
                ))
    return boxes


if __name__ == "__main__":
    # Quick validation check
    validator = YOLODatasetValidator("data/processed/dataset.yaml")
    validator.validate_all()
