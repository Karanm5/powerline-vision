"""
train.py
--------
Training pipeline for the PowerLine Vision detection model.
Fine-tunes YOLOv8 on the TTPLA dataset with configurable hyperparameters.

Usage:
    python src/train.py --config configs/config.yaml
    python src/train.py --config configs/config.yaml --resume results/weights/last.pt
"""

import argparse
import yaml
import shutil
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO

from dataset import YOLODatasetValidator


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def validate_dataset(data_yaml: str, class_names: list[str]) -> None:
    """Run pre-training dataset validation and print summary statistics."""
    print("\n── Dataset Validation ──────────────────────────────────────────")
    validator = YOLODatasetValidator(data_yaml, class_names)
    stats = validator.validate_all()

    for split, s in stats.items():
        if s.images_without_labels > 0:
            pct = 100 * s.images_without_labels / max(s.num_images, 1)
            print(f"  Warning: {s.images_without_labels} images ({pct:.1f}%) "
                  f"in '{split}' have no annotations.")
    print("── Validation complete ─────────────────────────────────────────\n")


def setup_output_dir(results_dir: str) -> Path:
    """Create a timestamped run directory under results/."""
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(results_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ── Training ──────────────────────────────────────────────────────────────────

def train(config_path: str, resume_weights: str = None) -> None:
    cfg = load_config(config_path)

    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    aug_cfg = cfg["augmentation"]
    out_cfg = cfg["output"]

    data_yaml = str(Path(dataset_cfg["root"]) / "dataset.yaml")
    class_names = dataset_cfg["names"]

    # Validate dataset before starting
    validate_dataset(data_yaml, class_names)

    # Initialise model
    weights = resume_weights if resume_weights else model_cfg["weights"]
    print(f"Loading model: {weights}")
    model = YOLO(weights)

    run_dir = setup_output_dir(out_cfg["results_dir"])
    print(f"Run output directory: {run_dir}")

    # Save config snapshot alongside results for reproducibility
    shutil.copy(config_path, run_dir / "config.yaml")

    # Launch training
    print("\nStarting training ...")
    results = model.train(
        data=data_yaml,
        epochs=train_cfg["epochs"],
        imgsz=train_cfg["image_size"],
        batch=train_cfg["batch_size"],
        lr0=train_cfg["learning_rate"],
        lrf=train_cfg["lr_final"],
        warmup_epochs=train_cfg["warmup_epochs"],
        weight_decay=train_cfg["weight_decay"],
        momentum=train_cfg["momentum"],
        patience=train_cfg["patience"],
        save_period=train_cfg["save_period"],
        workers=train_cfg["workers"],
        device=train_cfg["device"],
        # Augmentation
        hsv_h=aug_cfg["hsv_h"],
        hsv_s=aug_cfg["hsv_s"],
        hsv_v=aug_cfg["hsv_v"],
        degrees=aug_cfg["degrees"],
        translate=aug_cfg["translate"],
        scale=aug_cfg["scale"],
        flipud=aug_cfg["flipud"],
        fliplr=aug_cfg["fliplr"],
        mosaic=aug_cfg["mosaic"],
        mixup=aug_cfg["mixup"],
        # Output
        project=str(run_dir),
        name="train",
        exist_ok=True,
        resume=resume_weights is not None,
        verbose=True,
    )

    print("\n── Training complete ────────────────────────────────────────────")
    print(f"  Best weights: {run_dir}/train/weights/best.pt")
    print(f"  Last weights: {run_dir}/train/weights/last.pt")
    print("────────────────────────────────────────────────────────────────\n")

    return results


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PowerLine Vision detection model")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to training configuration YAML"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint weights to resume training from"
    )
    args = parser.parse_args()
    train(args.config, args.resume)
