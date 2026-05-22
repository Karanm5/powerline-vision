"""
download_data.py
----------------
Downloads the TTPLA dataset (Transmission Towers and Power Lines from Aerial imagery)
and converts COCO-format annotations to YOLO format for training.

Usage:
    python data/download_data.py
"""

import os
import json
import shutil
import argparse
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm


# ── Configuration ────────────────────────────────────────────────────────────

TTPLA_URL = "https://github.com/R3ab/ttpla_dataset/archive/refs/heads/master.zip"
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
SPLITS = ["train", "val", "test"]
CLASS_NAMES = ["power_line", "tower"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def download_file(url: str, dest: Path) -> None:
    """Stream-download a file with a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        desc=dest.name, total=total, unit="B", unit_scale=True
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def extract_zip(zip_path: Path, dest: Path) -> None:
    """Extract a zip archive."""
    print(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)


def coco_to_yolo(
    coco_json_path: Path,
    image_dir: Path,
    output_dir: Path,
    split: str
) -> None:
    """
    Convert COCO annotation format to YOLO format.

    YOLO label format per line:
        <class_id> <x_centre> <y_centre> <width> <height>
    All values are normalised to [0, 1] relative to image dimensions.
    """
    with open(coco_json_path) as f:
        coco = json.load(f)

    # Build lookup maps
    images = {img["id"]: img for img in coco["images"]}
    category_map = {cat["id"]: idx for idx, cat in enumerate(coco["categories"])}

    # Group annotations by image ID
    annotations_by_image: dict[int, list] = {}
    for ann in coco["annotations"]:
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    label_dir = output_dir / split / "labels"
    img_out_dir = output_dir / split / "images"
    label_dir.mkdir(parents=True, exist_ok=True)
    img_out_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    for img_id, img_info in tqdm(images.items(), desc=f"Converting {split}"):
        img_w = img_info["width"]
        img_h = img_info["height"]
        file_name = img_info["file_name"]

        # Copy image to output directory
        src_img = image_dir / file_name
        if src_img.exists():
            shutil.copy(src_img, img_out_dir / Path(file_name).name)

        # Write YOLO label file
        anns = annotations_by_image.get(img_id, [])
        label_path = label_dir / (Path(file_name).stem + ".txt")
        with open(label_path, "w") as lf:
            for ann in anns:
                cat_id = category_map.get(ann["category_id"])
                if cat_id is None:
                    continue
                x, y, w, h = ann["bbox"]  # COCO: top-left x, y, width, height
                # Convert to YOLO centre-normalised format
                x_c = (x + w / 2) / img_w
                y_c = (y + h / 2) / img_h
                w_n = w / img_w
                h_n = h / img_h
                lf.write(f"{cat_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")
        converted += 1

    print(f"  {split}: {converted} images converted.")


def write_dataset_yaml(output_dir: Path, class_names: list[str]) -> None:
    """Write the dataset YAML file expected by Ultralytics YOLO."""
    yaml_content = f"""# TTPLA Dataset — YOLO format
path: {output_dir.resolve()}
train: train/images
val: val/images
test: test/images

nc: {len(class_names)}
names: {class_names}
"""
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"Dataset YAML written to {yaml_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = RAW_DIR / "ttpla_dataset.zip"

    # Download
    if not zip_path.exists() or args.force:
        print("Downloading TTPLA dataset ...")
        download_file(TTPLA_URL, zip_path)
    else:
        print(f"Archive already exists at {zip_path}. Use --force to re-download.")

    # Extract
    extract_dir = RAW_DIR / "ttpla_dataset-master"
    if not extract_dir.exists() or args.force:
        extract_zip(zip_path, RAW_DIR)

    # Convert annotations for each split
    for split in SPLITS:
        json_path = extract_dir / "annotations" / f"{split}.json"
        image_dir = extract_dir / "images" / split
        if json_path.exists():
            coco_to_yolo(json_path, image_dir, PROCESSED_DIR, split)
        else:
            print(f"Warning: annotation file not found for split '{split}': {json_path}")

    write_dataset_yaml(PROCESSED_DIR, CLASS_NAMES)
    print("\nDataset preparation complete.")
    print(f"Processed data available at: {PROCESSED_DIR.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare TTPLA dataset")
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist")
    main(parser.parse_args())
