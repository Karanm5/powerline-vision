"""
inference.py
------------
Run trained PowerLine Vision model on new images, video, or a directory.
Renders bounding boxes and saves annotated outputs.

Usage:
    # Single image
    python scripts/inference.py --source path/to/image.jpg --weights results/best.pt

    # Directory of images
    python scripts/inference.py --source path/to/images/ --weights results/best.pt

    # Video file
    python scripts/inference.py --source path/to/video.mp4 --weights results/best.pt --video
"""

import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO
from tqdm import tqdm

# Add src to path for shared utilities
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from utils import load_image, save_image, draw_detections


SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def run_image_inference(
    model: YOLO,
    source_path: Path,
    output_dir: Path,
    conf: float,
    iou: float,
    save_annotated: bool,
) -> list[dict]:
    """
    Run detection on a single image and optionally save annotated output.

    Returns:
        List of detection dicts with keys: file, class_name, confidence, bbox
    """
    image = load_image(source_path)
    results = model.predict(
        source=str(source_path),
        conf=conf,
        iou=iou,
        verbose=False,
    )

    detections = []
    if results:
        r = results[0]
        boxes = r.boxes
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy().tolist()
            scores = boxes.conf.cpu().numpy().tolist()
            class_ids = boxes.cls.cpu().numpy().astype(int).tolist()

            for box, score, cls_id in zip(xyxy, scores, class_ids):
                detections.append({
                    "file": source_path.name,
                    "class_id": cls_id,
                    "class_name": model.names.get(cls_id, str(cls_id)),
                    "confidence": round(score, 4),
                    "bbox_xyxy": [round(v, 1) for v in box],
                })

            if save_annotated:
                annotated = draw_detections(image, xyxy, scores, class_ids, conf)
                out_path = output_dir / ("detected_" + source_path.name)
                save_image(annotated, out_path)

    return detections


def run_video_inference(
    model: YOLO,
    source_path: Path,
    output_dir: Path,
    conf: float,
    iou: float,
) -> None:
    """Run detection on each frame of a video and write annotated output."""
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {source_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = output_dir / ("detected_" + source_path.name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    print(f"Processing video: {source_path.name} ({total_frames} frames @ {fps:.1f} fps)")

    for _ in tqdm(range(total_frames), desc="Frames"):
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, conf=conf, iou=iou, verbose=False)
        if results and results[0].boxes is not None:
            r = results[0]
            boxes = r.boxes
            xyxy = boxes.xyxy.cpu().numpy().tolist()
            scores = boxes.conf.cpu().numpy().tolist()
            class_ids = boxes.cls.cpu().numpy().astype(int).tolist()
            frame = draw_detections(frame, xyxy, scores, class_ids, conf)
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Annotated video saved: {out_path}")


def main(args: argparse.Namespace) -> None:
    source = Path(args.source)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading weights: {args.weights}")
    model = YOLO(args.weights)

    if args.video or (source.is_file() and source.suffix.lower() in {".mp4", ".avi", ".mov"}):
        run_video_inference(model, source, output_dir, args.conf, args.iou)
        return

    # Collect image paths
    if source.is_dir():
        image_paths = sorted(
            p for p in source.iterdir()
            if p.suffix.lower() in SUPPORTED_IMAGE_FORMATS
        )
        print(f"Found {len(image_paths)} images in {source}")
    elif source.is_file():
        image_paths = [source]
    else:
        raise FileNotFoundError(f"Source not found: {source}")

    all_detections = []
    for img_path in tqdm(image_paths, desc="Detecting"):
        dets = run_image_inference(
            model, img_path, output_dir,
            args.conf, args.iou, not args.no_save
        )
        all_detections.extend(dets)

    # Summary
    print(f"\n── Inference Summary ────────────────────────────────────────────")
    print(f"  Images processed: {len(image_paths)}")
    print(f"  Total detections: {len(all_detections)}")
    if all_detections:
        from collections import Counter
        class_counts = Counter(d["class_name"] for d in all_detections)
        for cls, count in class_counts.items():
            print(f"  {cls}: {count}")
    print(f"  Annotated outputs: {output_dir}")
    print("─────────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PowerLine Vision — inference")
    parser.add_argument("--source", type=str, required=True,
                        help="Path to image, directory of images, or video file")
    parser.add_argument("--weights", type=str, default="results/weights/best.pt",
                        help="Path to trained model weights (.pt)")
    parser.add_argument("--output", type=str, default="results/inference",
                        help="Directory to save annotated outputs")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for detections")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="IoU threshold for NMS")
    parser.add_argument("--video", action="store_true",
                        help="Treat source as video")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not save annotated images to disk")
    main(parser.parse_args())
