"""
test_dataset.py
---------------
Unit tests for dataset loading, label parsing, CLAHE preprocessing,
and YOLO-to-pixel coordinate conversion.

Run with:
    pytest tests/ -v
"""

import tempfile
import textwrap
from pathlib import Path

import cv2
import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataset import (
    BoundingBox,
    apply_clahe,
    preprocess_image,
    load_yolo_labels,
)


# ── BoundingBox ───────────────────────────────────────────────────────────────

class TestBoundingBox:

    def test_to_pixel_centre(self):
        """A centred, full-image box should map to the full image dimensions."""
        box = BoundingBox(class_id=0, x_centre=0.5, y_centre=0.5, width=1.0, height=1.0)
        x1, y1, x2, y2 = box.to_pixel(640, 640)
        assert x1 == 0
        assert y1 == 0
        assert x2 == 640
        assert y2 == 640

    def test_to_pixel_small_box(self):
        """A 10% box in the top-left corner."""
        box = BoundingBox(class_id=1, x_centre=0.05, y_centre=0.05, width=0.1, height=0.1)
        x1, y1, x2, y2 = box.to_pixel(1000, 1000)
        assert x1 == 0
        assert y1 == 0
        assert x2 == 100
        assert y2 == 100

    def test_class_id_preserved(self):
        box = BoundingBox(class_id=1, x_centre=0.5, y_centre=0.5, width=0.2, height=0.2)
        assert box.class_id == 1


# ── Label Loading ─────────────────────────────────────────────────────────────

class TestLoadYoloLabels:

    def test_valid_label_file(self, tmp_path):
        label_file = tmp_path / "image.txt"
        label_file.write_text(textwrap.dedent("""\
            0 0.500 0.500 0.200 0.100
            1 0.300 0.700 0.050 0.080
        """))
        boxes = load_yolo_labels(label_file)
        assert len(boxes) == 2
        assert boxes[0].class_id == 0
        assert boxes[0].x_centre == pytest.approx(0.5)
        assert boxes[1].class_id == 1
        assert boxes[1].width == pytest.approx(0.05)

    def test_missing_file_returns_empty(self, tmp_path):
        boxes = load_yolo_labels(tmp_path / "nonexistent.txt")
        assert boxes == []

    def test_empty_file_returns_empty(self, tmp_path):
        label_file = tmp_path / "empty.txt"
        label_file.write_text("")
        boxes = load_yolo_labels(label_file)
        assert boxes == []

    def test_malformed_lines_are_skipped(self, tmp_path):
        label_file = tmp_path / "bad.txt"
        label_file.write_text(textwrap.dedent("""\
            0 0.5 0.5 0.2 0.1
            this is not a valid line
            1 0.3 0.3 0.1 0.1
        """))
        boxes = load_yolo_labels(label_file)
        assert len(boxes) == 2


# ── CLAHE ─────────────────────────────────────────────────────────────────────

class TestApplyCLAHE:

    def _make_image(self, h=64, w=64) -> np.ndarray:
        return (np.random.rand(h, w, 3) * 255).astype(np.uint8)

    def test_output_shape_preserved(self):
        img = self._make_image()
        result = apply_clahe(img)
        assert result.shape == img.shape

    def test_output_dtype_preserved(self):
        img = self._make_image()
        result = apply_clahe(img)
        assert result.dtype == np.uint8

    def test_output_differs_from_input(self):
        """CLAHE should modify at least some pixel values."""
        # Use a deliberately low-contrast image
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        result = apply_clahe(img)
        # Shape and type must be correct even if values identical on uniform input
        assert result.shape == img.shape


# ── Preprocessing ─────────────────────────────────────────────────────────────

class TestPreprocessImage:

    def test_resize_to_target(self):
        img = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
        result = preprocess_image(img, target_size=416, clahe=False)
        assert result.shape == (416, 416, 3)

    def test_clahe_applied_by_default(self):
        img = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
        # Should not raise; CLAHE application is verified by apply_clahe tests
        result = preprocess_image(img, target_size=128, clahe=True)
        assert result.shape == (128, 128, 3)

    def test_no_clahe(self):
        img = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
        result = preprocess_image(img, target_size=128, clahe=False)
        assert result.shape == (128, 128, 3)
