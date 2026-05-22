"""
evaluate.py
-----------
Post-training evaluation for the PowerLine Vision detection model.
Computes mAP@0.5, mAP@0.5:0.95, per-class precision/recall/F1,
and generates precision-recall curves and a confusion matrix.

Usage:
    python src/evaluate.py --weights results/best.pt --config configs/config.yaml
"""

import argparse
import yaml
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def plot_precision_recall_curve(
    results,
    class_names: list[str],
    output_dir: Path
) -> None:
    """Plot per-class and mean precision-recall curves."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colours = plt.cm.tab10(np.linspace(0, 1, len(class_names)))

    # Ultralytics stores PR curve data in results.curves
    if hasattr(results, "curves") and results.curves:
        for i, (name, colour) in enumerate(zip(class_names, colours)):
            # Extract per-class curve if available
            try:
                px = results.curves[0]  # recall axis
                py = results.curves[1][i]  # precision per class
                ax.plot(px, py, color=colour, linewidth=2, label=f"{name}")
            except (IndexError, TypeError):
                pass

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve — PowerLine Detection", fontsize=13)
    ax.legend(loc="lower left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    out_path = output_dir / "precision_recall_curve.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  PR curve saved: {out_path}")


def plot_confusion_matrix(
    results,
    class_names: list[str],
    output_dir: Path
) -> None:
    """Render and save the normalised confusion matrix."""
    if not hasattr(results, "confusion_matrix") or results.confusion_matrix is None:
        print("  Confusion matrix not available in results object.")
        return

    matrix = results.confusion_matrix.matrix
    # Normalise rows to [0, 1]
    row_sums = matrix.sum(axis=1, keepdims=True)
    norm_matrix = np.divide(matrix, row_sums, where=row_sums != 0)

    labels = class_names + ["Background"]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        norm_matrix,
        annot=True, fmt=".2f",
        xticklabels=labels, yticklabels=labels,
        cmap="Blues", ax=ax, vmin=0, vmax=1,
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title("Normalised Confusion Matrix", fontsize=12)

    out_path = output_dir / "confusion_matrix.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved: {out_path}")


def save_metrics_json(metrics: dict, output_dir: Path) -> None:
    """Persist evaluation metrics to JSON for downstream reporting."""
    out_path = output_dir / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics JSON saved: {out_path}")


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(weights_path: str, config_path: str) -> dict:
    cfg = load_config(config_path)
    dataset_cfg = cfg["dataset"]
    eval_cfg = cfg["evaluation"]
    out_cfg = cfg["output"]

    data_yaml = str(Path(dataset_cfg["root"]) / "dataset.yaml")
    class_names = dataset_cfg["names"]

    output_dir = Path(out_cfg["plots_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n── Evaluation ───────────────────────────────────────────────────")
    print(f"  Weights:   {weights_path}")
    print(f"  Data YAML: {data_yaml}")
    print(f"  IoU threshold:  {eval_cfg['iou_threshold']}")
    print(f"  Conf threshold: {eval_cfg['conf_threshold']}")
    print("─────────────────────────────────────────────────────────────────\n")

    model = YOLO(weights_path)

    results = model.val(
        data=data_yaml,
        iou=eval_cfg["iou_threshold"],
        conf=eval_cfg["conf_threshold"],
        max_det=eval_cfg["max_detections"],
        verbose=True,
        plots=True,
        save_json=True,
    )

    # Extract key metrics
    metrics = {
        "mAP_50":     round(float(results.box.map50), 4),
        "mAP_50_95":  round(float(results.box.map), 4),
        "precision":  round(float(results.box.mp), 4),
        "recall":     round(float(results.box.mr), 4),
        "per_class": {}
    }

    for i, name in enumerate(class_names):
        metrics["per_class"][name] = {
            "AP_50": round(float(results.box.ap50[i]), 4) if i < len(results.box.ap50) else None,
            "AP_50_95": round(float(results.box.ap[i]), 4) if i < len(results.box.ap) else None,
        }

    # Print summary
    print("\n── Results Summary ──────────────────────────────────────────────")
    print(f"  mAP@0.5:       {metrics['mAP_50']:.4f}")
    print(f"  mAP@0.5:0.95:  {metrics['mAP_50_95']:.4f}")
    print(f"  Precision:     {metrics['precision']:.4f}")
    print(f"  Recall:        {metrics['recall']:.4f}")
    for cls, vals in metrics["per_class"].items():
        print(f"  [{cls}] AP@0.5={vals['AP_50']}  AP@0.5:0.95={vals['AP_50_95']}")
    print("─────────────────────────────────────────────────────────────────\n")

    # Visualisations
    plot_precision_recall_curve(results, class_names, output_dir)
    plot_confusion_matrix(results, class_names, output_dir)
    save_metrics_json(metrics, output_dir)

    return metrics


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PowerLine Vision model")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights (.pt)")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    evaluate(args.weights, args.config)
