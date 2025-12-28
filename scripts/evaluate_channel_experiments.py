"""Evaluate trained models from channel combination experiments."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

# Experiment configurations (same as run_channel_experiments.py)
EXPERIMENTS = {
    "exp1_linear": {
        "channels": "linear_stretched",
        "description": "Single channel: linear stretched",
    },
    "exp2_db": {
        "channels": "magnitude_db",
        "description": "Single channel: magnitude dB",
    },
    "exp3_linear_db": {
        "channels": "linear_stretched,magnitude_db",
        "description": "Two channels: linear + dB",
    },
    "exp4_linear_entropy": {
        "channels": "linear_stretched,local_entropy",
        "description": "Two channels: linear + entropy",
    },
    "exp5_db_entropy": {
        "channels": "magnitude_db,local_entropy",
        "description": "Two channels: dB + entropy",
    },
    "exp6_linear_db_entropy": {
        "channels": "linear_stretched,magnitude_db,local_entropy",
        "description": "Three channels: linear + dB + entropy",
    },
}


def find_best_weights(project_dir: Path, exp_name: str) -> Path | None:
    """Find best.pt weights for an experiment."""
    # Try direct match first
    exp_dir = project_dir / exp_name
    if exp_dir.exists():
        best_pt = exp_dir / "weights" / "best.pt"
        if best_pt.exists():
            return best_pt

    # Try with numeric suffix (exp_name, exp_name2, exp_name3, etc.)
    for suffix in ["", "2", "3", "4", "5", "6", "7", "8", "9", "10"]:
        exp_dir = project_dir / f"{exp_name}{suffix}"
        if exp_dir.exists():
            best_pt = exp_dir / "weights" / "best.pt"
            if best_pt.exists():
                return best_pt

    return None


def evaluate_model(
    weights_path: Path,
    data_yaml: Path,
    device: str = "0",
    batch: int = 16,
    imgsz: list = None,
    conf: float = 0.001,
    iou: float = 0.6,
    verbose: bool = True,
) -> dict:
    """Evaluate a model and return metrics.

    Args:
        weights_path: Path to model weights (best.pt)
        data_yaml: Path to dataset config
        device: CUDA device
        batch: Batch size
        imgsz: Image size [h, w]
        conf: Confidence threshold
        iou: IoU threshold for NMS
        verbose: Print detailed output

    Returns:
        Dict with evaluation metrics
    """
    if imgsz is None:
        imgsz = [256, 1024]

    model = YOLO(str(weights_path))

    # Run validation
    metrics = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        batch=batch,
        device=device,
        conf=conf,
        iou=iou,
        rect=True,
        verbose=verbose,
    )

    # Extract metrics
    results = {
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "fitness": float(metrics.fitness),
    }

    # Per-class metrics
    if hasattr(metrics.box, "ap50") and metrics.box.ap50 is not None:
        results["per_class_mAP50"] = metrics.box.ap50.tolist()

    return results


def print_results_table(results: dict, experiments: dict):
    """Print results in a formatted table."""
    print("\n" + "=" * 100)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 100)

    # Header
    header = f"{'Experiment':<25} {'Channels':<35} {'mAP50':>10} {'mAP50-95':>10} {'Precision':>10} {'Recall':>10}"
    print(header)
    print("-" * 100)

    # Sort by mAP50-95 descending
    sorted_results = sorted(results.items(), key=lambda x: x[1].get("mAP50-95", 0), reverse=True)

    for exp_name, metrics in sorted_results:
        if "error" in metrics:
            print(f"{exp_name:<25} {'ERROR: ' + metrics['error']:<35}")
        else:
            channels = experiments.get(exp_name, {}).get("channels", "unknown")
            # Truncate channels if too long
            if len(channels) > 33:
                channels = channels[:30] + "..."
            print(
                f"{exp_name:<25} {channels:<35} "
                f"{metrics['mAP50']:>10.4f} {metrics['mAP50-95']:>10.4f} "
                f"{metrics['precision']:>10.4f} {metrics['recall']:>10.4f}"
            )

    print("=" * 100)

    # Find best experiment
    valid_results = [(k, v) for k, v in results.items() if "error" not in v]
    if valid_results:
        best_exp, best_metrics = max(valid_results, key=lambda x: x[1].get("mAP50-95", 0))
        print(f"\nBest experiment: {best_exp}")
        print(f"  mAP50: {best_metrics['mAP50']:.4f}")
        print(f"  mAP50-95: {best_metrics['mAP50-95']:.4f}")


def save_results_json(results: dict, output_path: Path):
    """Save results to JSON file."""
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained models from channel experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all experiments
  python evaluate_channel_experiments.py

  # Evaluate specific experiments
  python evaluate_channel_experiments.py --experiments exp1_linear exp3_linear_db

  # Custom paths
  python evaluate_channel_experiments.py --project runs/channel_experiments --datasets datasets/channel_experiments

  # Evaluate with custom thresholds
  python evaluate_channel_experiments.py --conf 0.25 --iou 0.5
        """,
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/channel_experiments",
        help="Training project directory containing experiment results",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="datasets/channel_experiments",
        help="Base directory containing generated datasets",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=None,
        help="Specific experiments to evaluate (default: all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--imgsz-h",
        type=int,
        default=256,
        help="Image height",
    )
    parser.add_argument(
        "--imgsz-w",
        type=int,
        default=1024,
        help="Image width",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.6,
        help="IoU threshold for NMS",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results (default: project/evaluation_results.json)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output during evaluation",
    )
    args = parser.parse_args()

    project_dir = Path(args.project)
    datasets_dir = Path(args.datasets)
    imgsz = [args.imgsz_h, args.imgsz_w]

    # Select experiments to evaluate
    if args.experiments:
        experiments = {k: v for k, v in EXPERIMENTS.items() if k in args.experiments}
        if not experiments:
            print(f"[ERROR] No valid experiments found. Available: {list(EXPERIMENTS.keys())}")
            return
    else:
        experiments = EXPERIMENTS

    print(f"\n{'#' * 60}")
    print(f"# Channel Experiment Evaluation")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Project: {project_dir}")
    print(f"# Datasets: {datasets_dir}")
    print(f"# Experiments: {list(experiments.keys())}")
    print(f"{'#' * 60}")

    results = {}

    for exp_name, config in experiments.items():
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {exp_name}")
        print(f"Description: {config['description']}")
        print(f"Channels: {config['channels']}")
        print("=" * 60)

        # Find weights
        weights_path = find_best_weights(project_dir, exp_name)
        if weights_path is None:
            print(f"[WARNING] Weights not found for {exp_name}, skipping...")
            results[exp_name] = {"error": "weights not found"}
            continue

        # Find dataset
        data_yaml = datasets_dir / exp_name / "data.yaml"
        if not data_yaml.exists():
            print(f"[WARNING] Dataset not found: {data_yaml}, skipping...")
            results[exp_name] = {"error": "dataset not found"}
            continue

        print(f"Weights: {weights_path}")
        print(f"Dataset: {data_yaml}")

        try:
            metrics = evaluate_model(
                weights_path=weights_path,
                data_yaml=data_yaml,
                device=args.device,
                batch=args.batch,
                imgsz=imgsz,
                conf=args.conf,
                iou=args.iou,
                verbose=not args.quiet,
            )
            results[exp_name] = metrics
            print(f"\n[SUCCESS] {exp_name}: mAP50={metrics['mAP50']:.4f}, mAP50-95={metrics['mAP50-95']:.4f}")
        except Exception as e:
            print(f"[ERROR] {exp_name}: {e}")
            results[exp_name] = {"error": str(e)}

    # Print summary table
    print_results_table(results, experiments)

    # Save results
    output_path = Path(args.output) if args.output else project_dir / "evaluation_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results_json(results, output_path)


if __name__ == "__main__":
    main()
