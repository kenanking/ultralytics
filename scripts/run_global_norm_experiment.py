"""Run experiment with global min-max normalization for magnitude data.

This script uses global normalization instead of per-image normalization:
1. First pass: Compute magnitude (linear or dB) for all images to find global min/max
2. Second pass: Normalize all images using global min/max and save
3. Train the model
4. Evaluate the model

Supported modes:
- linear: Use linear magnitude (|x|)
- db: Use magnitude in dB scale (20 * log10(|x|))
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from tifffile import imwrite
from tqdm import tqdm

from ultralytics import YOLO


def parse_class_mapping(class_str):
    """Parse class mapping string like '0:target,1:reflector'."""
    if not class_str:
        return None, None

    id_to_name = {}
    for item in class_str.split(","):
        item = item.strip()
        if ":" in item:
            cls_id, name = item.split(":", 1)
            id_to_name[int(cls_id)] = name.strip()

    sorted_ids = sorted(id_to_name.keys())
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(sorted_ids)}
    new_id_to_name = {old_to_new[old_id]: name for old_id, name in id_to_name.items()}

    return new_id_to_name, old_to_new


def compute_slice_positions(total_size, slice_size, min_overlap_ratio=0.2):
    """Compute slice positions with minimum overlap."""
    if total_size <= slice_size:
        return [(0, total_size)]

    overlap = int(slice_size * min_overlap_ratio)
    stride = slice_size - overlap
    positions = []
    start = 0

    while start + slice_size <= total_size:
        positions.append((start, start + slice_size))
        start += stride

    if positions[-1][1] < total_size:
        positions.append((total_size - slice_size, total_size))

    return positions


def adjust_labels_for_slice(labels, orig_h, orig_w, y_start, y_end, x_start, x_end, old_to_new=None):
    """Adjust labels for a slice, keeping only boxes with center in slice."""
    slice_h = y_end - y_start
    slice_w = x_end - x_start
    adjusted = []

    for label in labels:
        parts = label.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])

        if old_to_new is not None:
            if cls_id not in old_to_new:
                continue
            cls_id = old_to_new[cls_id]

        cx_px = cx * orig_w
        cy_px = cy * orig_h

        if not (x_start <= cx_px < x_end and y_start <= cy_px < y_end):
            continue

        new_cx = (cx_px - x_start) / slice_w
        new_cy = (cy_px - y_start) / slice_h
        new_w = w * orig_w / slice_w
        new_h = h * orig_h / slice_h

        new_cx = np.clip(new_cx, 0, 1)
        new_cy = np.clip(new_cy, 0, 1)
        new_w = np.clip(new_w, 0, 1)
        new_h = np.clip(new_h, 0, 1)

        adjusted.append(f"{cls_id} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f}")

    return adjusted


def compute_global_stats(input_dir: Path, splits: list[str], mode: str = "linear") -> tuple[float, float]:
    """Compute global min/max across all images.

    Args:
        input_dir: Input dataset directory
        splits: List of splits to process (e.g., ["train", "val"])
        mode: "linear" for magnitude, "db" for magnitude in dB scale

    Returns:
        Tuple of (global_min, global_max)
    """
    print(f"\n[Phase 1] Computing global statistics (mode: {mode})...")

    global_min = float("inf")
    global_max = float("-inf")
    total_files = 0

    for split in splits:
        img_dir = input_dir / split / "images"
        npy_files = sorted(img_dir.glob("*.npy"))

        for npy_path in tqdm(npy_files, desc=f"Scanning {split}"):
            data = np.load(npy_path)
            magnitude = np.abs(data)

            if mode == "db":
                # Convert to dB scale, avoid log(0)
                magnitude = 20 * np.log10(magnitude + 1e-10)

            global_min = min(global_min, magnitude.min())
            global_max = max(global_max, magnitude.max())
            total_files += 1

    print(f"  Total files scanned: {total_files}")
    unit = "dB" if mode == "db" else "linear"
    print(f"  Global {unit} range: [{global_min:.6f}, {global_max:.6f}]")

    return global_min, global_max


def generate_dataset_with_global_norm(
    input_dir: Path,
    output_dir: Path,
    global_min: float,
    global_max: float,
    target_h: int = 256,
    target_w: int = 1024,
    overlap_ratio: float = 0.2,
    old_to_new: dict = None,
    mode: str = "linear",
) -> dict:
    """Generate dataset using global normalization.

    Args:
        input_dir: Input dataset directory
        output_dir: Output dataset directory
        global_min: Global minimum magnitude value
        global_max: Global maximum magnitude value
        target_h: Target slice height
        target_w: Target slice width
        overlap_ratio: Overlap ratio between slices
        old_to_new: Class ID remapping dict
        mode: "linear" for magnitude, "db" for magnitude in dB scale

    Returns:
        Dict with statistics
    """
    print(f"\n[Phase 2] Generating dataset with global normalization (mode: {mode})...")

    stats = {"train": {"images": 0, "slices": 0}, "val": {"images": 0, "slices": 0}}
    norm_range = global_max - global_min

    for split in ["train", "val"]:
        input_img_dir = input_dir / split / "images"
        input_label_dir = input_dir / split / "labels"
        output_img_dir = output_dir / split / "images"
        output_label_dir = output_dir / split / "labels"

        output_img_dir.mkdir(parents=True, exist_ok=True)
        output_label_dir.mkdir(parents=True, exist_ok=True)

        npy_files = sorted(input_img_dir.glob("*.npy"))

        for npy_path in tqdm(npy_files, desc=f"Processing {split}"):
            # Load and compute magnitude
            data = np.load(npy_path)
            orig_h, orig_w = data.shape
            magnitude = np.abs(data)

            if mode == "db":
                # Convert to dB scale, avoid log(0)
                magnitude = 20 * np.log10(magnitude + 1e-10)

            # Global normalization
            if norm_range > 0:
                normalized = (magnitude - global_min) / norm_range
            else:
                normalized = np.zeros_like(magnitude, dtype=np.float32)

            # Load labels
            label_path = input_label_dir / f"{npy_path.stem}.txt"
            labels = []
            if label_path.exists():
                with open(label_path) as f:
                    labels = f.readlines()

            # Compute slice positions
            h_slices = compute_slice_positions(orig_h, target_h, overlap_ratio)
            w_slices = compute_slice_positions(orig_w, target_w, overlap_ratio)

            # Generate slices
            base_name = npy_path.stem
            slice_idx = 0

            for y_start, y_end in h_slices:
                for x_start, x_end in w_slices:
                    img_slice = normalized[y_start:y_end, x_start:x_end]
                    slice_labels = adjust_labels_for_slice(
                        labels, orig_h, orig_w, y_start, y_end, x_start, x_end, old_to_new
                    )

                    slice_name = f"{base_name}_s{slice_idx}"
                    img_out = output_img_dir / f"{slice_name}.tiff"
                    imwrite(str(img_out), img_slice.astype(np.float32))

                    label_out = output_label_dir / f"{slice_name}.txt"
                    with open(label_out, "w") as f:
                        f.write("\n".join(slice_labels))

                    slice_idx += 1

            stats[split]["images"] += 1
            stats[split]["slices"] += slice_idx

        print(f"  {split}: {stats[split]['images']} images -> {stats[split]['slices']} slices")

    return stats


def create_data_yaml(output_dir: Path, class_names: dict, global_min: float, global_max: float, mode: str = "linear"):
    """Create data.yaml with global normalization info."""
    names_lines = "\n".join(f"  {cls_id}: {name}" for cls_id, name in sorted(class_names.items()))

    yaml_content = f"""path: {output_dir.resolve()}
train: train/images
val: val/images

names:
{names_lines}

channels: 1

normalize:
  mean: [0.0]
  std: [1.0]

padding_value: 0.0

# Global normalization info (for reference)
# mode: {mode}
# global_min: {global_min}
# global_max: {global_max}
"""

    with open(output_dir / "data.yaml", "w") as f:
        f.write(yaml_content)


def train_model(
    data_yaml: Path,
    model_cfg: str = "yolo11n.yaml",
    epochs: int = 150,
    batch: int = 16,
    device: str = "0",
    project: str = "runs/global_norm_experiment",
    name: str = "magnitude_global_norm",
) -> Path:
    """Train YOLO model.

    Returns:
        Path to best weights
    """
    print(f"\n[Phase 3] Training model for {epochs} epochs...")

    model = YOLO(model_cfg)

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=[256, 1024],
        rect=True,
        batch=batch,
        close_mosaic=int(epochs * 0.67),  # Disable mosaic in last 1/3
        device=device,
        workers=8,
        project=project,
        name=name,
    )

    print(f"  Training complete. Results saved to {results.save_dir}")
    return Path(results.save_dir) / "weights" / "best.pt"


def evaluate_model(
    weights_path: Path,
    data_yaml: Path,
    device: str = "0",
    batch: int = 16,
) -> dict:
    """Evaluate trained model.

    Returns:
        Dict with evaluation metrics
    """
    print("\n[Phase 4] Evaluating model...")

    model = YOLO(str(weights_path))

    metrics = model.val(
        data=str(data_yaml),
        imgsz=[256, 1024],
        batch=batch,
        device=device,
        rect=True,
    )

    results = {
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run experiment with global min-max normalization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This experiment uses global normalization instead of per-image normalization:
  1. First pass: Scan all images to find global min/max magnitude
  2. Second pass: Normalize all images using global min/max
  3. Train the YOLO model
  4. Evaluate the trained model

Modes:
  linear - Use linear magnitude |x|
  db     - Use magnitude in dB scale: 20 * log10(|x|)

Examples:
  # Linear magnitude with global normalization
  python run_global_norm_experiment.py --mode linear --epochs 150

  # dB magnitude with global normalization
  python run_global_norm_experiment.py --mode db --epochs 150
        """,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="datasets/RD_DATA_1225_DEMO",
        help="Input dataset directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output dataset directory (default: datasets/global_norm_{mode})",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["linear", "db"],
        default="linear",
        help="Magnitude mode: 'linear' for |x|, 'db' for 20*log10(|x|)",
    )
    parser.add_argument(
        "--target-h",
        type=int,
        default=256,
        help="Target slice height",
    )
    parser.add_argument(
        "--target-w",
        type=int,
        default=1024,
        help="Target slice width",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.2,
        help="Slice overlap ratio",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default="0:target,1:reflector",
        help="Class mapping",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.yaml",
        help="YOLO model config",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Training epochs",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/global_norm_experiment",
        help="Training project directory",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="magnitude_global_norm",
        help="Experiment name",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only generate dataset, skip training and evaluation",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only train (dataset already generated)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only evaluate (model already trained)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to weights for evaluation (required with --eval-only)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    # Set default output directory based on mode
    if args.output is None:
        output_dir = Path(f"datasets/global_norm_{args.mode}")
    else:
        output_dir = Path(args.output)

    # Set default experiment name based on mode
    exp_name = args.name if args.name != "magnitude_global_norm" else f"{args.mode}_global_norm"

    print(f"\n{'#' * 70}")
    print(f"# Global Normalization Experiment")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Mode: {args.mode}")
    print(f"# Input: {input_dir}")
    print(f"# Output: {output_dir}")
    print(f"{'#' * 70}")

    # Parse class mapping
    class_names, old_to_new = parse_class_mapping(args.classes)
    data_yaml = output_dir / "data.yaml"

    # Phase 1 & 2: Generate dataset
    if not args.train_only and not args.eval_only:
        # Compute global statistics
        global_min, global_max = compute_global_stats(input_dir, ["train", "val"], mode=args.mode)

        # Generate dataset with global normalization
        stats = generate_dataset_with_global_norm(
            input_dir=input_dir,
            output_dir=output_dir,
            global_min=global_min,
            global_max=global_max,
            target_h=args.target_h,
            target_w=args.target_w,
            overlap_ratio=args.overlap,
            old_to_new=old_to_new,
            mode=args.mode,
        )

        # Create data.yaml
        create_data_yaml(output_dir, class_names or {0: "object"}, global_min, global_max, mode=args.mode)
        print(f"\n  Dataset saved to: {output_dir}")

        if args.generate_only:
            print("\n[Done] Dataset generation complete (--generate-only)")
            return

    # Phase 3: Train model
    if not args.eval_only:
        if not data_yaml.exists():
            print(f"[ERROR] Dataset not found: {data_yaml}")
            return

        weights_path = train_model(
            data_yaml=data_yaml,
            model_cfg=args.model,
            epochs=args.epochs,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=exp_name,
        )
    else:
        if args.weights:
            weights_path = Path(args.weights)
        else:
            print("[ERROR] --weights required with --eval-only")
            return

    # Phase 4: Evaluate
    if not weights_path.exists():
        print(f"[ERROR] Weights not found: {weights_path}")
        return

    metrics = evaluate_model(
        weights_path=weights_path,
        data_yaml=data_yaml,
        device=args.device,
        batch=args.batch,
    )

    # Print final results
    print(f"\n{'=' * 70}")
    print("EXPERIMENT RESULTS")
    print("=" * 70)
    print(f"  Mode:      {args.mode}")
    print(f"  Dataset:   {output_dir}")
    print(f"  Weights:   {weights_path}")
    print(f"  mAP50:     {metrics['mAP50']:.4f}")
    print(f"  mAP50-95:  {metrics['mAP50-95']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
