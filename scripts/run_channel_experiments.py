"""Run experiments with different channel combinations for radar detection."""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Define experiment configurations
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


def run_command(cmd, description):
    """Run a command and print output."""
    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {description}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"[ERROR] Command failed with return code {result.returncode}")
        return False
    return True


def generate_dataset(exp_name, channels, input_dir, output_base, classes, target_h, target_w, overlap):
    """Generate dataset for an experiment."""
    output_dir = output_base / exp_name

    cmd = [
        sys.executable,
        "scripts/generate_radar_tiff_dataset.py",
        "--input", str(input_dir),
        "--output", str(output_dir),
        "--channels", channels,
        "--classes", classes,
        "--target-h", str(target_h),
        "--target-w", str(target_w),
        "--overlap", str(overlap),
    ]

    success = run_command(cmd, f"Generating dataset for {exp_name}")
    return output_dir if success else None


def train_model(exp_name, data_yaml, model, epochs, batch, device, project):
    """Train YOLO model for an experiment."""
    cmd = [
        sys.executable,
        "scripts/train_radar_yolo.py",
        "--model", model,
        "--data", str(data_yaml),
        "--epochs", str(epochs),
        "--batch", str(batch),
        "--device", device,
        "--project", str(project),
        "--name", exp_name,
    ]

    return run_command(cmd, f"Training model for {exp_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Run channel combination experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Experiments:
  exp1_linear        - Single channel: linear stretched
  exp2_db            - Single channel: magnitude dB
  exp3_linear_db     - Two channels: linear + dB
  exp4_linear_entropy - Two channels: linear + entropy
  exp5_db_entropy    - Two channels: dB + entropy
  exp6_linear_db_entropy - Three channels: linear + dB + entropy

Examples:
  # Run all experiments
  python run_channel_experiments.py

  # Run specific experiments
  python run_channel_experiments.py --experiments exp1_linear exp3_linear_db

  # Generate datasets only (no training)
  python run_channel_experiments.py --generate-only

  # Train only (datasets already generated)
  python run_channel_experiments.py --train-only
        """,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="datasets/RD_DATA_1225_DEMO",
        help="Input dataset directory",
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default="datasets/channel_experiments",
        help="Base output directory for generated datasets",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default="0:target,1:reflector",
        help="Class mapping",
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
        default="runs/channel_experiments",
        help="Training project directory",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=None,
        help="Specific experiments to run (default: all)",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only generate datasets, skip training",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only train models, skip dataset generation",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_base = Path(args.output_base)
    project = Path(args.project)

    # Select experiments to run
    if args.experiments:
        experiments = {k: v for k, v in EXPERIMENTS.items() if k in args.experiments}
        if not experiments:
            print(f"[ERROR] No valid experiments found. Available: {list(EXPERIMENTS.keys())}")
            return
    else:
        experiments = EXPERIMENTS

    print(f"\n{'#'*60}")
    print(f"# Channel Combination Experiments")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Experiments: {list(experiments.keys())}")
    print(f"{'#'*60}")

    results = {}

    for exp_name, config in experiments.items():
        print(f"\n\n{'*'*60}")
        print(f"* Experiment: {exp_name}")
        print(f"* {config['description']}")
        print(f"* Channels: {config['channels']}")
        print(f"{'*'*60}")

        dataset_dir = output_base / exp_name
        data_yaml = dataset_dir / "data.yaml"

        # Generate dataset
        if not args.train_only:
            dataset_dir = generate_dataset(
                exp_name,
                config["channels"],
                input_dir,
                output_base,
                args.classes,
                args.target_h,
                args.target_w,
                args.overlap,
            )
            if dataset_dir is None:
                results[exp_name] = "FAILED (dataset generation)"
                continue

        # Train model
        if not args.generate_only:
            if not data_yaml.exists():
                print(f"[ERROR] Dataset not found: {data_yaml}")
                results[exp_name] = "FAILED (dataset not found)"
                continue

            success = train_model(
                exp_name,
                data_yaml,
                args.model,
                args.epochs,
                args.batch,
                args.device,
                project,
            )
            results[exp_name] = "SUCCESS" if success else "FAILED (training)"
        else:
            results[exp_name] = "GENERATED"

    # Print summary
    print(f"\n\n{'#'*60}")
    print(f"# Experiment Summary")
    print(f"# Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")

    for exp_name, status in results.items():
        print(f"  {exp_name}: {status}")

    print(f"\nDatasets saved to: {output_base}")
    if not args.generate_only:
        print(f"Training results saved to: {project}")


if __name__ == "__main__":
    main()
