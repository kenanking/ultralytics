"""Train YOLO model on radar TIFF dataset."""

import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLO on radar data")
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.yaml",
        help="Model config (yolo11n.yaml, yolo11s.yaml, etc.)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="datasets/RD_TIFF_DATA/data.yaml",
        help="Dataset config file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Number of training epochs",
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
        help="CUDA device (0, 1, cpu)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/radar",
        help="Project directory",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="yolo11n_radar",
        help="Experiment name",
    )
    args = parser.parse_args()

    # Initialize model
    model = YOLO(args.model)

    # Train
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=[256, 1024],  # h x w for rectangular images
        rect=True,  # Rectangular training
        batch=args.batch,
        close_mosaic=100,  # Disable mosaic after epoch 100
        device=args.device,
        workers=8,
        project=args.project,
        name=args.name,
        # Float image settings from data.yaml are auto-loaded
    )

    print(f"Training complete. Results saved to {results.save_dir}")

    # Validate
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
