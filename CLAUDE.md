# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
conda activate ultralytics
```

## Common Commands

```bash
# Install in development mode
pip install -e .

# Run tests
pytest tests/
pytest tests/test_python.py              # Specific test file
pytest tests/test_python.py -k "test_detect"  # Specific test pattern
pytest --slow                             # Include slow tests

# Linting and formatting
ruff check ultralytics/                   # Check for issues
ruff format ultralytics/                  # Format code

# CLI usage
yolo detect train data=coco8.yaml model=yolo11n.pt epochs=100
yolo predict model=yolo11n.pt source='image.jpg'
yolo export model=yolo11n.pt format=onnx
```

## Architecture Overview

### Package Structure

```
ultralytics/
├── cfg/          # Configuration system (YAML parsing, CLI entrypoint)
├── data/         # Dataset classes and augmentation pipeline
├── engine/       # Core training/validation/prediction orchestration
├── models/       # Model definitions and task-specific implementations
├── nn/           # Neural network architecture components
├── solutions/    # Pre-built computer vision solutions
├── trackers/     # Object tracking (ByteTrack, BOTSORT)
└── utils/        # Utilities (metrics, ops, plotting, torch helpers)
```

### Core Components

**engine/model.py**: Base `Model` class - unified API for all YOLO operations. Entry point for `train()`, `val()`, `predict()`, `export()`.

**engine/trainer.py**: `BaseTrainer` handles training loop, optimization, checkpointing, distributed training (DDP).

**engine/validator.py**: `BaseValidator` for model evaluation and metric computation.

**engine/predictor.py**: `BasePredictor` for inference across multiple input sources and model formats.

**nn/autobackend.py**: Multi-format model inference backend (PyTorch, ONNX, TensorRT, CoreML, etc.).

### Task Abstraction Pattern

Each task (detect, segment, classify, pose, obb) under `models/yolo/` follows:
- `train.py`: Task-specific trainer extending `BaseTrainer`
- `val.py`: Task-specific validator extending `BaseValidator`
- `predict.py`: Task-specific predictor extending `BasePredictor`

The `YOLO` class in `models/yolo/model.py` uses a `task_map` property to dispatch to task-specific components.

### Configuration System

- **cfg/default.yaml**: Central configuration with all parameters
- **cfg/__init__.py**: `entrypoint()` function handles CLI parsing and dispatches to model methods
- Configuration resolution: defaults → task overrides → CLI/programmatic overrides

### Data Pipeline

- **data/base.py**: `BaseDataset` for common dataset operations
- **data/build.py**: `build_yolo_dataset()`, `build_dataloader()` factories
- **data/augment.py**: Mosaic, MixUp, Albumentations, photometric/geometric transforms

## Code Style

- Line length: 120 characters
- Docstrings: Google-style format
- Type annotations encouraged (stubs available in `[typing]` extras)
- Formatting: Ruff (configured in pyproject.toml)
