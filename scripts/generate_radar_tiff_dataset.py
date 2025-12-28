"""Convert radar NPY data to float TIFF format with slicing and class filtering."""

import argparse
from pathlib import Path

import numpy as np
from tifffile import imwrite
from tqdm import tqdm


def parse_class_mapping(class_str):
    """Parse class mapping string like '0:target,1:reflector'.

    Args:
        class_str: String in format 'id:name,id:name,...'

    Returns:
        Tuple of (id_to_name dict, old_to_new_id dict)
    """
    if not class_str:
        return None, None

    id_to_name = {}
    for item in class_str.split(","):
        item = item.strip()
        if ":" in item:
            cls_id, name = item.split(":", 1)
            id_to_name[int(cls_id)] = name.strip()

    # Create remapping: old_id -> new_id (sequential from 0)
    sorted_ids = sorted(id_to_name.keys())
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(sorted_ids)}

    # Create final name mapping with new IDs
    new_id_to_name = {old_to_new[old_id]: name for old_id, name in id_to_name.items()}

    return new_id_to_name, old_to_new


def compute_slice_positions(total_size, slice_size, min_overlap_ratio=0.2):
    """Compute slice positions with minimum overlap.

    Args:
        total_size: Total size of the dimension
        slice_size: Target slice size
        min_overlap_ratio: Minimum overlap ratio between slices

    Returns:
        List of (start, end) tuples for each slice
    """
    if total_size <= slice_size:
        return [(0, total_size)]

    overlap = int(slice_size * min_overlap_ratio)
    stride = slice_size - overlap
    positions = []
    start = 0

    while start + slice_size <= total_size:
        positions.append((start, start + slice_size))
        start += stride

    # Handle last slice to cover remaining area
    if positions[-1][1] < total_size:
        positions.append((total_size - slice_size, total_size))

    return positions


def adjust_labels_for_slice(labels, orig_h, orig_w, y_start, y_end, x_start, x_end, old_to_new=None):
    """Adjust labels for a slice, keeping only boxes with center in slice.

    Args:
        labels: List of label lines in YOLO format
        orig_h: Original image height
        orig_w: Original image width
        y_start: Slice start in y dimension
        y_end: Slice end in y dimension
        x_start: Slice start in x dimension
        x_end: Slice end in x dimension
        old_to_new: Dict mapping old class IDs to new IDs (None to keep all)

    Returns:
        List of adjusted label strings
    """
    slice_h = y_end - y_start
    slice_w = x_end - x_start
    adjusted = []

    for label in labels:
        parts = label.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])

        # Filter by class ID if mapping provided
        if old_to_new is not None:
            if cls_id not in old_to_new:
                continue  # Skip labels with unmapped class IDs
            cls_id = old_to_new[cls_id]  # Remap to new ID

        # Convert to pixel coords
        cx_px = cx * orig_w
        cy_px = cy * orig_h

        # Check if center is in slice
        if not (x_start <= cx_px < x_end and y_start <= cy_px < y_end):
            continue

        # Adjust to slice coordinates
        new_cx = (cx_px - x_start) / slice_w
        new_cy = (cy_px - y_start) / slice_h
        new_w = w * orig_w / slice_w
        new_h = h * orig_h / slice_h

        # Clip values to [0, 1]
        new_cx = np.clip(new_cx, 0, 1)
        new_cy = np.clip(new_cy, 0, 1)
        new_w = np.clip(new_w, 0, 1)
        new_h = np.clip(new_h, 0, 1)

        adjusted.append(f"{cls_id} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f}")

    return adjusted


def process_image(
    npy_path,
    label_path,
    output_img_dir,
    output_label_dir,
    target_h=256,
    target_w=1024,
    overlap_ratio=0.2,
    old_to_new=None,
):
    """Process single image: convert to magnitude, normalize, slice, save.

    Args:
        npy_path: Path to input NPY file
        label_path: Path to input label file
        output_img_dir: Output directory for images
        output_label_dir: Output directory for labels
        target_h: Target slice height
        target_w: Target slice width
        overlap_ratio: Overlap ratio between slices
        old_to_new: Dict mapping old class IDs to new IDs

    Returns:
        Number of slices generated
    """
    # Load complex data
    data = np.load(npy_path)
    orig_h, orig_w = data.shape

    # Compute magnitude and normalize per-image to [0, 1]
    magnitude = np.abs(data)
    mag_min, mag_max = magnitude.min(), magnitude.max()
    if mag_max > mag_min:
        normalized = (magnitude - mag_min) / (mag_max - mag_min)
    else:
        normalized = np.zeros_like(magnitude, dtype=np.float32)

    # Load labels
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
            # Extract slice
            img_slice = normalized[y_start:y_end, x_start:x_end]

            # Adjust labels for this slice
            slice_labels = adjust_labels_for_slice(labels, orig_h, orig_w, y_start, y_end, x_start, x_end, old_to_new)

            # Save image as float32 TIFF
            slice_name = f"{base_name}_s{slice_idx}"
            img_out = output_img_dir / f"{slice_name}.tiff"
            imwrite(str(img_out), img_slice.astype(np.float32))

            # Save labels
            label_out = output_label_dir / f"{slice_name}.txt"
            with open(label_out, "w") as f:
                f.write("\n".join(slice_labels))

            slice_idx += 1

    return slice_idx


def create_data_yaml(output_dir, class_names):
    """Create data.yaml for the dataset.

    Args:
        output_dir: Output dataset directory
        class_names: Dict mapping class ID to name
    """
    # Build names section
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
"""

    with open(output_dir / "data.yaml", "w") as f:
        f.write(yaml_content)


def main():
    parser = argparse.ArgumentParser(
        description="Convert radar NPY to TIFF with class filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Keep only classes 0 and 1
  python generate_radar_tiff_dataset.py --classes "0:target,1:reflector"

  # Keep classes 0, 1, 3 (3 will be remapped to 2)
  python generate_radar_tiff_dataset.py --classes "0:target,1:reflector,3:clutter"
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
        default="datasets/RD_TIFF_DATA",
        help="Output dataset directory",
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
        help="Minimum overlap ratio between slices",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default="0:target,1:reflector",
        help="Class mapping in format 'id:name,id:name,...'. Unmapped classes will be ignored.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    # Parse class mapping
    class_names, old_to_new = parse_class_mapping(args.classes)

    if class_names:
        print(f"Class mapping: {args.classes}")
        print(f"  Keeping classes: {list(old_to_new.keys())}")
        print(f"  Remapped to: {class_names}")
    else:
        print("No class filtering applied (keeping all classes)")

    # Process train and val sets
    for split in ["train", "val"]:
        input_img_dir = input_dir / split / "images"
        input_label_dir = input_dir / split / "labels"
        output_img_dir = output_dir / split / "images"
        output_label_dir = output_dir / split / "labels"

        output_img_dir.mkdir(parents=True, exist_ok=True)
        output_label_dir.mkdir(parents=True, exist_ok=True)

        npy_files = sorted(input_img_dir.glob("*.npy"))
        total_slices = 0

        for npy_path in tqdm(npy_files, desc=f"Processing {split}"):
            label_path = input_label_dir / f"{npy_path.stem}.txt"
            slices = process_image(
                npy_path,
                label_path,
                output_img_dir,
                output_label_dir,
                args.target_h,
                args.target_w,
                args.overlap,
                old_to_new,
            )
            total_slices += slices

        print(f"{split}: {len(npy_files)} images -> {total_slices} slices")

    # Create data.yaml
    create_data_yaml(output_dir, class_names or {0: "object"})
    print(f"Dataset saved to {output_dir}")


if __name__ == "__main__":
    main()
