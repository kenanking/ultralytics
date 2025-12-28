"""Convert radar NPY data to float TIFF format with slicing and class filtering."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import maximum_filter, uniform_filter
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


def compute_magnitude_db(rd_complex: np.ndarray) -> np.ndarray:
    """Compute magnitude in dB scale.

    Args:
        rd_complex: Complex-valued RD matrix

    Returns:
        Magnitude in dB (normalized to 0-1 range)
    """
    magnitude = np.abs(rd_complex)
    # Avoid log(0) by adding small epsilon
    magnitude_db = 20 * np.log10(magnitude + 1e-10)
    # Normalize to 0-1 range
    magnitude_db = (magnitude_db - magnitude_db.min()) / (magnitude_db.max() - magnitude_db.min() + 1e-10)
    return magnitude_db.astype(np.float32)


def compute_linear_stretched(rd_complex: np.ndarray, sigma_factor: float = 25.0) -> np.ndarray:
    """Compute linear stretched image with sigma clipping.

    Args:
        rd_complex: Complex-valued RD matrix
        sigma_factor: Number of sigma for clipping (default 25)

    Returns:
        Linear stretched magnitude (normalized to 0-1 range)
    """
    magnitude = np.abs(rd_complex)
    mean_val = np.mean(magnitude)
    std_val = np.std(magnitude)

    # Clip values beyond sigma_factor * std
    lower_bound = 0
    upper_bound = mean_val + sigma_factor * std_val

    clipped = np.clip(magnitude, lower_bound, upper_bound)
    # Normalize to 0-1 range
    normalized = (clipped - clipped.min()) / (clipped.max() - clipped.min() + 1e-10)
    return normalized.astype(np.float32)


def compute_local_entropy(rd_complex: np.ndarray, window_size: int = 7, num_bins: int = 64) -> np.ndarray:
    """Compute local entropy using PyTorch for GPU acceleration.

    Args:
        rd_complex: Complex-valued RD matrix
        window_size: Size of the sliding window
        num_bins: Number of bins for histogram

    Returns:
        Local entropy map (normalized to 0-1 range)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    magnitude = np.abs(rd_complex)

    # Normalize magnitude to [0, 1]
    mag_normalized = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-10)

    # Quantize to bins
    quantized = np.clip((mag_normalized * num_bins).astype(np.int64), 0, num_bins - 1)

    # Convert to torch tensor
    quantized_tensor = torch.from_numpy(quantized).to(device)

    # Create one-hot encoding
    one_hot = torch.zeros(num_bins, *quantized.shape, dtype=torch.float32, device=device)
    for i in range(num_bins):
        one_hot[i] = (quantized_tensor == i).float()

    # Create averaging kernel
    kernel = torch.ones(1, 1, window_size, window_size, dtype=torch.float32, device=device)
    kernel = kernel / (window_size**2)

    # Apply convolution to each bin
    padding = window_size // 2
    one_hot = one_hot.unsqueeze(1)  # Shape: (num_bins, 1, H, W)
    local_probs = F.conv2d(one_hot, kernel, padding=padding)
    local_probs = local_probs.squeeze(1)  # Shape: (num_bins, H, W)

    # Compute entropy: -sum(p * log(p))
    log_probs = torch.log2(local_probs + 1e-10)
    entropy_map = -torch.sum(local_probs * log_probs, dim=0)

    # Convert back to numpy
    entropy_map = entropy_map.cpu().numpy()

    # Normalize to 0-1 range
    entropy_map = (entropy_map - entropy_map.min()) / (entropy_map.max() - entropy_map.min() + 1e-10)
    return entropy_map.astype(np.float32)


def compute_energy_concentration(rd_complex: np.ndarray, window_size: int = 7) -> np.ndarray:
    """Compute energy concentration ratio in a sliding window.

    Energy concentration = (max energy in window) / (total energy in window)
    High values indicate concentrated energy (potential targets).

    Args:
        rd_complex: Complex-valued RD matrix
        window_size: Size of the sliding window

    Returns:
        Energy concentration map (normalized to 0-1 range)
    """
    energy = np.abs(rd_complex) ** 2

    # Compute local max using fast morphological operation
    local_max = maximum_filter(energy, size=window_size, mode="reflect")

    # Compute local sum (total energy in window)
    local_sum = uniform_filter(energy, size=window_size, mode="reflect") * (window_size**2)

    # Compute concentration ratio: max / sum
    concentration_map = local_max / (local_sum + 1e-10)

    # Normalize to 0-1 range
    concentration_map = (concentration_map - concentration_map.min()) / (
        concentration_map.max() - concentration_map.min() + 1e-10
    )
    return concentration_map.astype(np.float32)


def compute_local_variance(rd_complex: np.ndarray, window_size: int = 7) -> np.ndarray:
    """Compute local variance in a sliding window.

    Args:
        rd_complex: Complex-valued RD matrix
        window_size: Size of the sliding window

    Returns:
        Local variance map (normalized to 0-1 range)
    """
    magnitude = np.abs(rd_complex)

    # Compute local mean
    local_mean = uniform_filter(magnitude, size=window_size, mode="reflect")
    # Compute local mean of squared values
    local_mean_sq = uniform_filter(magnitude**2, size=window_size, mode="reflect")
    # Variance = E[X^2] - E[X]^2
    variance_map = local_mean_sq - local_mean**2
    variance_map = np.maximum(variance_map, 0)  # Ensure non-negative

    # Normalize to 0-1 range (using log scale for better visualization)
    variance_map_log = np.log10(variance_map + 1e-10)
    variance_map_normalized = (variance_map_log - variance_map_log.min()) / (
        variance_map_log.max() - variance_map_log.min() + 1e-10
    )
    return variance_map_normalized.astype(np.float32)


# Registry of available channel methods
CHANNEL_METHODS = {
    "linear_stretched": compute_linear_stretched,
    "magnitude_db": compute_magnitude_db,
    "local_entropy": compute_local_entropy,
    "energy_concentration": compute_energy_concentration,
    "local_variance": compute_local_variance,
}


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
    channel_methods=None,
):
    """Process single image: compute multi-channel representation, slice, save.

    Args:
        npy_path: Path to input NPY file
        label_path: Path to input label file
        output_img_dir: Output directory for images
        output_label_dir: Output directory for labels
        target_h: Target slice height
        target_w: Target slice width
        overlap_ratio: Overlap ratio between slices
        old_to_new: Dict mapping old class IDs to new IDs
        channel_methods: List of channel method names (default: ["linear_stretched"])

    Returns:
        Number of slices generated
    """
    if channel_methods is None:
        channel_methods = ["linear_stretched"]

    # Load complex data
    data = np.load(npy_path)
    orig_h, orig_w = data.shape

    # Compute each channel
    channels = []
    for method_name in channel_methods:
        channel_data = CHANNEL_METHODS[method_name](data)
        channels.append(channel_data)

    # Stack channels: shape (C, H, W)
    multi_channel = np.stack(channels, axis=0)

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
            # Extract slice from all channels: (C, H, W)
            img_slice = multi_channel[:, y_start:y_end, x_start:x_end]

            # For single channel, squeeze to (H, W) for backward compatibility
            if img_slice.shape[0] == 1:
                img_slice = img_slice[0]

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


def create_data_yaml(output_dir, class_names, num_channels=1):
    """Create data.yaml for the dataset.

    Args:
        output_dir: Output dataset directory
        class_names: Dict mapping class ID to name
        num_channels: Number of input channels
    """
    # Build names section
    names_lines = "\n".join(f"  {cls_id}: {name}" for cls_id, name in sorted(class_names.items()))

    # Build normalization arrays for multi-channel
    mean_str = ", ".join(["0.0"] * num_channels)
    std_str = ", ".join(["1.0"] * num_channels)

    yaml_content = f"""path: {output_dir.resolve()}
train: train/images
val: val/images

names:
{names_lines}

channels: {num_channels}

normalize:
  mean: [{mean_str}]
  std: [{std_str}]

padding_value: 0.0
"""

    with open(output_dir / "data.yaml", "w") as f:
        f.write(yaml_content)


def main():
    available_methods = ", ".join(CHANNEL_METHODS.keys())
    parser = argparse.ArgumentParser(
        description="Convert radar NPY to TIFF with class filtering and multi-channel support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Keep only classes 0 and 1 (single channel, default)
  python generate_radar_tiff_dataset.py --classes "0:target,1:reflector"

  # Multi-channel: magnitude_db + local_entropy
  python generate_radar_tiff_dataset.py --channels "magnitude_db,local_entropy"

  # All 5 channels
  python generate_radar_tiff_dataset.py --channels "magnitude_db,linear_stretched,local_entropy,energy_concentration,local_variance"

Available channel methods: {available_methods}
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
    parser.add_argument(
        "--channels",
        type=str,
        default="linear_stretched",
        help=f"Comma-separated processing methods: {available_methods}",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    # Parse and validate channel methods
    channel_methods = [m.strip() for m in args.channels.split(",")]
    for method in channel_methods:
        if method not in CHANNEL_METHODS:
            raise ValueError(f"Unknown channel method: '{method}'. Available: {list(CHANNEL_METHODS.keys())}")

    print(f"Channel methods: {channel_methods} ({len(channel_methods)} channels)")

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
                channel_methods,
            )
            total_slices += slices

        print(f"{split}: {len(npy_files)} images -> {total_slices} slices")

    # Create data.yaml
    create_data_yaml(output_dir, class_names or {0: "object"}, len(channel_methods))
    print(f"Dataset saved to {output_dir}")


if __name__ == "__main__":
    main()
