import argparse
import glob
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.utils.nms import TorchNMS
from ultralytics.utils.patches import imread

IMG_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sliced inference with result merging.")
    parser.add_argument("--model", required=True, help="Path to a YOLO model.")
    parser.add_argument("--source", required=True, help="Image file, directory, or glob pattern.")
    parser.add_argument("--output", default="runs/sliced_predict", help="Output directory.")
    parser.add_argument("--imgsz", type=int, default=None, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image.")
    parser.add_argument("--device", default=None, help="Device, i.e. 0 or cpu.")
    parser.add_argument("--half", action="store_true", help="Use FP16 inference.")
    parser.add_argument("--classes", nargs="*", type=int, default=None, help="Filter by class IDs.")
    parser.add_argument("--save-img", action="store_true", help="Save annotated images.")
    parser.add_argument("--save-txt", action="store_true", help="Save YOLO-format labels.")
    parser.add_argument("--line-thickness", type=int, default=2, help="Bounding box line thickness.")
    parser.add_argument("--channels", type=int, default=None, help="Force input channels (1 or 3).")
    parser.add_argument("--normalize-mean", type=float, nargs="+", default=None, help="Override normalize_mean.")
    parser.add_argument("--normalize-std", type=float, nargs="+", default=None, help="Override normalize_std.")
    parser.add_argument("--padding-value", type=float, default=None, help="Override padding_value.")
    parser.add_argument("--tile-w", type=int, default=1024, help="Tile width for sliced inference.")
    parser.add_argument("--tile-h", type=int, default=256, help="Tile height for sliced inference.")
    parser.add_argument("--overlap", type=float, default=0.2, help="Tile overlap ratio.")
    parser.add_argument("--slice-thresh-w", type=int, default=1024, help="Slice if width exceeds this.")
    parser.add_argument("--slice-thresh-h", type=int, default=256, help="Slice if height exceeds this.")
    return parser.parse_args()


def collect_images(source: str) -> list[Path]:
    src = Path(source)
    if src.is_file():
        return [src]
    if src.is_dir():
        return sorted([p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS])
    return sorted([Path(p) for p in glob.glob(source) if Path(p).suffix.lower() in IMG_EXTS])


def build_starts(size: int, tile: int, stride: int) -> list[int]:
    if size <= tile:
        return [0]
    starts = list(range(0, size - tile + 1, stride))
    end = size - tile
    if starts[-1] != end:
        starts.append(end)
    return starts


def iter_tiles(img: np.ndarray, tile_w: int, tile_h: int, overlap: float):
    h, w = img.shape[:2]
    tile_w = min(tile_w, w)
    tile_h = min(tile_h, h)
    stride_w = max(1, int(tile_w * (1 - overlap)))
    stride_h = max(1, int(tile_h * (1 - overlap)))
    x_starts = build_starts(w, tile_w, stride_w)
    y_starts = build_starts(h, tile_h, stride_h)
    for y0 in y_starts:
        for x0 in x_starts:
            tile = img[y0 : y0 + tile_h, x0 : x0 + tile_w]
            yield tile, x0, y0


def to_uint8_vis(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        vis = img
    else:
        data = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        vmin = float(data.min())
        vmax = float(data.max())
        if vmax > vmin:
            vis = ((data - vmin) / (vmax - vmin) * 255).clip(0, 255).astype(np.uint8)
        else:
            vis = np.zeros_like(data, dtype=np.uint8)
    if vis.ndim == 2:
        vis = np.repeat(vis[..., None], 3, axis=2)
    elif vis.shape[2] == 1:
        vis = np.repeat(vis, 3, axis=2)
    elif vis.shape[2] > 3:
        vis = vis[..., :3]
    return vis


def color_for_class(cls_id: int) -> tuple[int, int, int]:
    rng = np.random.RandomState(cls_id)
    return tuple(int(x) for x in rng.randint(0, 255, size=3).tolist())


def draw_boxes(
    img: np.ndarray, boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, names: dict, thickness: int
):
    for box, score, cls_id in zip(boxes, scores, classes):
        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
        color = color_for_class(int(cls_id))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        label = f"{names.get(int(cls_id), cls_id)} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(y1, th + 2)
        cv2.rectangle(img, (x1, y_text - th - 2), (x1 + tw + 2, y_text), color, -1)
        cv2.putText(img, label, (x1 + 1, y_text - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def class_aware_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou_thres: float,
    max_det: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if boxes.size == 0:
        return boxes, scores, classes
    boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
    scores_t = torch.as_tensor(scores, dtype=torch.float32)
    classes_t = torch.as_tensor(classes, dtype=torch.int64)
    keep_all = []
    for cls_id in classes_t.unique():
        cls_mask = classes_t == cls_id
        idx = cls_mask.nonzero(as_tuple=False).squeeze(1)
        keep = TorchNMS.nms(boxes_t[idx], scores_t[idx], iou_thres)
        keep_all.append(idx[keep])
    keep = torch.cat(keep_all) if keep_all else torch.empty((0,), dtype=torch.int64)
    if keep.numel():
        keep = keep[scores_t[keep].argsort(descending=True)]
    if max_det:
        keep = keep[:max_det]
    return boxes_t[keep].cpu().numpy(), scores_t[keep].cpu().numpy(), classes_t[keep].cpu().numpy()


def save_yolo_labels(path: Path, boxes: np.ndarray, classes: np.ndarray, img_w: int, img_h: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for box, cls_id in zip(boxes, classes):
            x1, y1, x2, y2 = box.tolist()
            xc = (x1 + x2) / 2 / img_w
            yc = (y1 + y2) / 2 / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            f.write(f"{int(cls_id)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def main() -> None:
    args = parse_args()
    images = collect_images(args.source)
    if not images:
        raise FileNotFoundError(f"No images found in source: {args.source}")
    print(f"Found {len(images)} images. Output: {args.output}")

    model = YOLO(args.model)
    predict_kwargs = {
        "conf": args.conf,
        "iou": args.iou,
        "classes": args.classes,
        "device": args.device,
        "half": args.half,
        "verbose": False,
    }
    if args.imgsz is not None:
        predict_kwargs["imgsz"] = args.imgsz
    if args.normalize_mean is not None:
        predict_kwargs["normalize_mean"] = args.normalize_mean
    if args.normalize_std is not None:
        predict_kwargs["normalize_std"] = args.normalize_std
    if args.padding_value is not None:
        predict_kwargs["padding_value"] = args.padding_value

    out_dir = Path(args.output)
    img_dir = out_dir / "images"
    label_dir = out_dir / "labels"
    if args.save_img:
        img_dir.mkdir(parents=True, exist_ok=True)
    if args.save_txt:
        label_dir.mkdir(parents=True, exist_ok=True)

    if args.channels == 1:
        imread_flag = cv2.IMREAD_GRAYSCALE
    elif args.channels == 3:
        imread_flag = cv2.IMREAD_COLOR
    else:
        imread_flag = cv2.IMREAD_UNCHANGED

    for idx, img_path in enumerate(images, start=1):
        img = imread(str(img_path), flags=imread_flag)
        if img is None:
            print(f"[{idx}/{len(images)}] {img_path.name}: read failed")
            continue
        h, w = img.shape[:2]
        use_slices = w > args.slice_thresh_w or h > args.slice_thresh_h
        tiles = [(img, 0, 0)] if not use_slices else list(iter_tiles(img, args.tile_w, args.tile_h, args.overlap))

        all_boxes = []
        all_scores = []
        all_classes = []
        for tile, x0, y0 in tiles:
            results = model.predict(
                source=tile,
                **predict_kwargs,
            )
            r0 = results[0]
            if r0.boxes is None or r0.boxes.xyxy.numel() == 0:
                continue
            boxes = r0.boxes.xyxy.cpu().numpy()
            boxes[:, [0, 2]] += x0
            boxes[:, [1, 3]] += y0
            all_boxes.append(boxes)
            all_scores.append(r0.boxes.conf.cpu().numpy())
            all_classes.append(r0.boxes.cls.cpu().numpy())

        if all_boxes:
            boxes = np.concatenate(all_boxes, axis=0)
            scores = np.concatenate(all_scores, axis=0)
            classes = np.concatenate(all_classes, axis=0).astype(np.int64)
            boxes, scores, classes = class_aware_nms(boxes, scores, classes, args.iou, args.max_det)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
            classes = np.zeros((0,), dtype=np.int64)

        if args.save_txt:
            save_yolo_labels(label_dir / f"{img_path.stem}.txt", boxes, classes, w, h)

        if args.save_img:
            vis = to_uint8_vis(img.copy())
            draw_boxes(vis, boxes, scores, classes, model.names, args.line_thickness)
            out_path = img_dir / f"{img_path.stem}.jpg"
            cv2.imwrite(str(out_path), vis)
        print(
            f"[{idx}/{len(images)}] {img_path.name} {w}x{h} "
            f"{'sliced' if use_slices else 'full'} tiles={len(tiles)} dets={len(boxes)}"
        )

    print("Done.")


if __name__ == "__main__":
    main()
