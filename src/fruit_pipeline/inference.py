"""Run Ultralytics inference on complete images without SAHI tiling.

Examples:
    python -m fruit_pipeline.inference --image data/example.jpg --weights models/best.pt
    python -m fruit_pipeline.inference --image data/images --weights models/best.pt --conf-threshold 0.4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fruit_pipeline.utils.paths import resolve_model_path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def confidence_threshold(value: str) -> float:
    """Parse a confidence threshold and give argparse a useful error."""
    threshold = float(value)
    if not 0.0 <= threshold <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return threshold


def find_images(image: Path) -> list[Path]:
    """Return one image, or all supported images directly inside a directory."""
    if image.is_file():
        if image.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {image.suffix}")
        return [image]
    if image.is_dir():
        images = sorted(path for path in image.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            raise ValueError(f"No supported images found directly inside {image}")
        return images
    raise FileNotFoundError(f"Image path does not exist: {image}")


def result_to_records(result) -> list[dict]:
    """Convert an Ultralytics result to JSON-safe detection records."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.detach().cpu().tolist()
    scores = boxes.conf.detach().cpu().tolist()
    classes = boxes.cls.detach().cpu().tolist()
    names = result.names
    polygons = result.masks.xy if result.masks is not None else None

    records = []
    for index, (box, score, class_id) in enumerate(zip(xyxy, scores, classes)):
        class_id = int(class_id)
        record = {
            "box_xyxy": [round(float(value), 3) for value in box],
            "confidence": round(float(score), 6),
            "class_id": class_id,
            "class_name": names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id],
        }
        if polygons is not None and index < len(polygons):
            record["polygon"] = [[round(float(x), 3), round(float(y), 3)] for x, y in polygons[index]]
        records.append(record)
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run whole-image Ultralytics inference (no tiling).")
    parser.add_argument("--image", required=True, help="Input image or directory of images (non-recursive).")
    parser.add_argument(
        "--weights",
        "--detector-weights",
        dest="weights",
        default="models/yolo11x.pt",
        help="Ultralytics .pt checkpoint (default: models/yolo11x.pt).",
    )
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default="outputs/whole_image")
    parser.add_argument(
        "--conf-threshold",
        type=confidence_threshold,
        default=0.25,
        help="Minimum detection confidence, from 0 to 1 (default: 0.25).",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N (default: auto).")
    parser.add_argument("--imgsz", type=int, default=640, help="Ultralytics inference image size (default: 640).")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        images = find_images(Path(args.image))
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    import cv2
    from ultralytics import YOLO

    model = YOLO(resolve_model_path(args.weights))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = None if args.device == "auto" else args.device

    total = 0
    for index, image_path in enumerate(images, start=1):
        # This is the only prediction call: the entire source image is passed
        # directly to Ultralytics, with no crops or SAHI slicing.
        result = model.predict(
            source=str(image_path),
            conf=args.conf_threshold,
            imgsz=args.imgsz,
            device=device,
            verbose=False,
        )[0]
        records = result_to_records(result)
        total += len(records)

        annotated_path = output_dir / f"{image_path.stem}_prediction.jpg"
        json_path = output_dir / f"{image_path.stem}_detections.json"
        if not cv2.imwrite(str(annotated_path), result.plot()):
            raise RuntimeError(f"Could not write annotated image: {annotated_path}")
        json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"[{index}/{len(images)}] {image_path.name}: {len(records)} detections")

    print(f"Saved {total} detections for {len(images)} image(s) to {output_dir}")


if __name__ == "__main__":
    main()
