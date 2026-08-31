"""Run detector inference on complete images without SAHI tiling.

Examples:
    python inference.py --image data/example.jpg --weights models/best.pt
    python inference.py --image data/example.jpg --weights models/rf-detr-base.pth
    python inference.py --image data/images --weights models/best.pt --conf-threshold 0.4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:  # Supports both ``python inference.py`` and ``python -m fruit_pipeline.inference``.
    from fruit_pipeline.paths import resolve_model_path
    from fruit_pipeline.rfdetr_diagnostics import run_diagnostic_sweep
except ModuleNotFoundError:
    from paths import resolve_model_path
    from rfdetr_diagnostics import run_diagnostic_sweep


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BACKENDS = ("auto", "ultralytics", "rfdetr")


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


def ultralytics_result_to_records(result) -> list[dict]:
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


def rfdetr_result_to_records(detections, class_names) -> list[dict]:
    """Convert ``supervision.Detections`` returned by RF-DETR to records."""
    if detections is None or len(detections) == 0:
        return []

    records = []
    for box, score, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
        class_id = int(class_id)
        if isinstance(class_names, dict):
            class_name = class_names.get(class_id, str(class_id))
        elif class_names is not None and 0 <= class_id < len(class_names):
            class_name = class_names[class_id]
        else:
            class_name = str(class_id)
        records.append(
            {
                "box_xyxy": [round(float(value), 3) for value in box],
                "confidence": round(float(score), 6),
                "class_id": class_id,
                "class_name": class_name,
            }
        )
    return records


def select_backend(requested_backend: str, weights: str) -> str:
    """Infer the detector family from a checkpoint name when requested."""
    if requested_backend != "auto":
        return requested_backend
    path = Path(weights)
    if path.suffix.lower() in {".pth", ".ckpt"} or path.name.lower().startswith("rf-detr"):
        return "rfdetr"
    return "ultralytics"


def load_rfdetr_model(weights_path: str, device: str = "auto"):
    """Load a published RF-DETR detection checkpoint.

    Canonical checkpoint names are mapped to their model class explicitly,
    which works with both older and current releases of the ``rfdetr``
    package. Arbitrarily named fine-tuned checkpoints use the newer
    ``from_checkpoint`` API when available.
    """
    try:
        import rfdetr
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "RF-DETR inference requires the 'rfdetr' package. "
            "Install it with: python -m pip install rfdetr"
        ) from exc

    resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if resolved_device == "auto":
        resolved_device = "cpu"

    name = Path(weights_path).name.lower()
    variants = (
        ("nano", "RFDETRNano"),
        ("small", "RFDETRSmall"),
        ("medium", "RFDETRMedium"),
        ("large", "RFDETRLarge"),
        ("base", "RFDETRBase"),
    )
    for marker, class_name in variants:
        if marker in name:
            model_class = getattr(rfdetr, class_name, None)
            if model_class is None:
                raise RuntimeError(f"Installed rfdetr package does not provide {class_name}")
            return model_class(pretrain_weights=weights_path, device=resolved_device)

    model_class = getattr(rfdetr, "RFDETR", None)
    if model_class is not None and hasattr(model_class, "from_checkpoint"):
        return model_class.from_checkpoint(weights_path, device=resolved_device)
    raise ValueError(
        "Could not infer the RF-DETR variant from the checkpoint filename. "
        "Use a canonical name such as rf-detr-base.pth."
    )


def annotate_records(image, records: list[dict]):
    """Draw backend-independent detection records on a BGR image."""
    import cv2

    annotated = image.copy()
    for record in records:
        x1, y1, x2, y2 = (int(round(value)) for value in record["box_xyxy"])
        label = f'{record["class_name"]} {record["confidence"]:.2f}'
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 220, 0),
            1,
            cv2.LINE_AA,
        )
    return annotated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run whole-image detector inference (no tiling).")
    parser.add_argument("--image", required=True, help="Input image or directory of images (non-recursive).")
    parser.add_argument(
        "--weights",
        "--detector-weights",
        dest="weights",
        default="models/yolo11x.pt",
        help="Ultralytics .pt or RF-DETR .pth/.ckpt checkpoint (default: models/yolo11x.pt).",
    )
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default="auto",
        help="Detector backend; auto selects RF-DETR for .pth/.ckpt and Ultralytics otherwise (default: auto).",
    )
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default="outputs/whole_image")
    parser.add_argument(
        "--conf-threshold",
        type=confidence_threshold,
        default=0.25,
        help="Minimum detection confidence, from 0 to 1 (default: 0.25).",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N (default: auto).")
    parser.add_argument("--imgsz", type=int, default=640, help="Ultralytics inference image size (ignored by RF-DETR).")
    parser.add_argument(
        "--rfdetr-diagnostic-sweep",
        action="store_true",
        help="RF-DETR only: run the label-free num_select/threshold/resolution sweep and exit.",
    )
    parser.add_argument(
        "--max-rfdetr-resolution",
        type=int,
        default=None,
        help="Diagnostic sweep ceiling. Default: 2x the model's runtime native resolution; stops early on OOM.",
    )
    parser.add_argument(
        "--rfdetr-resolution-step",
        type=int,
        default=None,
        help="Diagnostic resolution increment. Default: the model's actual patch_size*num_windows divisor.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        images = find_images(Path(args.image))
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    import cv2

    weights_path = resolve_model_path(args.weights)
    backend = select_backend(args.backend, weights_path)
    if backend == "ultralytics":
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise SystemExit(
                "Ultralytics inference requires the 'ultralytics' package. "
                "Install it with: python -m pip install ultralytics"
            ) from exc
        model = YOLO(weights_path)
    else:
        try:
            model = load_rfdetr_model(weights_path, device=args.device)
        except (RuntimeError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = None if args.device == "auto" else args.device

    if args.rfdetr_diagnostic_sweep:
        if backend != "rfdetr":
            raise SystemExit("--rfdetr-diagnostic-sweep requires --backend rfdetr or an RF-DETR checkpoint")
        try:
            manifest = run_diagnostic_sweep(
                model=model,
                images=images,
                output_dir=output_dir,
                baseline_threshold=args.conf_threshold,
                max_resolution=args.max_rfdetr_resolution,
                resolution_step=args.rfdetr_resolution_step,
            )
        except (AssertionError, RuntimeError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        print(manifest["verdict"])
        print(f"Saved diagnostic artifacts to {output_dir}")
        return

    total = 0
    for index, image_path in enumerate(images, start=1):
        # In both backends this is the only prediction call: the complete
        # source image is passed directly, with no crops or SAHI slicing.
        if backend == "ultralytics":
            result = model.predict(
                source=str(image_path),
                conf=args.conf_threshold,
                imgsz=args.imgsz,
                device=device,
                verbose=False,
            )[0]
            records = ultralytics_result_to_records(result)
            annotated = result.plot()
        else:
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                raise RuntimeError(f"Could not read image: {image_path}")
            detections = model.predict(str(image_path), threshold=args.conf_threshold)
            records = rfdetr_result_to_records(detections, getattr(model, "class_names", None))
            annotated = annotate_records(image_bgr, records)
        total += len(records)

        annotated_path = output_dir / f"{image_path.stem}_prediction.jpg"
        json_path = output_dir / f"{image_path.stem}_detections.json"
        if not cv2.imwrite(str(annotated_path), annotated):
            raise RuntimeError(f"Could not write annotated image: {annotated_path}")
        json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"[{index}/{len(images)}] {image_path.name}: {len(records)} detections")

    print(f"Saved {total} detections for {len(images)} image(s) to {output_dir}")


if __name__ == "__main__":
    main()
