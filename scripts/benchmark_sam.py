#!/usr/bin/env python3
"""Compare cold FP32 SAM inference with persistent optimized inference."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# Keep the utility runnable from a source checkout before an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fruit_pipeline.segmentation.sam_manager import SAMModelManager


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Input image (kept at its existing resolution).")
    parser.add_argument("--checkpoint", default="models/sam_vit_l_0b3195.pth")
    parser.add_argument("--model-type", choices=("vit_b", "vit_l", "vit_h"), default="vit_l")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--boxes-json",
        help="JSON list of [x1,y1,x2,y2], or a pipeline detections JSON. Default: full image.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", help="Optional path for the JSON report.")
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Profile the optimized path in FP32 (FP16 is enabled by default).",
    )
    return parser


def _load_boxes(path: str | None, width: int, height: int) -> np.ndarray:
    if path is None:
        return np.asarray([[0, 0, width, height]], dtype=np.float32)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = [item["box"] for item in payload["detections"]]
    boxes = np.asarray(payload, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes JSON must contain an Nx4 array")
    return boxes


def _average(samples: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: statistics.fmean(sample[key] for sample in samples)
        for key in samples[0]
    }


def main() -> int:
    args = _parser().parse_args()
    image_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")
    # cvtColor allocates the one required RGB input; neither benchmark path
    # creates another full-resolution host copy.
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    boxes = _load_boxes(args.boxes_json, image_rgb.shape[1], image_rgb.shape[0])

    current_samples: list[dict[str, float]] = []
    current_masks = current_scores = None
    for iteration in range(args.iterations):
        manager = SAMModelManager(
            args.checkpoint,
            args.model_type,
            args.device,
            use_fp16=False,
            profile=True,
        )
        manager.load_model()
        result = manager.run_inference(image_rgb, boxes, batch_size=args.batch_size)
        sample = result.timings.to_dict()
        sample["cold_request_ms"] = sample["model_loading_ms"] + sample["total_inference_ms"]
        current_samples.append(sample)
        if iteration == 0:
            current_masks = result.masks.copy()
            current_scores = result.scores.copy()
        del result, manager
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    optimized = SAMModelManager(
        args.checkpoint,
        args.model_type,
        args.device,
        use_fp16=not args.no_fp16,
        profile=True,
    )
    optimized.load_model()
    for _ in range(args.warmup):
        optimized.run_inference(image_rgb, boxes, batch_size=args.batch_size)
    optimized_samples = []
    optimized_masks = optimized_scores = None
    for iteration in range(args.iterations):
        result = optimized.run_inference(image_rgb, boxes, batch_size=args.batch_size)
        sample = result.timings.to_dict()
        sample["cold_request_ms"] = sample["total_inference_ms"]
        optimized_samples.append(sample)
        if iteration == 0:
            optimized_masks = result.masks.copy()
            optimized_scores = result.scores.copy()

    current = _average(current_samples)
    optimized_result = _average(optimized_samples)
    intersections = np.count_nonzero(current_masks & optimized_masks, axis=(1, 2))
    unions = np.count_nonzero(current_masks | optimized_masks, axis=(1, 2))
    mask_ious = np.divide(
        intersections,
        unions,
        out=np.ones_like(intersections, dtype=np.float64),
        where=unions != 0,
    )
    report = {
        "configuration": {
            "image": str(Path(args.image).resolve()),
            "image_size": [image_rgb.shape[1], image_rgb.shape[0]],
            "box_count": len(boxes),
            "model_type": args.model_type,
            "device": args.device,
            "optimized_fp16": optimized.use_fp16,
            "iterations": args.iterations,
            "warmup": args.warmup,
        },
        "current_cold_fp32": current,
        "optimized_persistent": optimized_result,
        "speedup": {
            "inference": current["total_inference_ms"] / optimized_result["total_inference_ms"],
            "request_including_load": current["cold_request_ms"] / optimized_result["cold_request_ms"],
        },
        "output_equivalence": {
            "mean_mask_iou": float(mask_ious.mean()),
            "pixel_agreement": float(np.mean(current_masks == optimized_masks)),
            "max_score_absolute_difference": float(np.max(np.abs(current_scores - optimized_scores))),
        },
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
