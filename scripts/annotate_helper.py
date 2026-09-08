#!/usr/bin/env python3
"""Turn pipeline.py output into pre-annotations for a human-correction pass.

Bootstraps a ground-truth set for ``eval/`` without labelling from scratch:
read a directory of ``<stem>_detections.json`` (pipeline.py's own output,
which already carries each image's width/height and every detection's box +
score, so no re-reading of the source images is needed), and emit either

- ``--format coco``: a COCO JSON importable into a CVAT task created from
  the same images ("Upload annotations" -> COCO 1.0), or
- ``--format label-studio``: a Label Studio pre-annotation task JSON
  (percent-based ``rectanglelabels``, per Label Studio's prediction-import
  spec), so predictions show up ready-to-correct instead of drawn from
  scratch.

Either way, every detection carries its original detector score as an extra
field the annotation tool ignores but that's handy for spotting low-
confidence boxes worth extra scrutiny during correction.

Usage (run from fruit-pipeline/):
    python scripts/annotate_helper.py \\
        --pred-dir outputs/test1 --format coco --output gt_bootstrap.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

CATEGORY_NAME = "fruit"
CATEGORY_ID = 1


def _iter_detection_files(pred_dir: str):
    for path in sorted(Path(pred_dir).glob("*_detections.json")):
        with open(path) as f:
            yield path, json.load(f)


def build_coco_pre_annotations(pred_dir: str) -> dict:
    """Build a COCO JSON (images + un-reviewed annotations) from pipeline.py output."""
    images: list[dict] = []
    annotations: list[dict] = []
    next_ann_id = 1

    for image_id, (path, payload) in enumerate(_iter_detection_files(pred_dir), start=1):
        file_name = Path(payload["image_path"]).name
        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": payload["image_width"],
                "height": payload["image_height"],
            }
        )
        for det in payload.get("detections", []):
            x1, y1, x2, y2 = det["box"]
            annotations.append(
                {
                    "id": next_ann_id,
                    "image_id": image_id,
                    "category_id": CATEGORY_ID,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": max(0.0, x2 - x1) * max(0.0, y2 - y1),
                    "iscrowd": 0,
                    "score": det.get("detector_score"),  # kept for reference; ignored by COCO consumers
                }
            )
            next_ann_id += 1

    if not images:
        logger.warning("No '*_detections.json' files found in %s", pred_dir)

    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": CATEGORY_ID, "name": CATEGORY_NAME}],
    }


def build_label_studio_tasks(pred_dir: str, image_url_prefix: str = "") -> list[dict]:
    """Build Label Studio pre-annotation tasks (percent-based rectanglelabels) from pipeline.py output."""
    tasks: list[dict] = []

    for path, payload in _iter_detection_files(pred_dir):
        width, height = payload["image_width"], payload["image_height"]
        file_name = Path(payload["image_path"]).name
        results = []
        for i, det in enumerate(payload.get("detections", [])):
            x1, y1, x2, y2 = det["box"]
            results.append(
                {
                    "id": f"det_{i}",
                    "type": "rectanglelabels",
                    "from_name": "label",
                    "to_name": "image",
                    "original_width": width,
                    "original_height": height,
                    "image_rotation": 0,
                    "score": det.get("detector_score"),
                    "value": {
                        "x": 100.0 * x1 / width,
                        "y": 100.0 * y1 / height,
                        "width": 100.0 * (x2 - x1) / width,
                        "height": 100.0 * (y2 - y1) / height,
                        "rotation": 0,
                        "rectanglelabels": [CATEGORY_NAME],
                    },
                }
            )
        tasks.append(
            {
                "data": {"image": f"{image_url_prefix}{file_name}"},
                "predictions": [{"model_version": "fruit_pipeline", "result": results}],
            }
        )

    if not tasks:
        logger.warning("No '*_detections.json' files found in %s", pred_dir)

    return tasks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred-dir", required=True, help="Directory of '<stem>_detections.json' files (pipeline.py output).")
    parser.add_argument("--format", choices=["coco", "label-studio"], default="coco", help="Output format (default: coco).")
    parser.add_argument("--output", required=True, help="Path to write the pre-annotation file.")
    parser.add_argument(
        "--image-url-prefix",
        default="",
        help="--format label-studio only: prefix prepended to each image's basename in task data "
        "(e.g. a Label Studio local-storage URL like '/data/local-files/?d=my_images/&f=').",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = build_parser().parse_args()

    if args.format == "coco":
        payload = build_coco_pre_annotations(args.pred_dir)
        n_images, n_annotations = len(payload["images"]), len(payload["annotations"])
    else:
        payload = build_label_studio_tasks(args.pred_dir, image_url_prefix=args.image_url_prefix)
        n_images = len(payload)
        n_annotations = sum(len(t["predictions"][0]["result"]) for t in payload)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    logger.info(
        "Wrote %s pre-annotations for %d image(s), %d detection(s) to %s",
        args.format,
        n_images,
        n_annotations,
        output_path,
    )


if __name__ == "__main__":
    main()
