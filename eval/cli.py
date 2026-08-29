"""CLI entrypoint: ``python -m fruit_pipeline.eval run|compare``."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from fruit_pipeline.eval.coco_io import load_coco_gt, load_predictions_dir
from fruit_pipeline.eval.compare import build_delta_table
from fruit_pipeline.eval.metrics import evaluate_all

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m fruit_pipeline.eval", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Compute mAP/precision/recall/counting metrics vs. a GT COCO JSON.")
    run_parser.add_argument("--gt", required=True, help="Path to a ground-truth COCO JSON.")
    run_parser.add_argument(
        "--pred-dir",
        required=True,
        help="Directory of '<stem>_detections.json' files (pipeline.py output), matched to GT images by filename stem.",
    )
    run_parser.add_argument("--output", required=True, help="Where to write the result JSON.")
    run_parser.add_argument("--iou-thresh", type=float, default=0.5, help="IoU threshold for precision/recall matching (default 0.5).")
    run_parser.add_argument("--label", default=None, help="Optional label for this run, stored in the result JSON (e.g. a config name).")

    compare_parser = subparsers.add_parser("compare", help="Print a side-by-side delta table between two 'run' outputs.")
    compare_parser.add_argument("result_a", help="Path to the first result JSON.")
    compare_parser.add_argument("result_b", help="Path to the second result JSON.")

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    return parser


def run_command(args: argparse.Namespace) -> None:
    coco_gt = load_coco_gt(args.gt)
    coco_results, unmatched = load_predictions_dir(coco_gt, args.pred_dir)

    metrics = evaluate_all(coco_gt, coco_results, iou_thresh=args.iou_thresh)

    payload = {
        "label": args.label,
        "gt_path": str(args.gt),
        "pred_dir": str(args.pred_dir),
        "iou_thresh": args.iou_thresh,
        "num_images": len(coco_gt.imgs),
        "num_predictions": len(coco_results),
        "unmatched_prediction_stems": unmatched,
        "metrics": metrics,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    logger.info("Wrote metrics for %d image(s), %d prediction(s) to %s", payload["num_images"], payload["num_predictions"], output_path)
    print(f"all: mAP50={metrics['all']['mAP50']:.4f}  mAP50-95={metrics['all']['mAP50-95']:.4f}  "
          f"precision={metrics['all']['precision']}  recall={metrics['all']['recall']}  "
          f"MAE={metrics['all']['mae']}  RMSE={metrics['all']['rmse']}")


def compare_command(args: argparse.Namespace) -> None:
    with open(args.result_a) as f:
        result_a = json.load(f)
    with open(args.result_b) as f:
        result_b = json.load(f)

    label_a = result_a.get("label") or Path(args.result_a).stem
    label_b = result_b.get("label") or Path(args.result_b).stem
    print(build_delta_table(result_a, result_b, label_a=label_a, label_b=label_b))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if args.command == "run":
        run_command(args)
    elif args.command == "compare":
        compare_command(args)


if __name__ == "__main__":
    main()
