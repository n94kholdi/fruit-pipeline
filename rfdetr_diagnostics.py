"""Label-free, single-pass RF-DETR collapse diagnostics and config sweeps.

This module intentionally does not import RF-DETR or torch at import time.  The
pure geometry/statistics helpers remain unit-testable without the optional
inference dependency, while :func:`run_diagnostic_sweep` operates on an already
loaded RF-DETR model.
"""

from __future__ import annotations

import csv
import inspect
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np


SCORE_THRESHOLDS = (0.5, 0.3, 0.2, 0.1, 0.05)
NUM_SELECT_REQUESTS: tuple[int | str, ...] = (300, 600, 900, "num_queries")
DECILES = tuple(range(0, 101, 10))


@dataclass(frozen=True)
class RuntimeConfig:
    """Values read from the instantiated model, never package defaults."""

    num_queries: int
    num_select: int
    native_resolution: int
    patch_size: int
    num_windows: int

    @property
    def resolution_divisor(self) -> int:
        return self.patch_size * self.num_windows


@dataclass(frozen=True)
class SweepConfig:
    name: str
    axis: str
    requested_value: int | float | str
    num_select: int
    threshold: float
    resolution: int


def _runtime_module(model: Any) -> Any:
    context = getattr(model, "model", None)
    if context is None:
        raise AssertionError("RF-DETR model has no runtime model context")
    optimized = getattr(context, "inference_model", None)
    base = getattr(context, "model", None)
    module = optimized if getattr(model, "_is_optimized_for_inference", False) else base
    if module is None:
        module = optimized or base
    if module is None:
        raise AssertionError("RF-DETR model context has no reachable inference module")
    return module


def read_runtime_config(model: Any) -> RuntimeConfig:
    """Read active query, postprocess, and resolution values from *model*."""

    context = getattr(model, "model", None)
    module = _runtime_module(model)
    postprocess = getattr(context, "postprocess", None)
    model_config = getattr(model, "model_config", None)

    num_queries = getattr(module, "num_queries", None)
    num_select = getattr(postprocess, "num_select", None)
    native_resolution = getattr(context, "resolution", None)
    patch_size = getattr(model_config, "patch_size", None)
    num_windows = getattr(model_config, "num_windows", None)
    values = {
        "num_queries": num_queries,
        "num_select": num_select,
        "native_resolution": native_resolution,
        "patch_size": patch_size,
        "num_windows": num_windows,
    }
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise AssertionError(f"Could not read RF-DETR runtime value(s): {', '.join(missing)}")

    config = RuntimeConfig(**{name: int(value) for name, value in values.items()})
    assert config.num_queries > 0, f"Invalid runtime num_queries={config.num_queries}"
    assert config.num_select >= 0, f"Invalid runtime num_select={config.num_select}"
    assert config.native_resolution > 0, f"Invalid runtime resolution={config.native_resolution}"
    assert config.resolution_divisor > 0
    assert config.native_resolution % config.resolution_divisor == 0, (
        f"RF-DETR runtime resolution {config.native_resolution} is not divisible by its actual "
        f"patch_size*num_windows divisor {config.resolution_divisor}"
    )
    return config


def assert_nms_unreachable(model: Any) -> None:
    """Assert that no NMS module/op is present on the RF-DETR predict path.

    RF-DETR's path consists of the public ``predict`` method, its postprocessor,
    and the active torch module graph.  We inspect those exact runtime objects;
    unrelated utilities elsewhere in an installed package are deliberately not
    searched.
    """

    context = getattr(model, "model", None)
    module = _runtime_module(model)
    postprocess = getattr(context, "postprocess", None)
    hits: list[str] = []
    forbidden = ("non_max_suppression", "nonmaxsuppression", "torchvision::nms", "ops.nms")

    callables = (("predict", getattr(model, "predict", None)), ("postprocess", getattr(postprocess, "forward", None)))
    for label, function in callables:
        if function is None:
            hits.append(f"missing {label} callable")
            continue
        try:
            source = inspect.getsource(function).lower()
        except (OSError, TypeError):
            source = ""
        hits.extend(f"{label} source contains {token}" for token in forbidden if token in source)

    named_modules = getattr(module, "named_modules", None)
    if callable(named_modules):
        for name, child in named_modules():
            identity = f"{name} {type(child).__module__}.{type(child).__qualname__}".lower()
            if any(token in identity for token in ("nonmax", "non_max", ".nms", "::nms")):
                hits.append(f"module graph: {identity}")

    assert not hits, "NMS is reachable from RF-DETR inference: " + "; ".join(hits)


def assert_predictions_not_capped(prediction_count: int, num_select: int) -> None:
    """Loud cap assertion; callers may record it and continue the sweep."""

    assert prediction_count != num_select, (
        f"CAP BINDING: decoded {prediction_count} predictions == runtime num_select={num_select}; "
        "this image is truncated and its pre-threshold result is untrustworthy"
    )


def _deciles(values: Sequence[float] | np.ndarray, prefix: str) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {f"{prefix}_p{percent:03d}": math.nan for percent in DECILES}
    quantiles = np.percentile(array, DECILES)
    return {f"{prefix}_p{percent:03d}": float(value) for percent, value in zip(DECILES, quantiles)}


def score_statistics(scores: Sequence[float] | np.ndarray) -> dict[str, float]:
    stats = _deciles(scores, "score")
    stats["score_min"] = stats["score_p000"]
    stats["score_max"] = stats["score_p100"]
    return stats


def box_diagnostics(boxes: Sequence[Sequence[float]] | np.ndarray) -> dict[str, Any]:
    """Compute containment, relative-area distribution, and largest-box gap."""

    array = np.asarray(boxes, dtype=float).reshape((-1, 4))
    empty = {
        "containment_counts": [],
        "max_containment_count": 0,
        "total_contained_centers": 0,
        "collapse_candidate_count": 0,
        "area_ratios": [],
        "coverage_gap": math.nan,
        "largest_box_index": None,
    }
    if len(array) == 0:
        return {**empty, **_deciles([], "area_ratio")}

    widths = np.maximum(0.0, array[:, 2] - array[:, 0])
    heights = np.maximum(0.0, array[:, 3] - array[:, 1])
    areas = widths * heights
    centres = np.column_stack(((array[:, 0] + array[:, 2]) / 2, (array[:, 1] + array[:, 3]) / 2))
    inside = (
        (centres[None, :, 0] >= array[:, None, 0])
        & (centres[None, :, 0] <= array[:, None, 2])
        & (centres[None, :, 1] >= array[:, None, 1])
        & (centres[None, :, 1] <= array[:, None, 3])
    )
    np.fill_diagonal(inside, False)
    containment = inside.sum(axis=1).astype(int)

    positive_areas = areas[areas > 0]
    median_area = float(np.median(positive_areas)) if positive_areas.size else math.nan
    ratios = areas / median_area if median_area > 0 else np.full(len(areas), math.nan)
    largest_index = int(np.argmax(areas))
    largest = array[largest_index]
    smaller = array[areas < areas[largest_index]]
    covered_area = rectangle_union_area(_clip_boxes(smaller, largest))
    largest_area = float(areas[largest_index])
    gap = max(0.0, min(1.0, 1.0 - covered_area / largest_area)) if largest_area > 0 else math.nan

    return {
        "containment_counts": containment.tolist(),
        "max_containment_count": int(containment.max(initial=0)),
        "total_contained_centers": int(containment.sum()),
        "collapse_candidate_count": int((containment >= 2).sum()),
        "area_ratios": ratios.tolist(),
        "coverage_gap": gap,
        "largest_box_index": largest_index,
        **_deciles(ratios, "area_ratio"),
    }


def _clip_boxes(boxes: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.empty((0, 4), dtype=float)
    clipped = boxes.copy()
    clipped[:, [0, 2]] = np.clip(clipped[:, [0, 2]], boundary[0], boundary[2])
    clipped[:, [1, 3]] = np.clip(clipped[:, [1, 3]], boundary[1], boundary[3])
    valid = (clipped[:, 2] > clipped[:, 0]) & (clipped[:, 3] > clipped[:, 1])
    return clipped[valid]


def rectangle_union_area(boxes: Sequence[Sequence[float]] | np.ndarray) -> float:
    """Exact union area for axis-aligned rectangles."""

    array = np.asarray(boxes, dtype=float).reshape((-1, 4))
    if not len(array):
        return 0.0
    xs = np.unique(array[:, [0, 2]])
    area = 0.0
    for left, right in zip(xs[:-1], xs[1:]):
        if right <= left:
            continue
        active = array[(array[:, 0] < right) & (array[:, 2] > left)]
        intervals = sorted((float(row[1]), float(row[3])) for row in active if row[3] > row[1])
        if not intervals:
            continue
        covered_y = 0.0
        start, end = intervals[0]
        for next_start, next_end in intervals[1:]:
            if next_start > end:
                covered_y += end - start
                start, end = next_start, next_end
            else:
                end = max(end, next_end)
        covered_y += end - start
        area += (right - left) * covered_y
    return float(area)


def build_sweep_configs(
    runtime: RuntimeConfig,
    baseline_threshold: float,
    max_resolution: int,
    resolution_step: int | None = None,
) -> list[SweepConfig]:
    """Build a one-variable-at-a-time sweep, retaining clamped requests."""

    divisor = runtime.resolution_divisor
    step = resolution_step or divisor
    if step <= 0 or step % divisor:
        raise ValueError(f"resolution_step must be a positive multiple of {divisor}")
    if max_resolution < runtime.native_resolution:
        max_resolution = runtime.native_resolution
    max_resolution -= max_resolution % divisor

    configs = [
        SweepConfig("baseline", "baseline", "runtime", runtime.num_select, baseline_threshold, runtime.native_resolution)
    ]
    for requested in NUM_SELECT_REQUESTS:
        value = runtime.num_queries if requested == "num_queries" else int(requested)
        actual = min(value, runtime.num_queries)
        configs.append(
            SweepConfig(f"num_select={requested}", "num_select", requested, actual, baseline_threshold, runtime.native_resolution)
        )
    for threshold in SCORE_THRESHOLDS:
        configs.append(
            SweepConfig(
                f"threshold={threshold:g}", "threshold", threshold, runtime.num_select, threshold, runtime.native_resolution
            )
        )
    for resolution in range(runtime.native_resolution, max_resolution + 1, step):
        configs.append(
            SweepConfig(
                f"resolution={resolution}", "resolution", resolution, runtime.num_select, baseline_threshold, resolution
            )
        )
    return configs


@contextmanager
def _capture_input_shape(model: Any) -> Iterator[list[int]]:
    captured: list[int] = []

    def hook(_module: Any, args: tuple[Any, ...]) -> None:
        if args and hasattr(args[0], "shape"):
            captured[:] = [int(value) for value in args[0].shape]

    handle = _runtime_module(model).register_forward_pre_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


def _predict_unthresholded(model: Any, image_path: Path, num_select: int, resolution: int) -> tuple[Any, list[int]]:
    postprocess = model.model.postprocess
    original_num_select = int(postprocess.num_select)
    postprocess.num_select = int(num_select)
    try:
        with _capture_input_shape(model) as input_shape:
            detections = model.predict(
                str(image_path), threshold=-math.inf, shape=(resolution, resolution), include_source_image=False
            )
        assert input_shape, "Could not capture RF-DETR's actual preprocessed input tensor shape"
        return detections, input_shape
    finally:
        postprocess.num_select = original_num_select


def _detections_to_arrays(detections: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if detections is None or len(detections) == 0:
        return np.empty((0, 4), dtype=float), np.empty(0, dtype=float), np.empty(0, dtype=int)
    return (
        np.asarray(detections.xyxy, dtype=float).reshape((-1, 4)),
        np.asarray(detections.confidence, dtype=float),
        np.asarray(detections.class_id, dtype=int),
    )


def _row_for_config(
    config: SweepConfig,
    image_path: Path,
    image_shape: tuple[int, int],
    input_shape: Sequence[int],
    boxes: np.ndarray,
    scores: np.ndarray,
) -> dict[str, Any]:
    keep = scores > config.threshold
    kept_boxes = boxes[keep]
    diagnostics = box_diagnostics(kept_boxes)
    height, width = image_shape
    row: dict[str, Any] = {
        "config": config.name,
        "axis": config.axis,
        "requested_value": config.requested_value,
        "image": str(image_path),
        "num_queries": "",
        "num_select": config.num_select,
        "score_threshold": config.threshold,
        "native_resolution": "",
        "resolution_divisor": "",
        "requested_resolution": config.resolution,
        "original_height": height,
        "original_width": width,
        "input_tensor_shape": "x".join(str(value) for value in input_shape),
        "resized": bool(height != config.resolution or width != config.resolution),
        "resize_factor_y": config.resolution / height,
        "resize_factor_x": config.resolution / width,
        "predictions_before_threshold": len(scores),
        "detections_after_threshold": int(keep.sum()),
        "cap_binding": len(scores) == config.num_select,
        "max_containment_count": diagnostics["max_containment_count"],
        "total_contained_centers": diagnostics["total_contained_centers"],
        "collapse_candidate_count": diagnostics["collapse_candidate_count"],
        "coverage_gap": diagnostics["coverage_gap"],
        **score_statistics(scores),
        **{key: value for key, value in diagnostics.items() if key.startswith("area_ratio_p")},
    }
    return row


def _is_oom(error: BaseException) -> bool:
    return isinstance(error, MemoryError) or "out of memory" in str(error).lower()


def _smaller_centres_inside(boxes: np.ndarray, boundary: np.ndarray) -> int:
    if not len(boxes):
        return 0
    areas = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    boundary_area = max(0.0, boundary[2] - boundary[0]) * max(0.0, boundary[3] - boundary[1])
    centres = np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2))
    inside = (
        (centres[:, 0] >= boundary[0])
        & (centres[:, 0] <= boundary[2])
        & (centres[:, 1] >= boundary[1])
        & (centres[:, 1] <= boundary[3])
    )
    return int((inside & (areas < boundary_area)).sum())


def _select_best_and_verdict(
    configs: Sequence[SweepConfig],
    results: dict[tuple[str, Path], tuple[np.ndarray, np.ndarray]],
    images: Sequence[Path],
) -> tuple[SweepConfig, str, dict[str, int]]:
    baseline = next(config for config in configs if config.axis == "baseline")
    baseline_boundaries: dict[Path, np.ndarray | None] = {}
    baseline_counts: dict[Path, int] = {}
    for image in images:
        boxes, _scores = results.get((baseline.name, image), (np.empty((0, 4)), np.empty(0)))
        if len(boxes):
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            boundary = boxes[int(np.argmax(areas))]
            baseline_boundaries[image] = boundary
            baseline_counts[image] = _smaller_centres_inside(boxes, boundary)
        else:
            baseline_boundaries[image] = None
            baseline_counts[image] = 0

    gains: dict[str, int] = {}
    for config in configs:
        gain = 0
        for image in images:
            boundary = baseline_boundaries[image]
            if boundary is None:
                continue
            boxes, _scores = results.get((config.name, image), (np.empty((0, 4)), np.empty(0)))
            gain += max(0, _smaller_centres_inside(boxes, boundary) - baseline_counts[image])
        gains[config.name] = gain

    best = max(configs, key=lambda config: (gains.get(config.name, 0), -configs.index(config)))
    axis_gains = {
        axis: max((gains[config.name] for config in configs if config.axis == axis), default=0)
        for axis in ("num_select", "threshold", "resolution")
    }
    findings: list[str] = []
    if axis_gains["num_select"] > 0:
        findings.append(
            "(a) Raising num_select surfaced additional smaller predictions inside the baseline collapse box."
        )
    else:
        findings.append(
            "(a) did not improve the result: higher requests were clamped to the checkpoint's runtime num_queries."
        )
    if axis_gains["threshold"] > 0:
        findings.append(
            "(b) is the primary finding: smaller, low-score queries exist inside the baseline collapse box, and "
            "lowering the threshold surfaces them; the lowest setting is diagnostic and also exposes substantial "
            "low-score clutter, so this is not evidence of a production-quality fix."
        )
    if axis_gains["resolution"] > 0:
        findings.append(
            "(c) also occurs on a smaller scale: higher single-pass resolution changes some outputs and surfaces "
            "additional interior boxes, without tiling or merging."
        )
    if axis_gains["threshold"] > 0 or axis_gains["resolution"] > 0 or axis_gains["num_select"] > 0:
        verdict = " ".join(findings)
    else:
        verdict = (
            "(d) None of the inference-only changes recovered smaller predictions inside the collapse box: within "
            "the tested num_select, score, and full-image resolution range, this checkpoint does not emit individual-"
            "fruit queries that inference configuration can recover. Stop here; dense-supervision fine-tuning would "
            "be the next step, and it is outside this phase."
        )
    return best, verdict, axis_gains


def _draw_overlay(image: np.ndarray, boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
    import cv2

    output = image.copy()
    diagnostics = box_diagnostics(boxes)
    containment = diagnostics["containment_counts"]
    for index, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = (int(round(value)) for value in box)
        count = containment[index]
        colour = (0, 0, 255) if count >= 2 else (0, 220, 0)
        cv2.rectangle(output, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(
            output,
            f"{score:.2f} inside={count}",
            (x1, max(16, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            colour,
            1,
            cv2.LINE_AA,
        )
    return output


def _save_overlays(
    output_dir: Path,
    images: Sequence[Path],
    baseline: SweepConfig,
    best: SweepConfig,
    results: dict[tuple[str, Path], tuple[np.ndarray, np.ndarray]],
) -> list[Path]:
    import cv2

    ranked = []
    for image_path in images:
        boxes, _scores = results.get((baseline.name, image_path), (np.empty((0, 4)), np.empty(0)))
        ranked.append((box_diagnostics(boxes)["max_containment_count"], image_path))
    worst = [image_path for _count, image_path in sorted(ranked, key=lambda item: (-item[0], str(item[1])))[:10]]
    paths: list[Path] = []
    for stage, config in (("before", baseline), ("after", best)):
        stage_dir = output_dir / "overlays" / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        for image_path in worst:
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"Could not read image for overlay: {image_path}")
            boxes, scores = results.get((config.name, image_path), (np.empty((0, 4)), np.empty(0)))
            overlay = _draw_overlay(image, boxes, scores)
            destination = stage_dir / f"{image_path.stem}.jpg"
            if not cv2.imwrite(str(destination), overlay):
                raise RuntimeError(f"Could not write overlay: {destination}")
            paths.append(destination)
    return paths


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_diagnostic_sweep(
    model: Any,
    images: Sequence[Path],
    output_dir: Path,
    baseline_threshold: float = 0.25,
    max_resolution: int | None = None,
    resolution_step: int | None = None,
) -> dict[str, Any]:
    """Run the requested label-free sweep and write CSV, overlays, and report."""

    import cv2

    runtime = read_runtime_config(model)
    assert_nms_unreachable(model)
    maximum = max_resolution if max_resolution is not None else runtime.native_resolution * 2
    configs = build_sweep_configs(runtime, baseline_threshold, maximum, resolution_step)
    rows: list[dict[str, Any]] = []
    results: dict[tuple[str, Path], tuple[np.ndarray, np.ndarray]] = {}
    raw_cache: dict[tuple[int, int, Path], tuple[np.ndarray, np.ndarray, list[int]]] = {}
    stopped_resolution: int | None = None

    for config in configs:
        if config.axis == "resolution" and stopped_resolution is not None and config.resolution >= stopped_resolution:
            continue
        for image_path in images:
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"Could not read image: {image_path}")
            image_shape = image.shape[:2]
            cache_key = (config.num_select, config.resolution, image_path)
            try:
                if cache_key not in raw_cache:
                    detections, input_shape = _predict_unthresholded(
                        model, image_path, config.num_select, config.resolution
                    )
                    boxes, scores, _classes = _detections_to_arrays(detections)
                    raw_cache[cache_key] = (boxes, scores, input_shape)
                boxes, scores, input_shape = raw_cache[cache_key]
            except (RuntimeError, MemoryError) as error:
                if config.axis != "resolution" or not _is_oom(error):
                    raise
                stopped_resolution = config.resolution
                print(f"RESOLUTION SWEEP STOPPED at {config.resolution}: {error}", flush=True)
                break

            row = _row_for_config(config, image_path, image_shape, input_shape, boxes, scores)
            row["num_queries"] = runtime.num_queries
            row["native_resolution"] = runtime.native_resolution
            row["resolution_divisor"] = runtime.resolution_divisor
            rows.append(row)

            try:
                assert_predictions_not_capped(len(scores), config.num_select)
            except AssertionError as error:
                print(f"ASSERTION [{config.name}] {image_path.name}: {error}", flush=True)

            keep = scores > config.threshold
            results[(config.name, image_path)] = (boxes[keep], scores[keep])
            print(
                f"[{config.name}] {image_path.name}: {len(scores)} before threshold, {int(keep.sum())} kept, "
                f"max containment={row['max_containment_count']}",
                flush=True,
            )

    if not rows:
        raise RuntimeError("RF-DETR diagnostic produced no rows")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "rfdetr_diagnostics.csv"
    _write_csv(csv_path, rows)

    completed_names = {row["config"] for row in rows}
    completed_configs = [config for config in configs if config.name in completed_names]
    best, verdict, axis_gains = _select_best_and_verdict(completed_configs, results, images)
    baseline = next(config for config in completed_configs if config.axis == "baseline")
    overlays = _save_overlays(output_dir, images, baseline, best, results)
    report_path = output_dir / "verdict.md"
    report_path.write_text(
        "# RF-DETR inference-only collapse verdict\n\n"
        + verdict
        + "\n\n"
        + f"Best label-free config: `{best.name}`. Additional smaller-box centres inside baseline collapse boxes: "
        + f"`{axis_gains}`. Runtime config: num_queries={runtime.num_queries}, num_select={runtime.num_select}, "
        + f"native_resolution={runtime.native_resolution}, divisor={runtime.resolution_divisor}."
        + (f" Resolution sweep stopped on OOM at {stopped_resolution}." if stopped_resolution else "")
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "csv": str(csv_path),
        "report": str(report_path),
        "overlays": [str(path) for path in overlays],
        "best_config": best.name,
        "verdict": verdict,
        "axis_gains": axis_gains,
        "runtime": runtime.__dict__,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
