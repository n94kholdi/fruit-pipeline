"""SAHI-based tiled detection.

Runs a pretrained Ultralytics detector (YOLOv8/YOLO11, optionally YOLO-World
with a text prompt) over image tiles via SAHI, shifts every raw detection
back into original-image coordinates, and hands the unmerged list to the
dedicated merging module. No cross-tile deduplication happens here on
purpose: that is the merge stage's job.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from sahi.prediction import ObjectPrediction
from sahi.slicing import slice_image
from sahi.utils.cv import read_image_as_pil

from fruit_pipeline.detection.backends import DetectorBackend, _relabel_class_agnostic, load_detector_backend

logger = logging.getLogger(__name__)

# Deprecated: kept only so old callers referencing this name don't break.
# Prompting now goes through prompts.PromptConfig / --prompt-config.
DEFAULT_PROMPT_CLASSES = ["fruit", "round fruit", "apple", "orange", "citrus fruit"]


@dataclass
class TileStats:
    num_tiles: int
    raw_detection_count: int
    image_size: tuple[int, int]  # (width, height)
    tile_size: int = 0
    estimated_fruit_diameter_px: float | None = None


@dataclass
class TiledDetectionResult:
    """Full return value of ``detect_tiled``: raw predictions plus per-tile provenance.

    ``tile_ids``/``tile_rects`` let the seam-aware merging strategy
    can tell whether two overlapping detections came from different tiles
    near a shared seam (probably the same fruit, split by tiling -> union
    them) versus two genuinely different, merely-adjacent fruit detected
    within the same tile or from non-adjacent tiles (never union those).
    """

    raw_predictions: list[ObjectPrediction]
    tile_results: list[TileResult]
    stats: "TileStats"
    tile_ids: list[int]  # parallel to raw_predictions; -1 = standard full-image pass (never seam-eligible)
    tile_rects: dict[int, tuple[float, float, float, float]]  # tile_id -> (x1, y1, x2, y2) in full-image coords


@dataclass
class TileResult:
    """One tile's own crop + the raw detections the model made on it.

    Kept separate from the merged, full-image detections so the visualization layer
    can render a per-tile debugging view showing exactly what the detector
    saw and predicted on each individual tile, before any cross-tile merge.
    """

    label: str
    image_rgb: np.ndarray
    shift: tuple[int, int]  # (shift_x, shift_y) into the full image
    boxes_xyxy: list[list[float]]  # local (tile-coordinate) boxes
    scores: list[float]


def estimate_fruit_diameter_px(
    image_arr: np.ndarray,
    backend: DetectorBackend,
    target_long_edge: int = 1400,
    min_detections: int = 3,
    max_box_area_fraction: float = 0.08,
    conf_threshold: float = 0.25,
) -> float | None:
    """Fast pre-pass: estimate the median fruit diameter, in full-res pixels.

    Downscales ``image_arr`` so its long edge is ``target_long_edge``, runs a
    single (non-tiled) detection pass on the downscaled image, and returns the
    median bounding-box diagonal of whatever came back, rescaled up to
    full-resolution pixels. This is deliberately cheap (one forward pass, at
    low resolution) since it only needs to be roughly right — it feeds
    ``compute_tile_size_from_diameter``, not the final detections.

    On a dense multi-fruit crate photo, this coarse pass on a class-agnostic
    detector reliably returns a mix of individual-fruit boxes *and* a few
    boxes spanning a whole cluster/pile (the same "boxed the crate, not one
    fruit" failure ``filter_oversized_boxes`` guards against downstream) —
    with as few as ~10 coarse detections, those outliers can drag the raw
    median toward the cluster size instead of the fruit size. Boxes covering
    more than ``max_box_area_fraction`` of the (downscaled) image are dropped
    before taking the median, same heuristic as the downstream filter.

    Returns None (degenerate case) when too few boxes came back to trust a
    median from — e.g. a near-empty crate, or a coarse pass that missed
    everything. Callers should fall back to a fixed tile size in that case.
    """
    height, width = image_arr.shape[0], image_arr.shape[1]
    long_edge = max(height, width)
    scale = min(1.0, target_long_edge / long_edge) if long_edge > 0 else 1.0

    if scale < 1.0:
        small = cv2.resize(
            image_arr,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = image_arr
        scale = 1.0

    preds = backend.detect(
        small,
        shift=(0, 0),
        full_shape=(small.shape[0], small.shape[1]),
        conf_threshold=conf_threshold,
    )

    small_area = float(small.shape[0] * small.shape[1])
    max_box_area = max_box_area_fraction * small_area
    diagonals = []
    for pred in preds:
        x1, y1, x2, y2 = pred.bbox.to_xyxy()
        w, h = x2 - x1, y2 - y1
        if w > 0 and h > 0 and (w * h) <= max_box_area:
            diagonals.append(math.hypot(w, h))
    n_dropped_as_clusters = len(preds) - len(diagonals)

    if len(diagonals) < min_detections:
        logger.warning(
            "Coarse fruit-size pre-pass found only %d usable detection(s) (need >= %d; %d raw, %d dropped as "
            "whole-cluster boxes > %.0f%% of the downscaled (%dx%d) image) — treating as degenerate.",
            len(diagonals),
            min_detections,
            len(preds),
            n_dropped_as_clusters,
            max_box_area_fraction * 100,
            small.shape[1],
            small.shape[0],
        )
        return None

    median_diag_small = float(np.median(diagonals))
    if median_diag_small <= 0:
        return None

    return median_diag_small / scale


def compute_tile_size_from_diameter(
    diameter_px: float,
    k: float = 8.0,
    min_tile_size: int = 320,
    max_tile_size: int = 2048,
) -> int:
    """Turn an estimated fruit diameter into a tile size, ``tile_size ~= k * diameter``.

    ``k`` (default 8) controls how many fruit-diameters wide a tile is —
    large enough that a tile comfortably contains many whole fruits instead
    of splitting them, while still keeping tile count (and therefore SAM/
    detector passes) reasonable. Clamped to ``[min_tile_size, max_tile_size]``
    so unusual images (huge or tiny estimated fruit) can't produce degenerate
    tiling.
    """
    tile_size = diameter_px * k
    tile_size = max(min_tile_size, min(max_tile_size, tile_size))
    return int(round(tile_size))


def compute_adaptive_tile_size(
    width: int,
    height: int,
    overlap_ratio: float = 0.2,
    max_tiles: int = 12,
    base_tile_size: int = 640,
    growth: float = 1.25,
) -> int:
    """Pick a square tile size that keeps the total tile count within ``max_tiles``.

    Starts from ``base_tile_size`` and grows it (by ``growth`` each step) until
    the resulting tile grid fits the budget. Kept as the fallback used by
    ``detect_tiled`` when the fruit-diameter pre-pass (``estimate_fruit_diameter_px``)
    can't produce a usable estimate — e.g. a near-empty crate. When the
    pre-pass succeeds, ``compute_tile_size_from_diameter`` is used instead,
    since a tile-count budget alone says nothing about whether a tile
    actually contains whole fruit.
    """
    tile_size = float(base_tile_size)
    longest_side = max(width, height)
    while True:
        stride = max(1.0, tile_size * (1 - overlap_ratio))
        n_cols = 1 if width <= tile_size else math.ceil((width - tile_size) / stride) + 1
        n_rows = 1 if height <= tile_size else math.ceil((height - tile_size) / stride) + 1
        if n_cols * n_rows <= max_tiles or tile_size >= longest_side:
            return int(tile_size)
        tile_size *= growth


def load_detector(
    weights_path: str,
    device: str = "cpu",
    conf_threshold: float = 0.25,
    use_yolo_world: bool = False,
    prompt_classes: list[str] | None = None,
) -> DetectorBackend:
    """Deprecated: use ``detectors.load_detector_backend`` directly.

    Kept for backward compatibility with any external caller importing this
    name; returns a ``DetectorBackend`` now instead of a raw SAHI
    ``DetectionModel`` (``detect_tiled`` takes a backend, not a model).
    """
    return load_detector_backend(
        detector="yolo-world" if use_yolo_world else "default",
        weights_path=weights_path,
        device=device,
        conf_threshold=conf_threshold,
        fruit_prompts=prompt_classes or (DEFAULT_PROMPT_CLASSES if use_yolo_world else None),
    )


def detect_tiled(
    image_path: str,
    backend: DetectorBackend,
    tile_size: int | None = None,
    overlap_ratio: float = 0.15,
    conf_threshold: float = 0.25,
    include_standard_pred: bool = True,
    class_agnostic_relabel: bool = True,
    max_tiles: int = 12,
    tile_size_k: float = 8.0,
    min_tile_size: int = 320,
    max_tile_size: int = 2048,
    coarse_pass_long_edge: int = 1400,
    coarse_max_box_area_fraction: float = 0.08,
    fallback_tile_size: int = 640,
    debug_tiles_dir: str | None = None,
) -> TiledDetectionResult:
    """Slice ``image_path`` into tiles, run the detector on each, and return raw detections.

    Args:
        image_path: Path to the full-resolution input image.
        backend: A ``DetectorBackend`` from ``detectors.load_detector_backend``.
        tile_size: Square tile side length in pixels. If None (default), it is
            estimated per-image: a fast coarse pre-pass
            (``estimate_fruit_diameter_px``) finds the median fruit diameter,
            then ``tile_size = tile_size_k * diameter`` (clamped to
            ``[min_tile_size, max_tile_size]``) so each tile comfortably
            contains many whole fruits instead of a fixed size that's either
            wastefully small (hundreds of tiles on a high-res image) or too
            coarse (a small image barely tiled at all). Set explicitly to
            force a fixed size and skip the pre-pass entirely, e.g. to
            reproduce old fixed-tile_size behavior for comparison.
        overlap_ratio: Fractional overlap between adjacent tiles (0-1). With
            adaptive tile sizing, tiles are less likely to cut through a
            single fruit than with a small fixed size, so a smaller overlap
            (~0.10-0.15) is usually enough; the default here is 0.15.
        conf_threshold: Per-tile confidence threshold override.
        include_standard_pred: Also run one full-image (unsliced) pass, which
            helps recall for fruit large enough to be cut awkwardly by tiling.
        class_agnostic_relabel: Collapse all detector classes to a single
            "fruit" label (ignored for YOLO-World, which is already
            open-vocabulary and prompted with fruit-relevant text).
        max_tiles: Advisory tile-count budget, logged as a warning if the
            adaptively-chosen tile_size still produces more tiles than this
            (it is not enforced by shrinking tile_size — clamp via
            ``max_tile_size`` for that).
        tile_size_k: Multiplier from estimated fruit diameter to tile size
            (default 8: a tile is ~8 fruit-diameters wide).
        min_tile_size / max_tile_size: Clamp range for the adaptively
            computed tile size, to avoid degenerate tiling on unusual images.
        coarse_pass_long_edge: Long-edge resolution (pixels) the image is
            downscaled to for the fruit-diameter pre-pass.
        coarse_max_box_area_fraction: Coarse-pass boxes covering more than
            this fraction of the (downscaled) image are dropped before
            computing the median diameter — filters out "boxed a whole
            cluster/pile" detections that would otherwise skew the estimate.
        fallback_tile_size: Tile size used when the pre-pass is degenerate
            (too few coarse detections, e.g. a near-empty crate).
        debug_tiles_dir: If set, every tile crop is saved to this directory
            before being handed to the detector, so tiling can be visually
            sanity-checked (blank/duplicate/out-of-bounds crops would be
            immediately obvious).

    Returns:
        A ``TiledDetectionResult``: raw predictions (un-merged, already
        shifted into full original-image coordinates), each tile's own crop
        + local-coordinate detections (for the per-tile debugging
        visualization), and per-tile provenance (``tile_ids``/``tile_rects``)
        for seam-aware merging.
    """
    pil_image = read_image_as_pil(image_path)
    image_arr = np.ascontiguousarray(pil_image)
    height, width = image_arr.shape[0], image_arr.shape[1]

    estimated_diameter: float | None = None
    if tile_size is None:
        estimated_diameter = estimate_fruit_diameter_px(
            image_arr,
            backend,
            target_long_edge=coarse_pass_long_edge,
            max_box_area_fraction=coarse_max_box_area_fraction,
            conf_threshold=conf_threshold,
        )
        if estimated_diameter is None:
            tile_size = compute_adaptive_tile_size(
                width, height, overlap_ratio=overlap_ratio, max_tiles=max_tiles, base_tile_size=fallback_tile_size
            )
            logger.warning(
                "Fruit-diameter pre-pass degenerate for %s; falling back to budget-based tile_size=%d",
                image_path,
                tile_size,
            )
        else:
            tile_size = compute_tile_size_from_diameter(
                estimated_diameter, k=tile_size_k, min_tile_size=min_tile_size, max_tile_size=max_tile_size
            )
            logger.info(
                "Estimated median fruit diameter=%.1fpx for %s -> tile_size=%d (k=%.1f, clamp=[%d,%d])",
                estimated_diameter,
                image_path,
                tile_size,
                tile_size_k,
                min_tile_size,
                max_tile_size,
            )
    else:
        logger.info("Using fixed --tile-size=%d (adaptive fruit-diameter estimation disabled)", tile_size)

    slice_result = slice_image(
        image=image_arr,
        slice_height=tile_size,
        slice_width=tile_size,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        auto_slice_resolution=False,
    )
    logger.info(
        "Sliced %s (%dx%d) into %d tiles of size %d (overlap=%.2f)%s",
        image_path,
        width,
        height,
        len(slice_result),
        tile_size,
        overlap_ratio,
        " [WARNING: exceeds max_tiles budget]" if len(slice_result) > max_tiles else "",
    )
    if len(slice_result) > max_tiles:
        logger.warning(
            "Tile count %d exceeds max_tiles=%d budget; consider raising max_tile_size or tile_size_k.",
            len(slice_result),
            max_tiles,
        )

    if debug_tiles_dir:
        Path(debug_tiles_dir).mkdir(parents=True, exist_ok=True)

    raw_predictions: list[ObjectPrediction] = []
    tile_results: list[TileResult] = []
    tile_ids: list[int] = []
    tile_rects: dict[int, tuple[float, float, float, float]] = {}

    for idx, (sliced_image, starting_pixel) in enumerate(zip(slice_result.images, slice_result.starting_pixels)):
        shift_x, shift_y = int(starting_pixel[0]), int(starting_pixel[1])
        tile_h, tile_w = sliced_image.shape[0], sliced_image.shape[1]
        tile_rects[idx] = (float(shift_x), float(shift_y), float(shift_x + tile_w), float(shift_y + tile_h))

        if debug_tiles_dir:
            tile_bgr = cv2.cvtColor(sliced_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(Path(debug_tiles_dir) / f"tile_{idx:03d}_x{shift_x}_y{shift_y}.png"), tile_bgr)

        # backend.detect() already returns full-image-shifted predictions
        # (each DetectorBackend implementation handles its own shift, e.g.
        # SahiUltralyticsBackend via SAHI's get_shifted_object_prediction()),
        # so no separate shifting step is needed here.
        shifted_predictions = backend.detect(
            sliced_image,
            shift=(shift_x, shift_y),
            full_shape=(height, width),
            conf_threshold=conf_threshold,
        )
        raw_predictions.extend(shifted_predictions)
        tile_ids.extend([idx] * len(shifted_predictions))

        logger.info(
            "Tile %d: bounds=(%d,%d)-(%d,%d) size=%dx%d raw_detections=%d",
            idx,
            shift_x,
            shift_y,
            shift_x + tile_w,
            shift_y + tile_h,
            tile_w,
            tile_h,
            len(shifted_predictions),
        )

        tile_results.append(
            TileResult(
                label=f"tile {idx}",
                image_rgb=sliced_image,
                shift=(shift_x, shift_y),
                # boxes_xyxy is used only for the debug tile-grid visualization,
                # drawn on the tile's own (local-coordinate) crop -- shift back
                # down from the full-image coordinates backend.detect() returns.
                boxes_xyxy=[
                    [c - shift_x if i % 2 == 0 else c - shift_y for i, c in enumerate(pred.bbox.to_xyxy())]
                    for pred in shifted_predictions
                ],
                scores=[float(pred.score.value) for pred in shifted_predictions],
            )
        )

    if include_standard_pred:
        shifted_predictions = backend.detect(
            image_arr,
            shift=(0, 0),
            full_shape=(height, width),
            conf_threshold=conf_threshold,
        )
        raw_predictions.extend(shifted_predictions)
        tile_ids.extend([-1] * len(shifted_predictions))
        tile_rects[-1] = (0.0, 0.0, float(width), float(height))
        logger.info("Standard (full-image) pass: raw_detections=%d", len(shifted_predictions))
        tile_results.append(
            TileResult(
                label="full image",
                image_rgb=image_arr,
                shift=(0, 0),
                boxes_xyxy=[list(pred.bbox.to_xyxy()) for pred in shifted_predictions],
                scores=[float(pred.score.value) for pred in shifted_predictions],
            )
        )

    if class_agnostic_relabel:
        _relabel_class_agnostic(raw_predictions)

    stats = TileStats(
        num_tiles=len(slice_result),
        raw_detection_count=len(raw_predictions),
        image_size=(width, height),
        tile_size=tile_size,
        estimated_fruit_diameter_px=estimated_diameter,
    )
    logger.info(
        "Raw detections before merge: %d (from %d tiles%s)",
        stats.raw_detection_count,
        stats.num_tiles,
        " + 1 standard pass" if include_standard_pred else "",
    )
    return TiledDetectionResult(
        raw_predictions=raw_predictions,
        tile_results=tile_results,
        stats=stats,
        tile_ids=tile_ids,
        tile_rects=tile_rects,
    )
