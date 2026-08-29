"""Merge overlapping tiled detections into one clean set of per-fruit boxes.

Three strategies, chosen via ``--merge-strategy``:

- ``seam-aware`` (default): a self-contained implementation (not SAHI's
  postprocessors — see below) that only ever *unions* two overlapping boxes
  when they came from different tiles AND both lie near the shared overlap
  band between those tiles (i.e. plausibly two partial views of the same
  fruit, split by tiling). Every other overlapping pair — including two
  genuinely different, merely-adjacent/touching fruit detected within the
  same tile — goes through suppression only, never a union.
- ``nmm``: SAHI's own ``NMMPostprocess``, unchanged. This is the literal
  "old behaviour" A/B baseline: it unions ANY sufficiently-overlapping pair
  regardless of tile geometry, which is exactly the failure mode
  ``seam-aware`` exists to fix (two touching-but-distinct fruit getting
  unioned into one implausible blob).
- ``nms``: this module's own suppression-only core (see below) with
  unioning switched off entirely — every overlapping pair is resolved by
  keeping the higher-scoring box and dropping the other.

Why ``seam-aware``/``nms`` aren't built on SAHI's postprocessors: SAHI 0.12.6
hardcodes its match metric to plain IOU/IOS (``sahi/postprocess/utils.py:
has_match()``) with no extension point for DIoU, and silently drops tile
provenance on every merge (``merge_object_prediction_pair()`` just copies
one parent's ``shift_amount``). Reimplementing the suppression/merge core
here — using ``fruit_pipeline.geometry`` — is what makes DIoU-NMS,
containment suppression, and seam-gated unioning possible at all.

The suppression core (shared by ``seam-aware`` and ``nms``) applies two
extras beyond plain IoU suppression, both purely about dropping a box,
never merging boxes:
- Containment suppression (``--containment-threshold``): if one box is
  almost entirely inside another (intersection / smaller-box-area above the
  threshold), the lower-scoring one is dropped outright — the Mask-NMS rule
  from SDM-D (arXiv 2411.16196), applied to boxes.
- DIoU-NMS (``--nms-metric diou``): suppression uses Distance-IoU instead of
  plain IoU, which adds a normalized center-distance penalty. Two
  same-colored touching fruit have high IoU but distinct centers; plain
  IoU-NMS wrongly treats that as "same object" and kills one, DIoU-NMS
  doesn't. See the Cluster-DIoU-NMS idea in ASAHI (arXiv 2604.19233).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sahi.postprocess.combine import GreedyNMMPostprocess, NMMPostprocess
from sahi.prediction import ObjectPrediction

from fruit_pipeline.geometry import (
    box_area,
    containment_ratio,
    diou_xyxy,
    expand_rect,
    iou_xyxy,
    rect_intersection,
    rects_overlap,
)

logger = logging.getLogger(__name__)

# Legacy strategies that still delegate to SAHI's own postprocessors,
# unchanged, as literal "old behaviour" A/B baselines. "greedy_nmm" is kept
# only so existing scripts/configs referencing the old default don't break;
# prefer "seam-aware" (the new default) or "nmm" going forward.
_LEGACY_SAHI_STRATEGIES = {
    "nmm": NMMPostprocess,
    "greedy_nmm": GreedyNMMPostprocess,
}

_CUSTOM_STRATEGIES = {"seam-aware", "nms"}

ALL_STRATEGIES = tuple(_LEGACY_SAHI_STRATEGIES) + tuple(sorted(_CUSTOM_STRATEGIES))


@dataclass
class Detection:
    """A single merged, pre-segmentation fruit candidate."""

    instance_id: int
    box: list[float]  # [x1, y1, x2, y2] in original image coordinates
    score: float
    category_name: str


@dataclass
class _ActiveCandidate:
    """One in-progress kept box during the greedy suppression/merge pass."""

    box: list[float]
    score: float
    template: ObjectPrediction  # source prediction supplying category/full_shape for the final output
    tile_ids: set[int] = field(default_factory=set)  # tile ids unioned into this candidate so far (empty if none)


def merge_detections(
    object_predictions: list[ObjectPrediction],
    strategy: str = "seam-aware",
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
    class_agnostic: bool = True,
    tile_ids: list[int] | None = None,
    tile_rects: dict[int, tuple[float, float, float, float]] | None = None,
    seam_margin: float = 0.0,
    nms_metric: str = "iou",
    containment_threshold: float = 0.9,
) -> list[ObjectPrediction]:
    """Deduplicate/merge raw per-tile detections into one set per real fruit.

    Args:
        object_predictions: Raw detections from ``detect.detect_tiled``,
            already shifted to full-image coordinates but not yet merged.
        strategy: ``"seam-aware"`` (default), ``"nms"``, ``"nmm"``, or the
            deprecated ``"greedy_nmm"`` alias. See module docstring.
        match_metric: ``"IOU"`` or ``"IOS"`` — only used by the legacy
            ``nmm``/``greedy_nmm`` strategies (SAHI's own match metric
            choice). ``seam-aware``/``nms`` use ``nms_metric`` instead.
        match_threshold: Overlap threshold (whichever metric applies) above
            which two detections are considered the same fruit.
        class_agnostic: Merge across categories. True is the right default
            here since every box is (or was relabeled to) "fruit".
        tile_ids: Parallel to ``object_predictions`` — which tile (index
            into ``detect_tiled``'s tile loop, ``-1`` for the standard
            full-image pass) each detection came from. Required for
            ``seam-aware`` (ignored otherwise).
        tile_rects: ``tile_id -> (x1, y1, x2, y2)`` in full-image
            coordinates. Required for ``seam-aware`` (ignored otherwise).
        seam_margin: ``seam-aware`` only — how far (px) from the tiles'
            shared overlap band a box may be and still be considered
            seam-eligible for unioning.
        nms_metric: ``"iou"`` (default) or ``"diou"`` — suppression metric
            used by ``seam-aware``/``nms``'s shared suppression core.
        containment_threshold: ``seam-aware``/``nms`` only — intersection /
            smaller-box-area above which the lower-scoring of two boxes is
            dropped outright (suppression, never a union). ``<= 0`` disables
            containment suppression entirely.
    """
    if not object_predictions:
        return []

    if strategy in _LEGACY_SAHI_STRATEGIES:
        postprocess = _LEGACY_SAHI_STRATEGIES[strategy](
            match_threshold=match_threshold,
            match_metric=match_metric,
            class_agnostic=class_agnostic,
        )
        merged = postprocess(object_predictions)
    elif strategy in _CUSTOM_STRATEGIES:
        merged = _merge_custom(
            object_predictions,
            allow_union=(strategy == "seam-aware"),
            match_threshold=match_threshold,
            tile_ids=tile_ids,
            tile_rects=tile_rects,
            seam_margin=seam_margin,
            nms_metric=nms_metric,
            containment_threshold=containment_threshold,
            class_agnostic=class_agnostic,
        )
    else:
        raise ValueError(f"Unknown merge strategy '{strategy}', expected one of {ALL_STRATEGIES}")

    logger.info(
        "Merged %d raw detections -> %d fruit candidates (strategy=%s)",
        len(object_predictions),
        len(merged),
        strategy,
    )
    return merged


def _merge_custom(
    object_predictions: list[ObjectPrediction],
    allow_union: bool,
    match_threshold: float,
    tile_ids: list[int] | None,
    tile_rects: dict[int, tuple[float, float, float, float]] | None,
    seam_margin: float,
    nms_metric: str,
    containment_threshold: float,
    class_agnostic: bool,
) -> list[ObjectPrediction]:
    """Shared greedy suppression/merge core for ``seam-aware`` and ``nms``.

    Processes predictions in descending score order (standard greedy-NMS
    order). For each candidate, compares against already-kept entries in
    the order they were kept:
    - Containment above threshold -> drop the candidate (suppression).
    - Overlap (IoU or DIoU) above ``match_threshold``:
        - If unioning is allowed and the pair is seam-eligible -> union the
          candidate into that kept entry.
        - Otherwise -> drop the candidate (suppression, never a union).
    The first matching kept entry wins (greedy), same granularity as SAHI's
    own GreedyNMM, so ``seam-aware``/``nms`` are directly comparable to it.

    If ``class_agnostic`` is False, candidates only ever compare against
    kept entries of the same category.
    """
    if allow_union and (tile_ids is None or tile_rects is None):
        raise ValueError("seam-aware strategy requires tile_ids and tile_rects")
    if tile_ids is None:
        tile_ids = [-1] * len(object_predictions)

    order = np.argsort([-float(p.score.value) for p in object_predictions])
    active: list[_ActiveCandidate] = []

    for idx in order:
        pred = object_predictions[idx]
        box = list(pred.bbox.to_xyxy())
        score = float(pred.score.value)
        t_id = tile_ids[idx]

        suppressed = False
        union_target: _ActiveCandidate | None = None

        for cand in active:
            if not class_agnostic and cand.template.category.id != pred.category.id:
                continue

            if containment_threshold > 0 and containment_ratio(box, cand.box) >= containment_threshold:
                suppressed = True
                break

            metric_value = diou_xyxy(box, cand.box) if nms_metric == "diou" else iou_xyxy(box, cand.box)
            if metric_value >= match_threshold:
                if allow_union and t_id != -1 and t_id not in cand.tile_ids and _seam_eligible(
                    box, t_id, cand, tile_rects, seam_margin
                ):
                    union_target = cand
                else:
                    suppressed = True
                break

        if suppressed:
            continue
        if union_target is not None:
            union_target.box = _union_box(union_target.box, box)
            union_target.score = max(union_target.score, score)
            union_target.tile_ids.add(t_id)
        else:
            active.append(_ActiveCandidate(box=box, score=score, template=pred, tile_ids={t_id} if t_id != -1 else set()))

    return [_to_object_prediction(cand) for cand in active]


def _seam_eligible(
    box: list[float],
    t_id: int,
    cand: _ActiveCandidate,
    tile_rects: dict[int, tuple[float, float, float, float]],
    margin: float,
) -> bool:
    """True if ``box`` (from tile ``t_id``) plausibly overlaps ``cand`` because they're the
    same fruit split across a tile seam, not two different adjacent fruit.

    Requires the two tiles to actually be neighbors (their rects overlap —
    true by construction for adjacent tiles in an overlapping SAHI grid,
    false for non-adjacent tiles), and both boxes to intersect that shared
    band, grown outward by ``margin``. Checked against every tile id already
    folded into ``cand`` (not just its first), so a fruit split across 3+
    tiles at a grid corner can still be unioned in more than one step.
    """
    rect_a = tile_rects.get(t_id)
    if rect_a is None:
        return False
    for other_id in cand.tile_ids:
        rect_b = tile_rects.get(other_id)
        if rect_b is None:
            continue
        overlap_rect = rect_intersection(rect_a, rect_b)
        if overlap_rect is None:
            continue
        zone = expand_rect(overlap_rect, margin)
        if rects_overlap(tuple(box), zone) and rects_overlap(tuple(cand.box), zone):
            return True
    return False


def _union_box(box_a: list[float], box_b: list[float]) -> list[float]:
    x1 = min(box_a[0], box_b[0])
    y1 = min(box_a[1], box_b[1])
    x2 = max(box_a[2], box_b[2])
    y2 = max(box_a[3], box_b[3])
    return [x1, y1, x2, y2]


def _to_object_prediction(cand: _ActiveCandidate) -> ObjectPrediction:
    # NOTE: sahi's BoundingBox is a frozen dataclass holding only (box,
    # shift_amount) -- full_shape is used transiently in
    # ObjectAnnotation.__init__ to clip the box and then discarded, it's
    # never stored as a retrievable attribute. Boxes here are already in
    # final full-image coordinates (shift_amount=[0, 0] is correct, not a
    # placeholder), so there's nothing to reconstruct.
    template = cand.template
    x1, y1, x2, y2 = cand.box
    return ObjectPrediction(
        bbox=[x1, y1, x2, y2],
        category_id=template.category.id,
        category_name=template.category.name,
        score=cand.score,
        shift_amount=[0, 0],
    )


def filter_oversized_boxes(
    object_predictions: list[ObjectPrediction],
    max_area_ratio: float = 3.0,
    enabled: bool = True,
) -> list[ObjectPrediction]:
    """Drop boxes implausibly larger than the median fruit box in this image.

    Heuristic for "box drawn around the whole crate instead of one fruit":
    with dozens of similarly-sized fruit per image, the median box area is a
    reasonable stand-in for "one fruit," so anything far larger is suspect.
    """
    if not enabled or len(object_predictions) < 2:
        return object_predictions

    areas = np.array([box_area(list(pred.bbox.to_xyxy())) for pred in object_predictions])
    median_area = float(np.median(areas))
    if median_area <= 0:
        return object_predictions

    keep = [pred for pred, area in zip(object_predictions, areas) if area <= max_area_ratio * median_area]
    dropped = len(object_predictions) - len(keep)
    if dropped:
        logger.info(
            "Oversized-box filter dropped %d/%d detections (> %.1fx median area %.0fpx^2)",
            dropped,
            len(object_predictions),
            max_area_ratio,
            median_area,
        )
    return keep


def to_detections(object_predictions: list[ObjectPrediction]) -> list[Detection]:
    """Assign stable instance ids and convert to the plain ``Detection`` type.

    Ids are assigned in reading order (top-to-bottom, then left-to-right) so
    they stay intuitive when cross-referenced against the visualization, and
    remain stable identifiers that later pipeline stages (classification,
    sizing, rotten/fine detection) can key their own per-fruit data on.
    """
    ordered = sorted(object_predictions, key=lambda pred: (pred.bbox.to_xyxy()[1], pred.bbox.to_xyxy()[0]))
    return [
        Detection(
            instance_id=idx,
            box=[float(v) for v in pred.bbox.to_xyxy()],
            score=float(pred.score.value),
            category_name=pred.category.name,
        )
        for idx, pred in enumerate(ordered)
    ]
