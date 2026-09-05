"""Drawing boxes + mask overlays for a quick visual sanity check of a run."""

from __future__ import annotations

import cv2
import numpy as np

from fruit_pipeline.detection.tiling import TileResult
from fruit_pipeline.segmentation.sam import FruitInstance


def _color_for_instance(instance_id: int) -> tuple[int, int, int]:
    """Deterministic, visually distinct-ish BGR color per instance id."""
    rng = np.random.default_rng(instance_id + 1)
    hue = int(rng.integers(0, 180))
    hsv = np.uint8([[[hue, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_overlays(
    image_bgr: np.ndarray,
    instances: list[FruitInstance],
    mask_alpha: float = 0.45,
) -> np.ndarray:
    """Return a copy of ``image_bgr`` with mask overlays and boxes drawn on it."""
    overlay = image_bgr.copy()
    canvas = image_bgr.copy()

    for inst in instances:
        color = _color_for_instance(inst.instance_id)
        overlay[inst.mask] = color

    canvas = cv2.addWeighted(overlay, mask_alpha, canvas, 1 - mask_alpha, 0)

    for inst in instances:
        color = _color_for_instance(inst.instance_id)
        x1, y1, x2, y2 = (int(round(v)) for v in inst.box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"#{inst.instance_id}"
        cv2.putText(
            canvas,
            label,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            lineType=cv2.LINE_AA,
        )

    return canvas


def draw_tile_grid(tile_results: list[TileResult], cell_max_side: int = 480) -> np.ndarray:
    """Lay out every tile's crop with its own raw (pre-merge) detections drawn on it.

    Tiles are placed in a grid matching their actual spatial layout (inferred
    from each tile's shift), so it reads left-to-right/top-to-bottom like the
    real image. The one exception is the "full image" standard-pass tile (if
    present), appended as its own trailing row spanning the full width, since
    it isn't part of the tile grid geometry. Meant purely for debugging how
    the detector performs tile-by-tile, before merging.
    """
    grid_tiles = [t for t in tile_results if t.label != "full image"]
    standard_tiles = [t for t in tile_results if t.label == "full image"]

    xs = sorted({t.shift[0] for t in grid_tiles})
    ys = sorted({t.shift[1] for t in grid_tiles})
    col_of = {x: i for i, x in enumerate(xs)}
    row_of = {y: i for i, y in enumerate(ys)}
    n_cols = max(len(xs), 1)
    n_rows = max(len(ys), 1)

    cells: dict[tuple[int, int], np.ndarray] = {}
    for tile in grid_tiles:
        cells[(row_of[tile.shift[1]], col_of[tile.shift[0]])] = _render_tile_cell(tile, cell_max_side)

    cell_h = max((c.shape[0] for c in cells.values()), default=cell_max_side)
    cell_w = max((c.shape[1] for c in cells.values()), default=cell_max_side)
    blank_cell = np.full((cell_h, cell_w, 3), 40, dtype=np.uint8)

    rows = []
    for r in range(n_rows):
        row_cells = []
        for c in range(n_cols):
            cell = cells.get((r, c), blank_cell)
            padded = np.full((cell_h, cell_w, 3), 40, dtype=np.uint8)
            padded[: cell.shape[0], : cell.shape[1]] = cell
            row_cells.append(padded)
        rows.append(np.hstack(row_cells) if row_cells else blank_cell)
    grid_canvas = np.vstack(rows) if rows else blank_cell

    if standard_tiles:
        full_cell = _render_tile_cell(standard_tiles[0], max_side=grid_canvas.shape[1])
        full_row = np.full((full_cell.shape[0], grid_canvas.shape[1], 3), 40, dtype=np.uint8)
        full_row[:, : full_cell.shape[1]] = full_cell
        grid_canvas = np.vstack([grid_canvas, full_row])

    return grid_canvas


def _render_tile_cell(tile: TileResult, max_side: int) -> np.ndarray:
    tile_bgr = cv2.cvtColor(tile.image_rgb, cv2.COLOR_RGB2BGR)
    h, w = tile_bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w)) if max(h, w) > 0 else 1.0
    if scale != 1.0:
        tile_bgr = cv2.resize(tile_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    for box, score in zip(tile.boxes_xyxy, tile.scores):
        x1, y1, x2, y2 = (int(round(v * scale)) for v in box)
        cv2.rectangle(tile_bgr, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(
            tile_bgr,
            f"{score:.2f}",
            (x1, max(0, y1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 0),
            1,
            lineType=cv2.LINE_AA,
        )

    cv2.rectangle(tile_bgr, (0, 0), (tile_bgr.shape[1] - 1, tile_bgr.shape[0] - 1), (100, 100, 100), 1)
    cv2.putText(
        tile_bgr,
        f"{tile.label} ({len(tile.boxes_xyxy)} raw det.)",
        (4, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
        lineType=cv2.LINE_AA,
    )
    return tile_bgr
