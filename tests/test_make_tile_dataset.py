import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "make_tile_dataset.py"
_spec = importlib.util.spec_from_file_location("make_tile_dataset", SCRIPT_PATH)
make_tile_dataset_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_tile_dataset_module)

transform_box_to_tile = make_tile_dataset_module.transform_box_to_tile
visible_fraction = make_tile_dataset_module.visible_fraction
make_tile_dataset = make_tile_dataset_module.make_tile_dataset


# --- round-trip: the explicitly-required coverage for a shift-direction bug ---


def test_transform_box_to_tile_round_trip_for_fully_contained_box():
    tile_rect = (100.0, 200.0, 300.0, 400.0)  # x1,y1,x2,y2
    original_box = [120.0, 220.0, 150.0, 260.0]  # fully inside the tile

    tile_local = transform_box_to_tile(original_box, tile_rect)
    assert tile_local is not None

    # Reconstruct full-image coords by adding the tile's own offset back.
    tx1, ty1, _, _ = tile_rect
    reconstructed = [tile_local[0] + tx1, tile_local[1] + ty1, tile_local[2] + tx1, tile_local[3] + ty1]
    assert reconstructed == original_box


def test_transform_box_to_tile_round_trip_many_tiles_and_boxes():
    # A grid of tiles and boxes at various offsets -- catches an axis-swap
    # (x/y flipped) bug that a single hand-picked case might miss by luck.
    for tile_x in (0, 100, 250):
        for tile_y in (0, 50, 300):
            tile_rect = (float(tile_x), float(tile_y), float(tile_x + 200), float(tile_y + 200))
            for dx in (10, 50, 150):
                for dy in (10, 80, 150):
                    original_box = [tile_x + dx, tile_y + dy, tile_x + dx + 20, tile_y + dy + 20]
                    tile_local = transform_box_to_tile(original_box, tile_rect)
                    assert tile_local is not None
                    reconstructed = [
                        tile_local[0] + tile_x,
                        tile_local[1] + tile_y,
                        tile_local[2] + tile_x,
                        tile_local[3] + tile_y,
                    ]
                    assert reconstructed == original_box


def test_transform_box_to_tile_no_overlap_returns_none():
    tile_rect = (0.0, 0.0, 100.0, 100.0)
    original_box = [500.0, 500.0, 520.0, 520.0]
    assert transform_box_to_tile(original_box, tile_rect) is None


def test_transform_box_to_tile_clips_box_crossing_tile_edge():
    # tile_rect origin is (0,0) here, so tile-local == full-image coords --
    # confirms clipping alone, independent of the origin-offset subtraction
    # (covered separately below).
    tile_rect = (0.0, 0.0, 100.0, 100.0)
    original_box = [90.0, 40.0, 130.0, 60.0]  # extends 30px past the tile's right edge

    clipped = transform_box_to_tile(original_box, tile_rect)
    assert clipped == [90.0, 40.0, 100.0, 60.0]


def test_transform_box_to_tile_result_is_relative_to_tile_origin():
    # tile_rect origin is (50,50): the result must be shifted DOWN by that
    # origin, not left as full-image coordinates.
    tile_rect = (50.0, 50.0, 150.0, 150.0)
    original_box = [50.0, 50.0, 70.0, 70.0]  # touches the tile's top-left corner

    tile_local = transform_box_to_tile(original_box, tile_rect)
    assert tile_local == [0.0, 0.0, 20.0, 20.0]


# --- visible fraction ------------------------------------------------------


def test_visible_fraction_fully_contained_is_one():
    original = [10.0, 10.0, 30.0, 30.0]
    clipped = [10.0, 10.0, 30.0, 30.0]
    assert visible_fraction(original, clipped) == 1.0


def test_visible_fraction_half_clipped():
    original = [0.0, 0.0, 20.0, 10.0]  # area 200
    clipped = [0.0, 0.0, 10.0, 10.0]  # area 100
    assert visible_fraction(original, clipped) == 0.5


# --- full pipeline: negative-tile sampling + min-visible-fraction ----------


def _write_test_image(path: Path, width: int, height: int) -> None:
    img = np.full((height, width, 3), 200, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_make_tile_dataset_drops_boxes_below_min_visible_fraction(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    _write_test_image(images_dir / "img1.jpg", 200, 100)

    coco_gt = {
        "images": [{"id": 1, "file_name": "img1.jpg", "width": 200, "height": 100}],
        "annotations": [
            # Box mostly outside a left-half tile (assuming tile_size=100, no overlap):
            # box [90,10,130,30] straddles the x=100 tile seam.
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [90, 10, 40, 20], "area": 800, "iscrowd": 0},
        ],
        "categories": [{"id": 1, "name": "fruit"}],
    }

    output_dir = tmp_path / "out"
    new_coco = make_tile_dataset(
        coco_gt=coco_gt,
        images_dir=str(images_dir),
        output_dir=str(output_dir),
        tile_size=100,
        overlap_ratio=0.0,
        min_visible_fraction=0.9,  # strict: only near-fully-visible fragments kept
        negative_tile_fraction=0.0,
        adaptive_mode="diameter",
        tile_size_k=8.0,
        min_tile_size=320,
        max_tile_size=2048,
        fixed_slice_count=4,
    )

    tile_annotations = [ann for ann in new_coco["annotations"] if ann["image_id"] != 1]
    assert tile_annotations == []  # both fragments are ~25-75% visible, below the 0.9 threshold


def test_make_tile_dataset_keeps_boxes_above_min_visible_fraction(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    _write_test_image(images_dir / "img1.jpg", 200, 100)

    coco_gt = {
        "images": [{"id": 1, "file_name": "img1.jpg", "width": 200, "height": 100}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0},  # fully inside tile 0
        ],
        "categories": [{"id": 1, "name": "fruit"}],
    }

    output_dir = tmp_path / "out"
    new_coco = make_tile_dataset(
        coco_gt=coco_gt,
        images_dir=str(images_dir),
        output_dir=str(output_dir),
        tile_size=100,
        overlap_ratio=0.0,
        min_visible_fraction=0.3,
        negative_tile_fraction=0.0,
        adaptive_mode="diameter",
        tile_size_k=8.0,
        min_tile_size=320,
        max_tile_size=2048,
        fixed_slice_count=4,
    )

    tile_images = [img for img in new_coco["images"] if img.get("source_image_id") == 1]
    tile_annotations = [ann for ann in new_coco["annotations"] if ann["image_id"] in {img["id"] for img in tile_images}]
    assert len(tile_annotations) == 1
    # The full-resolution image's own original annotation also passes through unchanged.
    full_res_annotations = [ann for ann in new_coco["annotations"] if ann["image_id"] == 1]
    assert len(full_res_annotations) == 1
    assert full_res_annotations[0]["bbox"] == [10, 10, 20, 20]


def test_make_tile_dataset_negative_tile_fraction_zero_keeps_no_empty_tiles(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    _write_test_image(images_dir / "img1.jpg", 400, 100)  # 4 tiles of 100px, no overlap, zero annotations

    coco_gt = {
        "images": [{"id": 1, "file_name": "img1.jpg", "width": 400, "height": 100}],
        "annotations": [],
        "categories": [{"id": 1, "name": "fruit"}],
    }

    output_dir = tmp_path / "out"
    new_coco = make_tile_dataset(
        coco_gt=coco_gt,
        images_dir=str(images_dir),
        output_dir=str(output_dir),
        tile_size=100,
        overlap_ratio=0.0,
        min_visible_fraction=0.3,
        negative_tile_fraction=0.0,
        adaptive_mode="resolution",
        tile_size_k=8.0,
        min_tile_size=320,
        max_tile_size=2048,
        fixed_slice_count=4,
    )

    tile_images = [img for img in new_coco["images"] if img.get("source_image_id") == 1]
    assert tile_images == []


def test_make_tile_dataset_negative_tile_fraction_one_keeps_all_empty_tiles(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    _write_test_image(images_dir / "img1.jpg", 400, 100)

    coco_gt = {
        "images": [{"id": 1, "file_name": "img1.jpg", "width": 400, "height": 100}],
        "annotations": [],
        "categories": [{"id": 1, "name": "fruit"}],
    }

    output_dir = tmp_path / "out"
    new_coco = make_tile_dataset(
        coco_gt=coco_gt,
        images_dir=str(images_dir),
        output_dir=str(output_dir),
        tile_size=100,
        overlap_ratio=0.0,
        min_visible_fraction=0.3,
        negative_tile_fraction=1.0,
        adaptive_mode="resolution",
        tile_size_k=8.0,
        min_tile_size=320,
        max_tile_size=2048,
        fixed_slice_count=4,
    )

    tile_images = [img for img in new_coco["images"] if img.get("source_image_id") == 1]
    assert len(tile_images) == 4
    for img in tile_images:
        assert (output_dir / img["file_name"]).exists()
