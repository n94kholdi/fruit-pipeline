"""Standalone checkerboard/ChArUco calibration tool.

This module has no dependency on detection, segmentation, or inference.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np

from .calibration_store import CalibrationStore
from .models import CalibrationError, CameraCalibration

logger = logging.getLogger(__name__)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class BoardSpec:
    kind: str = "checkerboard"
    columns: int = 9
    rows: int = 6
    square_size_mm: float = 25.0
    marker_size_mm: float = 18.0
    dictionary: str = "DICT_4X4_50"

    def __post_init__(self) -> None:
        if self.kind not in {"checkerboard", "charuco"}:
            raise ValueError("Board kind must be 'checkerboard' or 'charuco'")
        if min(self.columns, self.rows) < 2 or self.square_size_mm <= 0:
            raise ValueError("Board dimensions and square size must be positive")
        if self.kind == "charuco" and not 0 < self.marker_size_mm < self.square_size_mm:
            raise ValueError("ChArUco marker size must be between zero and square size")


def iter_frames(
    source: str | Path, frame_step: int = 10, max_sampled_frames: int | None = 100,
) -> Iterator[tuple[str, np.ndarray]]:
    """Yield BGR frames from a directory, image, video, camera index, or stream URL."""
    source_text = str(source)
    path = Path(source_text)
    if path.is_dir():
        files = sorted(item for item in path.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            raise CalibrationError(f"No calibration images found in {path}")
        for item in files:
            image = cv2.imread(str(item))
            if image is not None:
                yield str(item), image
        return
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        image = cv2.imread(str(path))
        if image is None:
            raise CalibrationError(f"Cannot read calibration image: {path}")
        yield str(path), image
        return

    capture_source: str | int = int(source_text) if source_text.isdigit() else source_text
    capture = cv2.VideoCapture(capture_source)
    if not capture.isOpened():
        raise CalibrationError(f"Cannot open calibration source: {source_text}")
    try:
        index = 0
        sampled = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % max(1, frame_step) == 0:
                yield f"{source_text}#{index}", frame
                sampled += 1
                if max_sampled_frames is not None and sampled >= max_sampled_frames:
                    break
            index += 1
    finally:
        capture.release()


def _checkerboard_object_points(spec: BoardSpec) -> np.ndarray:
    points = np.zeros((spec.rows * spec.columns, 3), np.float32)
    points[:, :2] = np.mgrid[0 : spec.columns, 0 : spec.rows].T.reshape(-1, 2)
    points[:, :2] *= spec.square_size_mm
    return points


def _charuco_board(spec: BoardSpec):
    if not hasattr(cv2, "aruco"):
        raise CalibrationError("OpenCV was built without the aruco module; install opencv-contrib-python")
    try:
        dictionary_id = getattr(cv2.aruco, spec.dictionary)
    except AttributeError as exc:
        raise CalibrationError(f"Unknown ArUco dictionary: {spec.dictionary}") from exc
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    if hasattr(cv2.aruco, "CharucoBoard"):
        board = cv2.aruco.CharucoBoard(
            (spec.columns, spec.rows), spec.square_size_mm, spec.marker_size_mm, dictionary
        )
    else:  # OpenCV 4.6 and earlier
        board = cv2.aruco.CharucoBoard_create(
            spec.columns, spec.rows, spec.square_size_mm, spec.marker_size_mm, dictionary
        )
    return board, dictionary


def detect_board(frame: np.ndarray, spec: BoardSpec) -> tuple[np.ndarray, np.ndarray] | None:
    """Return matching Nx3 board and Nx2 image points for one valid frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if spec.kind == "checkerboard":
        pattern = (spec.columns, spec.rows)
        found, corners = (
            cv2.findChessboardCornersSB(gray, pattern)
            if hasattr(cv2, "findChessboardCornersSB")
            else (False, None)
        )
        if not found:
            found, corners = cv2.findChessboardCorners(gray, pattern)
            if found:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        if not found:
            return None
        return _checkerboard_object_points(spec), corners.reshape(-1, 2).astype(np.float32)

    board, dictionary = _charuco_board(spec)
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
    if marker_ids is None or len(marker_ids) < 2:
        return None
    count, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners, marker_ids, gray, board
    )
    if charuco_ids is None or count < 6:
        return None
    board_points = board.getChessboardCorners() if hasattr(board, "getChessboardCorners") else board.chessboardCorners
    object_points = np.asarray(board_points, dtype=np.float32)[charuco_ids.reshape(-1)]
    return object_points, charuco_corners.reshape(-1, 2).astype(np.float32)


def _calibrate(
    object_points: list[np.ndarray], image_points: list[np.ndarray], resolution: tuple[int, int]
) -> tuple[float, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    return cv2.calibrateCamera(object_points, image_points, resolution, None, None)


def _view_errors(
    object_points: list[np.ndarray], image_points: list[np.ndarray], rvecs: list[np.ndarray],
    tvecs: list[np.ndarray], matrix: np.ndarray, distortion: np.ndarray,
) -> np.ndarray:
    errors = []
    for objects, images, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(objects, rvec, tvec, matrix, distortion)
        delta = images.reshape(-1, 2) - projected.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))))
    return np.asarray(errors)


def calibrate_camera(
    frames: Iterable[tuple[str, np.ndarray]], camera_id: str, board: BoardSpec,
    camera_group: str | None = None, min_frames: int = 8, max_reprojection_error: float = 2.0,
) -> CameraCalibration:
    """Detect board observations, reject outlier views, and estimate intrinsics."""
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    labels: list[str] = []
    resolution: tuple[int, int] | None = None
    for label, frame in frames:
        current = (frame.shape[1], frame.shape[0])
        if resolution is None:
            resolution = current
        if current != resolution:
            logger.warning("Skipping %s: resolution %s differs from %s", label, current, resolution)
            continue
        observation = detect_board(frame, board)
        if observation is not None:
            objects, images = observation
            object_points.append(objects)
            image_points.append(images)
            labels.append(label)

    if resolution is None or len(object_points) < min_frames:
        raise CalibrationError(f"Only {len(object_points)} valid board frames found; need at least {min_frames}")

    rms, matrix, distortion, rvecs, tvecs = _calibrate(object_points, image_points, resolution)
    errors = _view_errors(object_points, image_points, rvecs, tvecs, matrix, distortion)
    median = float(np.median(errors))
    mad = float(np.median(np.abs(errors - median)))
    cutoff = median + max(3.0 * 1.4826 * mad, 0.25)
    keep = errors <= cutoff
    if int(keep.sum()) >= min_frames and not keep.all():
        rejected = [label for label, accepted in zip(labels, keep) if not accepted]
        logger.info("Rejecting %d high-error calibration frame(s): %s", len(rejected), ", ".join(rejected))
        object_points = [value for value, accepted in zip(object_points, keep) if accepted]
        image_points = [value for value, accepted in zip(image_points, keep) if accepted]
        rms, matrix, distortion, _, _ = _calibrate(object_points, image_points, resolution)

    if not np.isfinite(rms) or rms > max_reprojection_error:
        raise CalibrationError(
            f"Calibration reprojection error {rms:.3f}px exceeds limit {max_reprojection_error:.3f}px"
        )
    return CameraCalibration(
        camera_id=camera_id,
        camera_group=camera_group,
        resolution=resolution,
        camera_matrix=matrix,
        distortion_coefficients=distortion,
        reprojection_error=float(rms),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate one camera from checkerboard or ChArUco frames.")
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--camera-group")
    parser.add_argument("--images", "--source", dest="source", required=True, help="Folder, image, video, camera index, or RTSP URL")
    parser.add_argument("--output-dir", default="calibrations")
    parser.add_argument("--save-as-group", action="store_true", help="Save as shared group calibration instead of camera-specific")
    parser.add_argument("--board", choices=["checkerboard", "charuco"], default="checkerboard")
    parser.add_argument("--columns", type=int, default=9, help="Checkerboard inner corners, or ChArUco squares, horizontally")
    parser.add_argument("--rows", type=int, default=6, help="Checkerboard inner corners, or ChArUco squares, vertically")
    parser.add_argument("--square-size-mm", type=float, default=25.0)
    parser.add_argument("--marker-size-mm", type=float, default=18.0)
    parser.add_argument("--dictionary", default="DICT_4X4_50")
    parser.add_argument("--frame-step", type=int, default=10)
    parser.add_argument(
        "--max-sampled-frames", type=int, default=100,
        help="Stop a video/live stream after this many sampled frames (default: 100)",
    )
    parser.add_argument("--min-frames", type=int, default=8)
    parser.add_argument("--max-reprojection-error", type=float, default=2.0)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    board = BoardSpec(args.board, args.columns, args.rows, args.square_size_mm, args.marker_size_mm, args.dictionary)
    try:
        calibration = calibrate_camera(
            iter_frames(args.source, args.frame_step, args.max_sampled_frames),
            args.camera_id, board, args.camera_group,
            args.min_frames, args.max_reprojection_error,
        )
        path = CalibrationStore(args.output_dir).save(calibration, as_group=args.save_as_group)
    except (CalibrationError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Saved calibration to {path} (RMS reprojection error: {calibration.reprojection_error:.4f}px)")


if __name__ == "__main__":
    main()
