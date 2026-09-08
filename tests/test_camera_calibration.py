import cv2
import numpy as np
import pytest

from fruit_pipeline.camera_calibration.calibrate import (
    BoardSpec,
    _charuco_board,
    build_parser,
    detect_board,
    save_detection_outputs,
)


@pytest.mark.skipif(not hasattr(cv2, "aruco"), reason="OpenCV has no ArUco module")
def test_detect_charuco_board_with_installed_opencv_api():
    spec = BoardSpec(
        kind="charuco",
        columns=11,
        rows=8,
        square_size_mm=20,
        marker_size_mm=15,
        dictionary="DICT_4X4_50",
    )
    board, _ = _charuco_board(spec)
    if hasattr(board, "generateImage"):
        image = board.generateImage((1100, 800))
    else:
        image = board.draw((1100, 800))
    frame = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    observation = detect_board(frame, spec)

    assert observation is not None
    object_points, image_points = observation
    assert object_points.shape == (70, 3)
    assert image_points.shape == (70, 2)
    assert object_points.dtype == np.float32
    assert image_points.dtype == np.float32


@pytest.mark.skipif(
    not hasattr(cv2, "aruco") or not hasattr(cv2.aruco.CharucoBoard, "setLegacyPattern"),
    reason="OpenCV does not support legacy ChArUco patterns",
)
def test_detect_inverted_legacy_charuco_board():
    spec = BoardSpec("charuco", 11, 8, 20, 15, "DICT_5X5_50")
    board, _ = _charuco_board(spec)
    board.setLegacyPattern(True)
    image = board.generateImage((1100, 800))
    inverted = cv2.bitwise_not(image)
    frame = cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)

    observation = detect_board(frame, spec)

    assert observation is not None
    assert observation[0].shape == (70, 3)
    assert observation[1].shape == (70, 2)


@pytest.mark.skipif(not hasattr(cv2, "aruco"), reason="OpenCV has no ArUco module")
def test_save_detection_outputs_writes_annotated_image(tmp_path):
    spec = BoardSpec("charuco", 11, 8, 20, 15, "DICT_5X5_50")
    board, _ = _charuco_board(spec)
    frame = cv2.cvtColor(board.generateImage((1100, 800)), cv2.COLOR_GRAY2BGR)

    passed_through = list(save_detection_outputs([("board.png", frame)], spec, tmp_path))

    assert len(passed_through) == 1
    output_path = tmp_path / "0001_board_aruco.jpg"
    assert output_path.is_file()
    annotated = cv2.imread(str(output_path))
    assert annotated is not None
    assert np.any(annotated[:42] != frame[:42])


def test_parser_accepts_detection_output_directory():
    args = build_parser().parse_args(
        ["--camera-id", "cam", "--images", "images", "--detection-output-dir", "detections"]
    )

    assert args.detection_output_dir == "detections"
