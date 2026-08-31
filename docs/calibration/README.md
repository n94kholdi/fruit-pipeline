# Camera Calibration, Pallet Homography, and Fruit Size Tutorial

This document is the working tutorial for the real-world fruit-size stage. It
is intentionally designed to be updated as the pallet keypoint model and the
end-to-end production integration are completed.

The system estimates fruit dimensions **projected onto the pallet plane**. It
does not estimate fruit height, full 3D shape, or volume.

## Current implementation status

| Stage | Status | Main location |
|---|---|---|
| Checkerboard and ChArUco camera calibration | Implemented | `src/fruit_pipeline/camera_calibration/` |
| Camera-specific and camera-group calibration loading | Implemented | `camera_calibration/calibration_store.py` |
| Pallet type and physical-dimension configuration | Implemented | `config/pallet_types.yaml` |
| Pallet detector interface and corner validation | Implemented | `pallet_geometry/detector.py` |
| Trained pallet corner model | **TODO** | Must implement `PalletDetector` |
| Distortion correction and pallet homography | Implemented | `pallet_geometry/homography.py` |
| Mask-based planar fruit measurements | Implemented | `measurement/` |
| Debug overlay and rectified pallet view | Implemented | `size_estimation/pipeline.py` |
| Size-estimation command-line integration | **TODO** | Currently called from Python |
| Validation against manually measured fruit | **TODO** | Requires a physical validation dataset |

## 1. Before calibrating a camera

Calibration is performed separately from normal fruit inference. Calibrate a
camera once, save its parameters, and recalibrate when the camera geometry or
image configuration changes.

You need:

- A printed rigid checkerboard or ChArUco board with known square size.
- The camera and lens used in production.
- The exact production resolution and preferably the same focus and zoom.
- At least 15-30 sharp images of the board at varied positions and angles.

For a checkerboard, `--columns` and `--rows` mean **inner corner counts**, not
the number of printed squares. Measure `--square-size-mm` accurately.

Capture useful calibration views:

1. Keep the complete board visible and in focus.
2. Move it into the image center, edges, and all four corners.
3. Include front-facing and tilted views in both directions.
4. Vary the board distance while keeping its corners detectable.
5. Avoid repeated, almost identical frames, motion blur, glare, and a bent
   paper board.
6. Do not resize the images before calibration.

A suggested directory layout is:

```text
calibration/
└── cam_001/
    ├── frame_001.jpg
    ├── frame_002.jpg
    └── ...
```

## 2. Run camera calibration

Install the project first if needed:

```bash
pip install -e .
```

### Checkerboard image folder

The following example uses a board with 9 by 6 inner corners and 25 mm
squares:

```bash
python calibrate_camera.py \
  --camera-id cam_001 \
  --camera-group camera_model_A \
  --images calibration/cam_001/ \
  --board checkerboard \
  --columns 9 \
  --rows 6 \
  --square-size-mm 25 \
  --output-dir calibrations
```

The installed equivalent is:

```bash
calibrate-camera \
  --camera-id cam_001 \
  --images calibration/cam_001/ \
  --columns 9 --rows 6 --square-size-mm 25
```

### ChArUco board

For ChArUco, `--columns` and `--rows` are the board's square counts. The
dictionary and physical marker size must match the printed board.

```bash
python calibrate_camera.py \
  --camera-id cam_001 \
  --camera-group camera_model_A \
  --images calibration/cam_001/ \
  --board charuco \
  --columns 9 \
  --rows 6 \
  --square-size-mm 25 \
  --marker-size-mm 18 \
  --dictionary DICT_4X4_50 \
  --output-dir calibrations
```

### Video, camera device, or RTSP stream

`--images` and `--source` are aliases. Either can point to a video, numeric
camera device, or RTSP URL:

```bash
python calibrate_camera.py \
  --camera-id cam_001 \
  --source rtsp://USER:PASSWORD@CAMERA/STREAM \
  --frame-step 10 \
  --max-sampled-frames 100 \
  --columns 9 --rows 6 --square-size-mm 25
```

Do not commit real RTSP credentials. The tool automatically ignores frames in
which the board cannot be detected, skips frames with a different resolution,
and rejects high-error calibration observations when enough good frames
remain. Calibration fails if fewer than `--min-frames` valid observations are
available or if the final reprojection error exceeds
`--max-reprojection-error`.

## 3. Calibration output location

The default output root is `calibrations/`. A camera-specific run saves:

```text
calibrations/cameras/cam_001.json
```

The file contains:

```json
{
  "camera_id": "cam_001",
  "camera_group": "camera_model_A",
  "resolution": [1920, 1080],
  "camera_matrix": [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
  "fx": 0,
  "fy": 0,
  "cx": 0,
  "cy": 0,
  "distortion_coefficients": [],
  "reprojection_error": 0.0
}
```

The zeros above only illustrate the schema. Do not manually replace computed
values with zeros.

To save one calibration as a shared camera-group fallback, provide a group and
add `--save-as-group`:

```bash
python calibrate_camera.py \
  --camera-id calibration_reference_camera \
  --camera-group camera_model_A \
  --images calibration/reference_camera/ \
  --columns 9 --rows 6 --square-size-mm 25 \
  --save-as-group
```

This writes:

```text
calibrations/groups/camera_model_A.json
```

At runtime, calibration lookup order is:

```text
calibrations/cameras/<camera_id>.json
                 ↓ if missing
calibrations/groups/<camera_group>.json
                 ↓ if missing
error: camera is not calibrated
```

A camera-specific file therefore always overrides a group file. Group
calibration should only be used when cameras genuinely share the same optical
setup, lens configuration, focus, and resolution. Individual calibration is
preferred for the best accuracy.

## 4. Check calibration quality

After calibration:

1. Confirm that `resolution` exactly matches production images.
2. Inspect `reprojection_error`. Lower is better; the current default maximum
   is 2.0 pixels, but the acceptable limit should be selected from physical
   measurement validation.
3. Undistort a few images and visually check that straight pallet edges no
   longer bow near the image boundaries.
4. Repeat calibration if the board images lacked position or angle diversity.
5. Recalibrate after changing the lens, zoom, focus, camera resolution, or
   image crop.

## 5. Complete the pallet detector

The homography requires the four **physical pallet-plane corners**, not only a
bounding box. The detector must return semantically labelled points in this
exact order:

```text
0: top-left
1: top-right
2: bottom-right
3: bottom-left
```

The terms refer to the configured pallet coordinate system. The edge from
top-left to top-right represents pallet `width_mm`; the edge from top-left to
bottom-left represents pallet `length_mm`.

### Remaining pallet-model work

1. **Define the visible physical corner convention.** Decide which repeatable
   pallet surface defines the measurement plane. Fruit masks and all four
   pallet corners must correspond to the same physical plane assumption.
2. **Collect images.** Include every production camera, pallet type, loading
   state, viewing angle, lighting condition, partial obstruction, and expected
   perspective.
3. **Annotate five fields per pallet.** Label the four semantic keypoints and
   `pallet_type`. Do not derive corner labels later from a generic bounding
   box.
4. **Train a keypoint-capable model.** A four-keypoint pose model is the
   intended implementation. It should produce keypoint coordinates, pallet
   confidence, and pallet type.
5. **Implement the adapter.** Wrap model inference behind `PalletDetector`:

```python
import numpy as np

from fruit_pipeline.pallet_geometry.detector import (
    PalletDetection,
    PalletDetector,
)


class ProductionPalletDetector(PalletDetector):
    def __init__(self, weights_path: str):
        # TODO: load the selected keypoint model once here.
        self.model = load_pallet_keypoint_model(weights_path)

    def detect(self, image_bgr: np.ndarray) -> PalletDetection | None:
        # TODO: replace this with real model inference.
        prediction = self.model.predict(image_bgr)
        if prediction is None:
            return None

        corners = np.asarray(
            [
                prediction.top_left,
                prediction.top_right,
                prediction.bottom_right,
                prediction.bottom_left,
            ],
            dtype=np.float32,
        )
        return PalletDetection(
            corners_px=corners,
            confidence=float(prediction.confidence),
            pallet_type=str(prediction.pallet_type),
        )
```

6. **Validate ordering and confidence.** Incorrect corner ordering can rotate,
   mirror, or invalidate the physical coordinate system. Evaluate corner error
   per keypoint, not only pallet detection precision.
7. **Test difficult frames.** Reject detections with missing/duplicate/crossed
   corners or confidence below the configured threshold. Do not invent missing
   corners from a bounding box in production.

The code interface and validation exist now; the trained model and its adapter
are the primary unfinished pallet-detection steps.

## 6. Pallet parameters required for homography

Known pallet dimensions are stored separately from camera calibration in
`config/pallet_types.yaml`:

```yaml
pallet_types:
  standard_large:
    width_mm: 1200
    length_mm: 1800

  standard_small:
    width_mm: 1000
    length_mm: 1200
```

For every supported pallet type, provide:

- `pallet_type`: the exact name returned by the pallet detector.
- `width_mm`: physical distance from top-left to top-right.
- `length_mm`: physical distance from top-left to bottom-left.
- Four detected image points in TL, TR, BR, BL order.
- The calibration identity: `camera_id` and optional `camera_group`.
- An image at exactly the calibrated resolution.

The homography uses these local pallet coordinates, in millimetres:

```text
(0, 0) -------- (width_mm, 0)
   |                   |
   |                   |
(0, length_mm) -- (width_mm, length_mm)
```

Global pallet XYZ coordinates, camera world position, and one global
`mm_per_pixel` value are not required. Scale changes across a perspective
image, which is why all contour points are transformed through the homography.

Measure the same physical edges used by the keypoint annotations. If the YAML
dimensions describe the outside pallet edges but annotations use inset deck
corners, every fruit measurement will inherit a systematic scale error.

## 7. Run the size-estimation stage

The normal detection pipeline remains responsible for producing the existing
fruit instance masks. The size stage then performs:

1. Load camera-specific calibration, falling back to camera-group calibration.
2. Verify calibration error and image resolution.
3. Run the injected pallet keypoint detector.
4. Look up `width_mm` and `length_mm` for the detected pallet type.
5. Undistort the pallet corner points.
6. Compute the image-to-pallet homography.
7. Extract each fruit's segmentation contour.
8. Undistort every contour point and transform it into pallet millimetres.
9. Compute rotated length and width with `cv2.minAreaRect()`.
10. Compute projected area and equivalent diameter.
11. Optionally save the debug overlay, rectified pallet, and measurement JSON.

Until a dedicated size-estimation CLI is added, call it from Python after the
existing segmentation pipeline:

```python
import cv2

from fruit_pipeline.pipeline import PipelineConfig, run_pipeline
from fruit_pipeline.size_estimation import (
    SizeEstimationConfig,
    SizeEstimationPipeline,
)

image_path = "data/frame_001.jpg"

fruit_instances = run_pipeline(
    PipelineConfig(
        image_path=image_path,
        output_dir="outputs/segmentation",
    )
)

# TODO: replace with the completed four-keypoint model adapter from section 5.
pallet_detector = ProductionPalletDetector("models/pallet_keypoints.pt")

size_pipeline = SizeEstimationPipeline(
    SizeEstimationConfig(
        camera_id="cam_001",
        camera_group="camera_model_A",
        calibration_dir="calibrations",
        pallet_config_path="config/pallet_types.yaml",
        min_pallet_confidence=0.5,
        max_calibration_error=2.0,
        debug=True,
    ),
    pallet_detector=pallet_detector,
)

image_bgr = cv2.imread(image_path)
result = size_pipeline.run(image_bgr, fruit_instances)
result.save("outputs/size", stem="frame_001")
```

The saved files are:

```text
outputs/size/
├── frame_001_measurements.json
├── frame_001_measurement_debug.png
└── frame_001_rectified_pallet.png
```

Each fruit measurement contains:

```json
{
  "fruit_id": 1,
  "width_mm": 72.4,
  "length_mm": 78.1,
  "area_mm2": 4210.5,
  "equivalent_diameter_mm": 73.2,
  "confidence": 0.86
}
```

## 8. Remaining validation work

Before treating the measurements as production-ready:

1. Build a dataset containing images and manually measured fruit dimensions.
2. Include fruit near the pallet center, boundaries, and image corners.
3. Report absolute error in millimetres and percentage error by camera,
   pallet type, fruit size, and image region.
4. Compare results before and after distortion correction.
5. Measure sensitivity to pallet corner annotation error.
6. Select acceptable reprojection-error and pallet-confidence thresholds from
   the validation results.
7. Check whether fruit significantly above the pallet reference plane creates
   unacceptable perspective bias. A planar homography cannot correct height
   parallax.
8. Add the trained detector and validated thresholds to automated integration
   tests.

## 9. Tutorial update checklist

Update this document when completing later stages:

- [ ] Record the final pallet annotation convention.
- [ ] Document the pallet keypoint dataset format.
- [ ] Add the selected model architecture and training command.
- [ ] Replace the detector adapter pseudocode with the actual class and CLI.
- [ ] Document pallet-model weights and their expected location.
- [ ] Add a complete end-to-end command.
- [ ] Record calibrated acceptance thresholds and physical validation results.
- [ ] Add troubleshooting examples from real cameras.
