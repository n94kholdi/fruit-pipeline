# Manual pallet-plane size validation

This temporary workflow validates camera calibration, pallet homography, and
planar object measurement before a pallet keypoint model is available. Manual
clicking is isolated behind `PalletDetector`; it is not embedded in the
homography or measurement code.

The result is a **projection onto the pallet top plane**. It does not estimate
object height or 3D dimensions.

## 1. Configure the pallet

Add the physical top-surface dimensions to `config/pallet_types.yaml`:

```yaml
pallet_types:
  my_pallet:
    width_mm: 1200
    length_mm: 1800
```

`width_mm` is the TL-to-TR edge. `length_mm` is the TL-to-BL edge.

## 2. Select the pallet corners

After `pip install -e .`, run:

```bash
select-pallet-corners \
  --image path/to/frame.jpg \
  --pallet-type my_pallet \
  --output outputs/manual/frame_pallet.json
```

The preview is scaled to fit within 900 pixels by default while saved points
remain in original-image coordinates. On a smaller display, add—for example—
`--max-preview-size 650`.

Click exactly in this order:

1. top-left (TL)
2. top-right (TR)
3. bottom-right (BR)
4. bottom-left (BL)

Use Backspace or `U` to undo, `R` to reset, Enter to accept, and Escape to
cancel. The command saves both the original-resolution pixel coordinates and
an ordered-corner overlay. It also records the image resolution to prevent the
selection from accidentally being reused on a resized image.

The checkout script equivalent is:

```bash
python tools/select_pallet_corners.py ...
```

## 3. Select and measure an object

```bash
measure-pallet-object \
  --image path/to/frame.jpg \
  --pallet-corners outputs/manual/frame_pallet.json \
  --calibration-dir outputs/calibrations/camera_model_A/cam_iphone \
  --camera-id cam_iphone \
  --pallet-config config/pallet_types.yaml \
  --selection-mode polygon \
  --ground-truth-width-mm 90 \
  --ground-truth-length-mm 120 \
  --output-dir outputs/manual/frame_measurement
```

For a rectangular selection, use `--selection-mode bbox`. Polygon mode is
preferred when the object outline is visibly non-rectangular. Enter finishes a
polygon after at least three points.

The checkout script equivalent is:

```bash
python tools/measure_object.py ...
```

Outputs include:

- an untouched copy of the original image;
- the ordered pallet-corner overlay;
- the selected object with width, length, and projected area;
- a rectified bird's-eye pallet view;
- JSON containing raw pixel points, pallet-local millimetre points, the
  homography, measurements, and optional ground-truth errors.

Object contour points and pallet corners are undistorted using the selected
camera calibration before they are transformed. Width and length come from a
minimum-area rectangle in pallet-local millimetres; area comes from the
transformed contour, not an image bounding-box scale.

For headless/repeatable runs, both tools accept JSON coordinates:

```bash
select-pallet-corners ... --points-file pallet_points.json
measure-pallet-object ... --object-points object_points.json
```

Each file can be either an `[[x, y], ...]` array or an object with a
`points_px` field.

## Replacing manual selection

`ManualPalletDetector` is only one implementation of this interface:

```python
class PalletDetector(Protocol):
    def detect(self, image_bgr: np.ndarray) -> PalletDetection | None:
        ...
```

A keypoint model, classical vision method, or external service can replace it
by returning `PalletDetection(corners_px, confidence, pallet_type)` with
corners in TL, TR, BR, BL order. `compute_pallet_homography`, contour
measurement, and the production `SizeEstimationPipeline` remain unchanged.
