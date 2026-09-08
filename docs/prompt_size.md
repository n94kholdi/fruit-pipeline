Sure. Here is the whole prompt in a single copyable block:

````markdown
# Temporary pallet homography test implementation

Implement a simple experimental pipeline to validate camera calibration + homography based size estimation.

Do NOT implement pallet detection yet.

The pallet corners will be manually selected by the user from an image.

The goal is to verify that:
- camera calibration parameters
- pallet dimensions
- homography transformation

are sufficient to estimate real-world object dimensions.

---

## 1. Manual pallet corner selection

Create a simple tool/script that allows the user to manually click the four pallet corners in an image.

The user should select corners in this exact order:

1. top-left
2. top-right
3. bottom-right
4. bottom-left

Example output:

```python
image_points = [
    [x1, y1],
    [x2, y2],
    [x3, y3],
    [x4, y4]
]
````

Add visualization showing:

* selected points
* pallet polygon
* corner ordering

Save the selected pixel coordinates for later processing.

---

## 2. Pallet real-world configuration

The user provides the real pallet dimensions.

Example:

```yaml
pallet:
  width_mm: 1200
  length_mm: 1800
```

Create the corresponding real-world coordinates:

```python
real_points = [
    [0, 0],
    [1200, 0],
    [1200, 1800],
    [0, 1800]
]
```

Assume the pallet top surface is the measurement plane.

Do not require global XYZ location of the pallet.

Only use the pallet local coordinate system.

---

## 3. Camera calibration loading

Load the existing camera calibration parameters:

* camera intrinsic matrix K
* distortion coefficients

Undistort the image before calculating homography.

The calibration parameters should come from the existing camera calibration module.

---

## 4. Compute pallet homography

Using OpenCV, compute the transformation between:

* image pixel coordinates
* pallet real-world coordinates in millimeters

Example:

```python
H, status = cv2.findHomography(
    image_points,
    real_points
)
```

The transformation should map:

```
pixel coordinates
        ↓
pallet coordinates (mm)
```

The system should handle:

* camera angle
* pallet perspective distortion
* pallet slope relative to the camera

---

## 5. Object measurement test

For testing, do not use object detection yet.

Allow the user to manually select an object:

Options:

* draw a bounding box
* draw/select a contour
* provide polygon points

Example:

```
image object contour
        |
        ↓
homography transformation
        |
        ↓
real-world contour in mm
```

Transform the object points from image space into pallet coordinates.

Calculate:

* width in mm
* length in mm
* projected area in mm²

Use real-world transformed coordinates, not pixel measurements.

For width and length, use suitable geometry such as:

* minimum area rectangle
* principal axes

Do not simply use image bounding box dimensions.

---

## 6. Visualization and validation

Create debug outputs showing:

1. Original image
2. Selected pallet corners
3. Pallet polygon
4. Rectified bird-eye-view pallet
5. Selected object
6. Estimated measurements

Example:

```
Width: 85 mm
Length: 120 mm
Area: 9000 mm²
```

Also provide a way to compare estimated size against a manually entered ground truth size.

Example:

```
Real width: 90 mm
Estimated width: 87 mm
Error: 3.3%
```

---

## 7. Keep modules separate

Create a clean temporary structure:

```
geometry/
    pallet_homography.py
    measurement.py

tools/
    select_pallet_corners.py
    measure_object.py

configs/
    pallet.yaml
```

Do not couple this implementation with future pallet detection.

The purpose of this prototype is only:

```
camera calibration
        +
manual pallet corners
        +
known pallet dimensions
        ↓
homography
        ↓
real-world object measurement validation
```

After validation, the manual corner selection will later be replaced by a pallet corner/keypoint detection model.

```
```
