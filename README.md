# fruit_pipeline

Detects and segments individual fruits in images of pallets/boxes containing
many densely packed, small, touching fruits — using **pretrained models
only** (no fine-tuning), combined via tiling, merging, and prompted
segmentation.

Given one image, it outputs (all written flat into `--output_dir`, prefixed
with the image's filename stem — no per-image subfolders):
- per-fruit bounding boxes in **original image coordinates**
- one instance segmentation mask per fruit
- `<stem>_final.png` — the final result: boxes + colored mask overlays on
  the original image
- `<stem>_tiles.png` — a debugging view showing every SAHI tile's own crop
  with the detector's raw (pre-merge) boxes on it, laid out in the tiles'
  actual spatial grid, so you can see how the detector performs tile by tile
- `<stem>_detections.json` with box, score, and mask polygon per fruit, each
  keyed by a stable `instance_id`

It intentionally stops there. Classification, sizing, and rotten/fine
detection are separate future stages — see "Next stages" below.

## Project structure

```
src/fruit_pipeline/
├── detection/       detector backends, SAHI tiling, and cross-tile merging
├── segmentation/    SAM loading, box prompting, and mask filters
├── visualization/   final overlays and tile-debug rendering
├── config/          prompt loading and packaged YAML defaults
├── eval/            COCO adapters, metrics, and evaluation CLI
├── utils/           shared geometry and path helpers
├── pipeline.py      end-to-end orchestration for one image
├── cli.py           full-pipeline command
└── inference.py     standalone whole-image inference command
scripts/             dataset and annotation utilities
tests/               unit and CLI tests
```

The short modules such as `fruit_pipeline.detect` and
`fruit_pipeline.merge` are compatibility imports. New features should use
the focused subpackages, for example
`fruit_pipeline.detection.tiling` and
`fruit_pipeline.detection.merging`.

## Setup

### 1. Install the project

```bash
pip install -e .
```

For evaluation or development tools, use `pip install -e '.[eval]'` or
`pip install -e '.[dev]'` respectively. The editable install exposes the
`fruit-pipeline`, `fruit-inference`, and `fruit-eval` commands.
SAM is installed from Meta's official GitHub repository, matching its
upstream installation guidance rather than relying on an unrelated PyPI
package with a similar name.

### 2. Model weights (reuse what's already downloaded in this project)

| Model | Used for | Where this project already has it |
|---|---|---|
| A standard Ultralytics YOLO checkpoint (`yolo11x.pt`, `yolov8x.pt`, `models/yolo11m.pt`, `models/yolov8m.pt`, ...) | tiled object-like-region detector (class-agnostic) | project root / `models/` |
| SAM ViT-L checkpoint (`sam_vit_l_0b3195.pth`) | box-prompted segmentation | `models/sam_vit_l_0b3195.pth` |

Defaults (`--detector-weights yolo11x.pt`, `--sam-checkpoint
models/sam_vit_l_0b3195.pth`) point at checkpoints already present in this
project, so no downloads are required to run it as-is.

**YOLO-World is optional, not the default.** The prompt spec's first choice
for the detector is an open-vocabulary model like YOLO-World, prompted with
text like `"fruit"`, `"round fruit"`. This repo does have a YOLO-World
checkpoint (`yolov8s-world.pt`), but enabling it (`--use-yolo-world`)
triggers `ultralytics` to install an extra `CLIP` package and download a
~350MB CLIP text-encoder checkpoint on first use — a new download beyond
what's already in the project. The default path instead runs the plain YOLO
checkpoint in **class-agnostic** mode: every detection across all 80 COCO
classes is relabeled to a generic "fruit" category before merging, so
detection recall isn't limited to COCO's `apple` / `banana` / `orange`
classes. Use `--use-yolo-world` if you're fine with that one-time download
and want true open-vocabulary prompting.

SAM2 was not used here even though the repo has a `sam2.1_t.pt` checkpoint,
because the `sam2` package (plus its Hydra config files) isn't installed in
this environment, while `segment-anything` (SAM1) and a matching ViT-L
checkpoint already are. Swapping `segmentation/sam.py`'s `load_sam` for a SAM2
loader later is a contained change if that's ever worth it.

## Run

```bash
python -m fruit_pipeline.cli --image path/to/image.jpg --output_dir ./out
```

Produces `out/<stem>_detections.json`, `out/<stem>_final.png`, and
`out/<stem>_tiles.png`, and prints the total fruit count to the console.

### Whole-image detector inference (no tiling)

To inspect the detector's predictions on the complete image without SAHI,
merging, or SAM segmentation, use the standalone inference command:

```bash
python -m fruit_pipeline.inference \
  --image path/to/image.jpg \
  --weights models/best.pt \
  --conf-threshold 0.25 \
  --output-dir outputs/whole_image
```

`--image` may also be a directory (processed non-recursively). For every
input, this writes `<stem>_prediction.jpg` and `<stem>_detections.json`.

### Batch mode (a folder of images)

Pass a directory to `--image` instead of a single file to process every
image directly inside it (non-recursive). The detector and SAM are loaded
once and reused across all images; every image's outputs land flat in the
same `--output_dir` (no per-image subfolders), named by that image's stem:

```bash
python -m fruit_pipeline.cli --image data/fruits --output_dir fruit_pipeline/outputs/test1
```

This produces, for every `.jpg`/`.jpeg`/`.png`/`.bmp`/`.webp` file found
directly in `data/fruits/` (subfolders like `data/fruits/archive/` are not
descended into):
- `fruit_pipeline/outputs/test1/<stem>_final.png`
- `fruit_pipeline/outputs/test1/<stem>_tiles.png`
- `fruit_pipeline/outputs/test1/<stem>_detections.json`

A failure on one image is logged and skipped rather than aborting the whole
batch.

## CLI arguments

**Detection**
- `--detector-weights` (default `yolo11x.pt`): Ultralytics checkpoint path.
- `--use-yolo-world`: treat `--detector-weights` as YOLO-World and prompt it
  with `--prompt-classes` (see note above).
- `--prompt-classes` (default `fruit,round fruit,apple,orange,citrus fruit`):
  comma-separated text prompt, only used with `--use-yolo-world`.
- `--tile-size` (default: adaptive): SAHI tile side length in pixels. Left
  unset, it's estimated per image by a fast coarse pre-pass: the detector
  runs once on a downscaled (`--coarse-pass-long-edge`) copy of the image to
  measure the median fruit diameter, then `tile_size = --tile-size-k *
  diameter` (clamped to `[--min-tile-size, --max-tile-size]`). This keeps
  each tile large enough to contain many *whole* fruit — a fixed size tends
  to either split individual fruit across tile boundaries (too small) or
  blow up to 100+ tiles on an 8000px-wide crate photo (too many). If the
  pre-pass is degenerate (too few coarse detections, e.g. a near-empty
  crate), it falls back to `--fallback-tile-size` and logs a warning. Set
  `--tile-size` explicitly to **disable** the pre-pass entirely and force a
  fixed size regardless of resolution — mainly useful for debugging/
  comparing against the old fixed-size behavior.
- `--tile-size-k` (default `8.0`): adaptive tiling only — how many fruit-
  diameters wide a tile is. Try 6-10; lower means smaller/more tiles
  (more context isolation), higher means larger/fewer tiles (less compute).
- `--min-tile-size` / `--max-tile-size` (default `320` / `2048`): adaptive
  tiling only — clamp range for the estimated tile size, to avoid degenerate
  tiling on unusual images.
- `--coarse-pass-long-edge` (default `1400`): adaptive tiling only —
  resolution the image is downscaled to for the fruit-diameter pre-pass.
- `--coarse-max-box-area-fraction` (default `0.08`): adaptive tiling only —
  coarse-pass boxes covering more than this fraction of the downscaled image
  are dropped before computing the median diameter. On a busy, dense crate
  photo, a class-agnostic detector's coarse pass reliably returns a mix of
  individual-fruit boxes *and* a few boxes spanning a whole cluster/pile;
  without this filter those outliers drag the diameter estimate (and hence
  tile size) toward "cluster-sized" instead of "one fruit".
- `--fallback-tile-size` (default `640`): tile size used when the pre-pass
  can't produce a usable fruit-diameter estimate.
- `--max-tiles` (default `12`): advisory tile-count budget — logged as a
  warning if the adaptively-chosen tile size still produces more tiles than
  this. Not enforced by itself; use `--max-tile-size` to actually cap it.
- `--overlap-ratio` (default `0.15`): fractional overlap between tiles.
  Needs to be large enough that a fruit near a tile boundary is fully
  contained in at least one tile. With adaptive tile sizing, tiles rarely
  cut through a single fruit the way small fixed tiles did, so 10-15%
  overlap is usually enough — raise it if you see missed or double-counted
  fruit near tile seams (check `--debug-save-tiles` / `<stem>_tiles.png`).
- `--conf-threshold` (default `0.25`): per-tile detector confidence cutoff.
  The main lever for false positives (too low) vs. missed fruit (too high).
- `--no-standard-pred`: skip the extra full-image detection pass (normally
  run in addition to tiles, to help catch larger fruit).
- `--debug-save-tiles`: save every individual tile crop to
  `<output_dir>/tiles_debug/<image_stem>/` before it's handed to the
  detector, so tiling can be visually sanity-checked — confirms tiles aren't
  blank, duplicated, or out-of-bounds crops from an off-by-one in the tiling
  loop.
- `--two-resolution`: opt-in, off by default. Runs detection/tiling on a
  downscaled "working" copy of the image (`--working-long-edge`) instead of
  full resolution — cheaper tiling on very high-res images — then maps the
  merged boxes back to full-resolution coordinates before segmentation.
  Also saves a full-resolution crop per instance to `<output_dir>/crops/`,
  for a later sizing stage that needs true pixel precision without paying
  full-res detection cost.
- `--working-long-edge` (default `2800`): `--two-resolution` only — the
  long-edge resolution used for the downscaled detection pass. Ignored if
  the image is already smaller than this.

**Merge**
- `--merge-strategy` (default `greedy_nmm`): `greedy_nmm` merges
  overlapping/touching detections into one; `nmm` does the same non-greedily
  (transitive merging); `nms` is plain hard suppression — only use `nms` if
  you specifically want deletion instead of merging, since it can discard
  legitimate adjacent/touching fruit.
- `--merge-metric` (default `IOU`): `IOU` or `IOS` (intersection-over-
  smaller-area). SAHI's NMM/GreedyNMM *merges* a matched pair by taking the
  **union** of their boxes, not just keeping one — IOS was tried as the
  default first (more forgiving when a smaller sliced-tile detection sits
  inside a larger one, which helps merge the same fruit's duplicate
  detections across overlapping tiles), but on a dense crate of touching
  round fruit it also matches two *different* neighboring fruits whenever
  one's box is mostly contained in the other's, unioning them into one bad
  box — which then either gets caught by the oversized-box filter (both
  real fruits silently vanish) or produces a bad SAM mask that trips the
  edge/aspect filters. Measured on a real crate photo: IOS -> IOU recovered
  ~50% more distinct fruit post-merge. Switch back to `IOS` only if you're
  instead seeing duplicate boxes for the same fruit surviving into the
  final output.
- `--merge-iou-threshold` (default `0.5`): overlap above which two detections
  are treated as the same fruit.
- `--no-class-agnostic-merge`: merge per-category instead of across all
  categories (only matters if `--use-yolo-world` is prompted with multiple
  distinct classes you want kept separate).
- `--no-oversized-filter` / `--oversized-ratio` (default `3.0`): the
  crate-vs-single-fruit heuristic filter and its threshold (reject boxes
  larger than this multiple of the median detected box area in the image).

**Segmentation**
- `--sam-checkpoint` (default `models/sam_vit_l_0b3195.pth`).
- `--sam-model-type` (default `vit_l`): `vit_b` (fastest, lowest quality),
  `vit_l` (default, balanced), or `vit_h` (best quality, slowest/heaviest).
- `--sam-batch-size` (default `16`): boxes per batched SAM `predict_torch`
  call; raise/lower based on available GPU/CPU memory.

**Mask sanity filters**
- `--min-mask-area` (default `30` px): drop degenerate near-zero-area masks.
- `--no-border-filter` / `--border-touch-ratio` (default `0.6`): disable, or
  tune, the filter that rejects a mask hugging an entire image edge
  (background/crate wall picked up instead of a single fruit).
- `--no-aspect-ratio-filter` / `--max-aspect-ratio` (default `3.0`): the
  round/oval-fruit-shape heuristic and its threshold. This is a heuristic,
  not a learned rule — disable it for elongated fruit (bananas, etc.).

**Other**
- `--device` (default `auto`): `auto`, `cpu`, `cuda`, or `cuda:N`.
- `--no-visualization`: skip writing `visualization.png`.
- `-v` / `--verbose`: debug logging.

## Known limitations (pretrained-only, no fine-tuning)

- No fruit-specific weights exist, so recall/precision depend entirely on
  how well a general-purpose detector's "object-like region" boxes line up
  with actual fruit. Expect more false positives/negatives on unusual fruit
  types, heavy occlusion, or fruit that doesn't look round/convex.
- **The default (plain, class-agnostic) detector recalls uniformly-colored,
  densely-packed round fruit poorly — oranges/tangerines/mandarins being the
  clearest example.** Instead of individual boxes, it tends to draw one
  giant box around the *entire pile* (COCO's "orange" class matching "a mass
  of orange-colored roundness" rather than one fruit); that giant box then
  gets correctly dropped by the oversized-box filter, which is why an
  all-orange region can end up with *zero* fruit in the final output even
  though the crate is full of them. This is a detector-recall limitation,
  not a tiling/merge bug — verified by running the same crop through
  `--use-yolo-world` with the default `--prompt-classes` (which already
  includes `orange` and `citrus fruit`): raw per-tile detections on an
  orange-packed 2048x2048 crop went from 2 (one real box, one giant blob) to
  63 well-sized individual boxes. If your crates are dominated by
  citrus/round uniformly-colored fruit, `--use-yolo-world` is the fix to
  reach for — the CLIP text encoder it needs may already be cached locally
  from a prior run; if not, expect the one-time ~350MB download noted above.
- Without labeled data to tune against, the three levers that matter most
  are `--conf-threshold`, `--tile-size` (and `--overlap-ratio`), and the
  oversized-box filter (`--oversized-ratio`). Start there before touching
  anything else.
- The mask sanity filters (border-touch, aspect-ratio) are heuristics tuned
  by eyeballing typical crate photos, not learned thresholds — expect to
  adjust `--border-touch-ratio` / `--max-aspect-ratio` per camera setup, or
  disable them if they're rejecting valid fruit.
- SAM mask quality on tightly touching fruit is bounded by how tight the
  detector's box is; a loose box (covering part of a neighboring fruit) can
  make SAM bleed the mask into the neighbor.

## Running command

```
python -m fruit_pipeline.cli \                        
  --image data/test_fruits_HD \
  --output_dir outputs/yolo/adaptive_gpu_test \
  --detector-weights models/yolo11x.pt \
  --sam-checkpoint models/sam_vit_l_0b3195.pth \
  --device cuda:0 \
  --sam-batch-size 1 \
  --tile-size-k 8 \
  --min-tile-size 320 \
  --max-tile-size 2048 \
  --max-tiles 12 \
  --overlap-ratio 0.15 \
  --nms-metric diou \
  --merge-iou-threshold 0.5 \
  --containment-threshold 0 \
  --conf-threshold 0.05 \
  -v
```
## Next stages (not implemented here)

Classification (fruit type), sizing, and rotten/fine detection are meant to
be separate modules that consume `detections.json` from this stage, joining
on each record's `instance_id`. Nothing here needs to change to add them.
