# SAM1 → TensorRT: feasibility notes (not implemented)

This is an investigation, not an implementation: TensorRT export/inference
for SAM1 is not wired into `fruit-pipeline` yet. It documents blockers,
what would need to change, and how the code is already positioned to make
that change additive later.

## Why not now

- `segment_anything` is an unmodified upstream pip dependency (see
  `sam_manager.py` module docstring), not vendored into this repo. Exporting
  it to ONNX/TensorRT reliably usually means controlling the source (to swap
  out or annotate dynamic-control-flow ops), so this would start as a fork or
  a vendored copy -- a real maintenance decision, not a config flag.
- `SAMModelManager` currently drives the model in two separate stages
  (`image_encoder`, then `prompt_encoder` + `mask_decoder` per box batch --
  see `sam_manager.py:encode_image`/`decode_boxes`). A TensorRT engine is a
  fixed input/output graph; this two-call, variable-box-batch-size shape
  doesn't map onto one engine and needs the encoder and decoder exported (and
  invoked) as separate engines, matching how Meta's own `onnx_model_example`
  in `segment-anything` splits them.
- The mask decoder's box-prompt path takes a **variable number of boxes per
  image** (`decode_boxes(boxes, batch_size=16, ...)`, chunked). TensorRT
  engines want fixed or bucketed shapes; this needs either a fixed max batch
  with padding/masking, or a small set of pre-built engines per bucket size
  (e.g. 1/4/16/64 boxes), adding real complexity to `decode_boxes`.
- The ViT image encoder's relative-position attention
  (`image_encoder.Attention`, see `sam_attention_patch.py` docstring) does a
  data-dependent `einsum`-based bias before softmax. That op traces fine to
  ONNX but is one more place ONNX/TensorRT op coverage needs to be checked
  per TensorRT version; it is exactly the same code path FlashAttention
  cannot use either (§6 of the accompanying report).
- Production pins `torch==2.5.1+cu118` with **no `tensorrt` or `onnx`
  dependency anywhere** in `pyproject.toml`/the Dockerfile (`fruit-pipeline`
  audit, confirmed via `grep`); the base image would need a TensorRT install
  matching that CUDA 11.8 toolchain, which is its own compatibility surface
  to validate (TensorRT version support windows move faster than CUDA 11.8
  stays current).
- No existing `.onnx` export, `trtexec` script, or `torch2trt` usage exists
  in this repo to build on (confirmed via repo-wide grep) -- this would be
  new infrastructure, not a wire-up of something partially done.

## What would need to change, if pursued

1. **Fork or vendor `segment_anything`** (or adopt a maintained
   TensorRT-friendly SAM fork) so the encoder/decoder graphs can be exported
   and, if needed, patched for ONNX op coverage.
2. **Export two engines**: image encoder (fixed `1024x1024x3` input, as SAM
   already preprocesses to -- see `encode_image()`) and mask decoder/prompt
   encoder (fixed or bucketed box-prompt batch size).
3. **Add a TensorRT execution path in `SAMModelManager`**, parallel to the
   current PyTorch path, selected the same way `use_fp16`/`use_compile`/
   `use_sdpa_attention` are today -- i.e. an opt-in constructor flag read from
   an env var (`FRUIT_PIPELINE_SAM_RUNTIME=pytorch|tensorrt`, mirroring the
   `SAM2_RUNTIME` naming already used by this repo's SAM2 branch, so the
   convention is consistent across both) with a hard fallback to the PyTorch
   path if engine loading fails at startup -- never a hard dependency.
4. **Numerically validate** the exported engines against the eager PyTorch
   path the same way `scripts/benchmark_sam.py` already validates FP16 vs
   FP32 today (mask IoU, pixel agreement, score delta) before trusting it in
   production.
5. **Rebuild/version the engines per GPU architecture** (TensorRT engines are
   not portable across GPU generations/driver versions the way a `.pth`
   checkpoint is), which has deployment implications for
   `tarebar-deployment` (engine build/cache step, likely in
   `scripts/download-models.sh` or a new provisioning step).

## Why the current code doesn't block this later

- `SAMModelManager` already isolates "encode" and "decode" as two distinct,
  independently timed calls (`encode_image`/`decode_boxes`) rather than one
  opaque `predict()` -- the natural seam for two separate engines.
- `use_compile`/`use_sdpa_attention` establish the pattern a future
  `use_tensorrt` flag would follow: constructor kwarg, env-driven default
  threaded through `PipelineConfig`/`SamOnlyConfig`/`dashboard_api.py`, try
  the optimization, log and fall back to eager PyTorch on any failure. No
  interface changes would be needed in `sam.py`/`sam_auto.py`/the pipelines
  above `SAMModelManager`.

## Bottom line

Feasible, but it is a multi-week infrastructure project (forking/vendoring
the model source, export tooling, engine build/versioning, deployment
changes), not a config flag -- correctly out of scope for this pass. The
`FRUIT_PIPELINE_SAM_TORCH_COMPILE` work done here already captures most of
the kernel-fusion upside `torch.compile` and TensorRT both chase, at a
fraction of the integration cost.
