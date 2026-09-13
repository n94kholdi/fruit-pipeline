#!/usr/bin/env python3
"""Real-GPU SAM/SAM2 video benchmark with separated stage timings.

Run this explicitly on representative production videos; normal CI never
loads a checkpoint. One invocation benchmarks the one container-selected SAM2
variant, avoiding simultaneous model allocations.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path

import cv2
import torch

from fruit_pipeline.segmentation.sam2_config import SAM2Config
from fruit_pipeline.segmentation.sam2_manager import SAM2ModelManager


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("video")
    result.add_argument("--frame-step", type=int, default=10)
    result.add_argument("--max-processed-frames", type=int, default=100)
    result.add_argument("--output", default="sam2_benchmark.json")
    result.add_argument(
        "--memory-trace", action="store_true",
        help="Print one line per processed frame (detected/tracked object "
             "counts plus CUDA allocated/reserved/max-allocated) instead of "
             "the discovery/propagation scenario report. Use this to check "
             "that VRAM stabilizes instead of growing across a long run.",
    )
    result.add_argument(
        "--sweep-discovery", action="store_true",
        help="Run one discovery pass per (points_per_side, crop_n_layers) "
             "combo on the first frame and report latency + raw/final mask "
             "counts, instead of the default scenario report. Use this "
             "before assuming a denser grid is necessary.",
    )
    result.add_argument(
        "--sweep-batch-size", action="store_true",
        help="Run the full scenario benchmark once per "
             "tracking_object_batch_size value and report registration/"
             "propagation latency plus peak VRAM for each, instead of the "
             "default scenario report.",
    )
    result.add_argument(
        "--batch-sizes", default="8,16,32,64",
        help="Comma-separated tracking_object_batch_size values for --sweep-batch-size.",
    )
    result.add_argument(
        "--realistic", action="store_true",
        help="Realistic long-sequence benchmark: discovery every "
             "--refresh-processed-frames processed frames, several hundred "
             "frames. Reports discovery latency, average propagation "
             "latency, average FPS, discovered/tracked counts, peak VRAM, "
             "and VRAM over time.",
    )
    result.add_argument("--refresh-processed-frames", type=int, default=30)
    result.add_argument(
        "--warmup-propagations", type=int, default=1,
        help="Exclude this many initial propagation calls from the steady-state summary.",
    )
    result.add_argument(
        "--video-matrix", action="store_true",
        help="Run realistic benchmarks for VOS optimization off/on and CPU-state "
             "offload on/off. The GPU-resident-state cases may OOM on small cards.",
    )
    result.add_argument(
        "--model-sweep", action="store_true",
        help="Discover once with Base+, then propagate the exact same masks with "
             "SAM2.1 Tiny, Small, and Base+ (all checkpoints must be available).",
    )
    result.add_argument(
        "--implementation-label", default="new-predictor",
        help="Label embedded in output (use old-commit/new-unoptimized/new-optimized "
             "when comparing separately built container images).",
    )
    return result


def summarize(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    percentile = lambda q: ordered[min(len(ordered) - 1, round((len(ordered) - 1) * q))]
    mean = statistics.fmean(values) if values else 0.0
    return {
        "samples": len(values), "mean_ms": mean,
        "p50_ms": percentile(.50) if values else 0.0,
        "p95_ms": percentile(.95) if values else 0.0,
        "p99_ms": percentile(.99) if values else 0.0,
        "fps": 1000 / mean if mean else 0.0,
    }


def run_scenario(config: SAM2Config, video: str, frame_step: int, limit: int,
                 scenario: str) -> dict[str, object]:
    manager = SAM2ModelManager(config)
    capture = cv2.VideoCapture(video)
    ok, first = capture.read()
    if not ok:
        raise ValueError(f"Cannot read {video}")
    manager.start_camera(f"benchmark-{scenario}", video, first.shape[:2])
    discovery, propagation, counts, end_to_end = [], [], [], []
    index = processed = 0
    frame = first
    try:
        while ok and processed < limit:
            if index % frame_step == 0:
                started = time.perf_counter()
                instances, timing = manager.process_frame(
                    f"benchmark-{scenario}", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), index,
                    force_refresh=scenario == "sam2_discovery_every_frame",
                )
                end_to_end.append((time.perf_counter() - started) * 1000)
                if timing.discovery_ms:
                    discovery.append(timing.discovery_ms)
                if timing.propagation_ms:
                    propagation.append(timing.propagation_ms)
                counts.append(len(instances))
                processed += 1
            ok, frame = capture.read()
            index += 1
    finally:
        capture.release()
        manager.stop_camera(f"benchmark-{scenario}")
    return {
        "scenario": scenario,
        "discovery": summarize(discovery),
        "propagation": summarize(propagation),
        "end_to_end": summarize(end_to_end),
        "mean_count": statistics.fmean(counts) if counts else 0,
        "model": manager.status(),
    }


def run_memory_trace(config: SAM2Config, video: str, frame_step: int, limit: int) -> None:
    """Print one line per processed frame: object counts plus CUDA memory.

    A quick manual check that GPU memory stabilizes across a long stream
    instead of growing until CUDA OOM. Pair with ``SAM2_DEBUG_MEMORY=1`` for
    the manager's own lifecycle-boundary memory logs (discovery/propagation/
    state reset) alongside this table.
    """
    config = replace(config, debug_memory=True)
    manager = SAM2ModelManager(config)
    capture = cv2.VideoCapture(video)
    ok, first = capture.read()
    if not ok:
        raise ValueError(f"Cannot read {video}")
    manager.start_camera("memory-trace", video, first.shape[:2])
    device = manager.device
    print(f"{'frame':>6} {'detected':>9} {'tracked':>8} {'alloc_mb':>10} {'reserved_mb':>12} {'max_alloc_mb':>13}")
    index = processed = 0
    frame = first
    try:
        while ok and processed < limit:
            if index % frame_step == 0:
                instances, timing = manager.process_frame(
                    "memory-trace", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), index,
                )
                detected = len(instances) if timing.discovery_ms else 0
                tracked = len(instances) if timing.propagation_ms else 0
                if device.type == "cuda" and torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(device) / 1024**2
                    reserved = torch.cuda.memory_reserved(device) / 1024**2
                    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
                else:
                    allocated = reserved = max_allocated = 0.0
                print(f"{index:6d} {detected:9d} {tracked:8d} {allocated:10.1f} {reserved:12.1f} {max_allocated:13.1f}")
                processed += 1
            ok, frame = capture.read()
            index += 1
    finally:
        capture.release()
        manager.stop_camera("memory-trace")


def _peak_vram_mb(device) -> float:
    import torch as _torch

    if device.type != "cuda" or not _torch.cuda.is_available():
        return 0.0
    return _torch.cuda.max_memory_allocated(device) / 1024**2


def _reset_peak_vram(device) -> None:
    import torch as _torch

    if device.type == "cuda" and _torch.cuda.is_available():
        _torch.cuda.reset_peak_memory_stats(device)


def sweep_discovery(base: SAM2Config, video: str,
                    combos: list[tuple[int, int]]) -> list[dict[str, object]]:
    """One discovery pass per (points_per_side, crop_n_layers) combo, on the
    first frame only. Reports latency + raw/final mask counts so a denser
    grid or extra crop layer is a measured choice, not an assumption.
    """
    capture = cv2.VideoCapture(video)
    ok, first = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"Cannot read {video}")
    image_rgb = cv2.cvtColor(first, cv2.COLOR_BGR2RGB)
    rows = []
    for points_per_side, crop_n_layers in combos:
        config = replace(base, points_per_side=points_per_side, crop_n_layers=crop_n_layers,
                         debug_memory=True, debug_timing=True)
        manager = SAM2ModelManager(config)
        _reset_peak_vram(manager.device)
        instances, timing = manager.discover_image(image_rgb)
        rows.append({
            "points_per_side": points_per_side,
            "crop_n_layers": crop_n_layers,
            "discovery_ms": round(timing.discovery_ms, 1),
            "mask_generation_ms": round(timing.mask_generation_ms, 1),
            "filtering_ms": round(timing.filtering_ms, 1),
            "raw_masks": manager.last_raw_mask_count,
            "final_fruit_count": len(instances),
            "peak_vram_mb": round(_peak_vram_mb(manager.device), 1),
        })
        manager.unload()
    return rows


def sweep_tracking_batch_size(base: SAM2Config, video: str, frame_step: int, limit: int,
                              batch_sizes: list[int]) -> list[dict[str, object]]:
    """Run the full discovery+propagation scenario once per
    tracking_object_batch_size and report registration/propagation latency
    plus peak VRAM, so the batch size is chosen from measurements rather
    than assumed to help just because objects are divided into groups.
    """
    rows = []
    for batch_size in batch_sizes:
        config = replace(base, tracking_object_batch_size=batch_size, debug_timing=True)
        manager = SAM2ModelManager(config)
        _reset_peak_vram(manager.device)
        capture = cv2.VideoCapture(video)
        ok, first = capture.read()
        if not ok:
            raise ValueError(f"Cannot read {video}")
        camera_id = f"batch-{batch_size}"
        manager.start_camera(camera_id, video, first.shape[:2])
        registration, propagation = [], []
        index = processed = 0
        frame = first
        try:
            while ok and processed < limit:
                if index % frame_step == 0:
                    instances, timing = manager.process_frame(
                        camera_id, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), index,
                    )
                    if timing.registration_ms:
                        registration.append(timing.registration_ms)
                    if timing.propagation_ms:
                        propagation.append(timing.propagation_ms)
                    processed += 1
                ok, frame = capture.read()
                index += 1
        finally:
            capture.release()
            manager.stop_camera(camera_id)
        rows.append({
            "tracking_object_batch_size": batch_size,
            "registration": summarize(registration),
            "propagation": summarize(propagation),
            "peak_vram_mb": round(_peak_vram_mb(manager.device), 1),
        })
    return rows


def _timing_row(timing) -> dict[str, float | int]:
    return {
        "objects": timing.tracked_objects,
        "propagation_number": timing.propagation_number,
        "feature_encoding_ms": timing.frame_feature_encoding_ms,
        "sam2_propagation_ms": timing.sam2_propagation_ms,
        "mask_postprocessing_ms": timing.mask_postprocessing_ms,
        "mask_transfer_ms": timing.mask_transfer_ms,
        "bbox_extraction_ms": timing.bbox_extraction_ms,
        "propagation_total_ms": timing.propagation_ms,
        "frame_total_ms": timing.total_frame_ms,
        "cuda_allocated_mb": timing.cuda_allocated_mb,
        "cuda_reserved_mb": timing.cuda_reserved_mb,
        "cuda_peak_mb": timing.cuda_peak_mb,
    }


def run_realistic_benchmark(base: SAM2Config, video: str, frame_step: int, limit: int,
                            refresh_processed_frames: int, warmup_propagations: int = 1,
                            implementation_label: str = "new-predictor") -> dict[str, object]:
    """Discovery every ``refresh_processed_frames`` processed frames, over a
    long run. Reports the numbers the acceptance criteria actually care
    about: discovery latency, average propagation latency, average FPS over
    the whole sequence, discovered/tracked fruit counts, peak VRAM, and how
    VRAM evolved over time (to confirm it plateaus instead of growing).
    """
    config = replace(base, refresh_seconds=0, refresh_processed_frames=refresh_processed_frames,
                     debug_memory=True, debug_timing=True)
    manager = SAM2ModelManager(config)
    _reset_peak_vram(manager.device)
    startup_started = time.perf_counter()
    manager.load()
    model_startup_ms = (time.perf_counter() - startup_started) * 1000
    capture = cv2.VideoCapture(video)
    decode_started = time.perf_counter()
    ok, first = capture.read()
    decode_ms = [(time.perf_counter() - decode_started) * 1000]
    if not ok:
        raise ValueError(f"Cannot read {video}")
    predictor_started = time.perf_counter()
    manager.start_camera("realistic", video, first.shape[:2])
    predictor_init_ms = (time.perf_counter() - predictor_started) * 1000
    discovery, registration, propagation, end_to_end = [], [], [], []
    feature_encoding, sam2_propagation = [], []
    mask_postprocessing, mask_transfer, bbox_extraction = [], [], []
    propagation_rows = []
    discovered_counts, tracked_counts = [], []
    uncertain_counts = []
    vram_over_time = []  # (processed_frame_number, allocated_mb, reserved_mb)
    index = processed = 0
    frame = first
    try:
        while ok and processed < limit:
            if index % frame_step == 0:
                started = time.perf_counter()
                instances, timing = manager.process_frame(
                    "realistic", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), index,
                )
                end_to_end.append((time.perf_counter() - started) * 1000)
                if timing.discovery_ms:
                    discovery.append(timing.discovery_ms)
                    registration.append(timing.registration_ms)
                    discovered_counts.append(len(instances))
                if timing.propagation_ms:
                    propagation.append(timing.propagation_ms)
                    feature_encoding.append(timing.frame_feature_encoding_ms)
                    sam2_propagation.append(timing.sam2_propagation_ms)
                    mask_postprocessing.append(timing.mask_postprocessing_ms)
                    mask_transfer.append(timing.mask_transfer_ms)
                    bbox_extraction.append(timing.bbox_extraction_ms)
                    propagation_rows.append(_timing_row(timing))
                    tracked_counts.append(len(instances))
                    uncertain_counts.append(sum(
                        item.tracking_state == "uncertain" for item in instances
                    ))
                device = manager.device
                if device.type == "cuda" and torch.cuda.is_available():
                    vram_over_time.append((
                        processed,
                        round(torch.cuda.memory_allocated(device) / 1024**2, 1),
                        round(torch.cuda.memory_reserved(device) / 1024**2, 1),
                    ))
                processed += 1
            decode_started = time.perf_counter()
            ok, frame = capture.read()
            decode_ms.append((time.perf_counter() - decode_started) * 1000)
            index += 1
    finally:
        capture.release()
        manager.stop_camera("realistic")
    mean_end_to_end = statistics.fmean(end_to_end) if end_to_end else 0.0
    warmup = propagation_rows[:warmup_propagations]
    steady_propagation = propagation[warmup_propagations:]
    return {
        "implementation": implementation_label,
        "model": config.model_name,
        "vos_optimized": config.vos_optimized,
        "offload_video_to_cpu": config.offload_video_to_cpu,
        "offload_state_to_cpu": config.offload_state_to_cpu,
        "processed_frames": processed,
        "refresh_processed_frames": refresh_processed_frames,
        "model_startup_ms": round(model_startup_ms, 1),
        "predictor_init_ms": round(predictor_init_ms, 1),
        "video_decode": summarize(decode_ms),
        "discovery": summarize(discovery),
        "registration": summarize(registration),
        "first_or_warmup_propagations": warmup,
        "steady_state_propagation": summarize(steady_propagation),
        "all_propagation": summarize(propagation),
        "stage_timings": {
            "frame_feature_encoding": summarize(feature_encoding[warmup_propagations:]),
            # Includes SAM2's internal state prefetch/offload operations. The
            # on/off matrix quantifies their incremental cost without fragile
            # monkey-patching of Tensor.to inside upstream SAM2.
            "sam2_propagation_including_state_transfer": summarize(
                sam2_propagation[warmup_propagations:]
            ),
            "mask_postprocessing": summarize(mask_postprocessing[warmup_propagations:]),
            "mask_gpu_to_cpu_transfer": summarize(mask_transfer[warmup_propagations:]),
            "bbox_extraction": summarize(bbox_extraction[warmup_propagations:]),
        },
        "average_fps_over_sequence": 1000 / mean_end_to_end if mean_end_to_end else 0.0,
        "mean_discovered_fruit_count": statistics.fmean(discovered_counts) if discovered_counts else 0,
        "mean_tracked_fruit_count": statistics.fmean(tracked_counts) if tracked_counts else 0,
        "crowded_target_200_to_400_met": bool(
            discovered_counts and 200 <= statistics.fmean(discovered_counts) <= 400
        ),
        "mean_uncertain_object_count": statistics.fmean(uncertain_counts) if uncertain_counts else 0,
        "peak_vram_mb": round(_peak_vram_mb(manager.device), 1),
        "vram_over_time": vram_over_time,
    }


def run_video_matrix(base: SAM2Config, args) -> list[dict[str, object]]:
    rows = []
    for optimized, offload_state in ((False, True), (True, True), (False, False), (True, False)):
        config = replace(base, vos_optimized=optimized, offload_state_to_cpu=offload_state)
        label = f"new-v2-optimized={int(optimized)}-state_offload={int(offload_state)}"
        try:
            rows.append(run_realistic_benchmark(
                config, args.video, args.frame_step, args.max_processed_frames,
                args.refresh_processed_frames, args.warmup_propagations, label,
            ))
        except torch.cuda.OutOfMemoryError as exc:
            rows.append({
                "implementation": label,
                "error": "CUDA out of memory",
                "detail": str(exc),
            })
            torch.cuda.empty_cache()
    return rows


def run_model_sweep(base: SAM2Config, args) -> dict[str, object]:
    """Compare tracker sizes using one shared Base+ discovery result.

    This intentionally disables rediscovery: otherwise each model could start
    from a different set of objects and propagation latency/retention would no
    longer be comparable.
    """
    capture = cv2.VideoCapture(args.video)
    ok, first = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"Cannot read {args.video}")
    first_rgb = cv2.cvtColor(first, cv2.COLOR_BGR2RGB)
    discovery_config = replace(
        base, model_name="sam2.1_hiera_base_plus", checkpoint=None,
        config_file=None, vos_optimized=False,
    )
    discovery_manager = SAM2ModelManager(discovery_config)
    seed_instances, discovery_timing = discovery_manager.discover_image(first_rgb)
    seed_count = len(seed_instances)
    discovery_manager.unload()

    results = []
    for model_name in (
        "sam2.1_hiera_tiny", "sam2.1_hiera_small", "sam2.1_hiera_base_plus",
    ):
        config = replace(
            base, model_name=model_name, checkpoint=None, config_file=None,
            refresh_seconds=0, refresh_processed_frames=0,
            debug_memory=True, debug_timing=True,
        )
        manager = SAM2ModelManager(config)
        _reset_peak_vram(manager.device)
        capture = cv2.VideoCapture(args.video)
        ok, _first = capture.read()
        if not ok:
            raise ValueError(f"Cannot read {args.video}")
        camera_id = f"model-sweep-{model_name}"
        state = manager.start_camera(camera_id, args.video, first.shape[:2])
        state.instances = [replace(item) for item in seed_instances]
        with torch.inference_mode(), manager._autocast():
            registration_ms = manager._register_instances(state, 0)
        state.last_discovery = time.monotonic()
        state.processed_frames = 1
        propagation = []
        sam2_only = []
        retained = []
        uncertain = []
        index = 1
        processed = 0
        try:
            while processed < args.max_processed_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                if index % args.frame_step == 0:
                    instances, timing = manager.process_frame(
                        camera_id, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), index,
                    )
                    propagation.append(timing.propagation_ms)
                    sam2_only.append(timing.sam2_propagation_ms)
                    uncertain.append(sum(
                        item.tracking_state == "uncertain" for item in instances
                    ))
                    retained.append(sum(
                        item.tracking_state != "uncertain" for item in instances
                    ))
                    processed += 1
                index += 1
        finally:
            capture.release()
            manager.stop_camera(camera_id)
        warmup = args.warmup_propagations
        results.append({
            "model": model_name,
            "vos_optimized": config.vos_optimized,
            "seed_object_count": seed_count,
            "registration_ms": round(registration_ms, 1),
            "first_or_warmup_propagation_ms": propagation[:warmup],
            "steady_state_propagation": summarize(propagation[warmup:]),
            "steady_state_sam2_only": summarize(sam2_only[warmup:]),
            "mean_retained_objects": statistics.fmean(retained) if retained else 0,
            "mean_uncertain_objects": statistics.fmean(uncertain) if uncertain else 0,
            "peak_vram_mb": round(_peak_vram_mb(manager.device), 1),
        })
    return {
        "shared_discovery_model": "sam2.1_hiera_base_plus",
        "shared_discovery_ms": round(discovery_timing.discovery_ms, 1),
        "shared_seed_object_count": seed_count,
        "results": results,
    }


def main() -> int:
    args = parser().parse_args()
    base = SAM2Config.from_env()
    if args.model_sweep:
        report = run_model_sweep(base, args)
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0
    if args.video_matrix:
        report = run_video_matrix(base, args)
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0
    if args.memory_trace:
        run_memory_trace(base, args.video, args.frame_step, args.max_processed_frames)
        return 0
    if args.sweep_discovery:
        combos = [(32, 0), (48, 0), (64, 0), (48, 1)]
        rows = sweep_discovery(base, args.video, combos)
        print(json.dumps(rows, indent=2))
        return 0
    if args.sweep_batch_size:
        batch_sizes = [int(value) for value in args.batch_sizes.split(",")]
        rows = sweep_tracking_batch_size(
            base, args.video, args.frame_step, args.max_processed_frames, batch_sizes,
        )
        print(json.dumps(rows, indent=2))
        return 0
    if args.realistic:
        report = run_realistic_benchmark(
            base, args.video, args.frame_step, args.max_processed_frames,
            args.refresh_processed_frames, args.warmup_propagations,
            args.implementation_label,
        )
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0
    scenarios = [
        ("sam2_discovery_every_frame", base),
        ("sam2_refresh_30_processed_frames", SAM2Config(**{
            **base.__dict__, "refresh_seconds": 0, "refresh_processed_frames": 30,
        })),
        ("sam2_adaptive_refresh", base),
    ]
    report = {
        "video": args.video,
        "variant": base.model_name,
        "results": [run_scenario(config, args.video, args.frame_step,
                                 args.max_processed_frames, name)
                    for name, config in scenarios],
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
