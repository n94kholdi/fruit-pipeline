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


def run_realistic_benchmark(base: SAM2Config, video: str, frame_step: int, limit: int,
                            refresh_processed_frames: int) -> dict[str, object]:
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
    capture = cv2.VideoCapture(video)
    ok, first = capture.read()
    if not ok:
        raise ValueError(f"Cannot read {video}")
    manager.start_camera("realistic", video, first.shape[:2])
    discovery, propagation, end_to_end = [], [], []
    discovered_counts, tracked_counts = [], []
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
                    discovered_counts.append(len(instances))
                if timing.propagation_ms:
                    propagation.append(timing.propagation_ms)
                    tracked_counts.append(len(instances))
                device = manager.device
                if device.type == "cuda" and torch.cuda.is_available():
                    vram_over_time.append((
                        processed,
                        round(torch.cuda.memory_allocated(device) / 1024**2, 1),
                        round(torch.cuda.memory_reserved(device) / 1024**2, 1),
                    ))
                processed += 1
            ok, frame = capture.read()
            index += 1
    finally:
        capture.release()
        manager.stop_camera("realistic")
    mean_end_to_end = statistics.fmean(end_to_end) if end_to_end else 0.0
    return {
        "processed_frames": processed,
        "refresh_processed_frames": refresh_processed_frames,
        "discovery": summarize(discovery),
        "propagation": summarize(propagation),
        "average_fps_over_sequence": 1000 / mean_end_to_end if mean_end_to_end else 0.0,
        "mean_discovered_fruit_count": statistics.fmean(discovered_counts) if discovered_counts else 0,
        "mean_tracked_fruit_count": statistics.fmean(tracked_counts) if tracked_counts else 0,
        "peak_vram_mb": round(_peak_vram_mb(manager.device), 1),
        "vram_over_time": vram_over_time,
    }


def main() -> int:
    args = parser().parse_args()
    base = SAM2Config.from_env()
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
            args.refresh_processed_frames,
        )
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
