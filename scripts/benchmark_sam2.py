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


def main() -> int:
    args = parser().parse_args()
    base = SAM2Config.from_env()
    if args.memory_trace:
        run_memory_trace(base, args.video, args.frame_step, args.max_processed_frames)
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
