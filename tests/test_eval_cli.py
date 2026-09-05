"""End-to-end smoke test for the eval CLI: GT COCO JSON + a pred-dir of
'<stem>_detections.json' -> 'run' -> a result JSON -> 'compare'.

Uses the hand-built fixtures in tests/fixtures/ (tiny_gt.json + tiny_preds/),
with expected precision/recall/counting numbers computed by hand in the
comments below, so this test also catches wiring bugs (coco_io <-> metrics
<-> cli) that per-function unit tests wouldn't.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from fruit_pipeline.eval.cli import compare_command, run_command

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_run_command_writes_expected_metrics(tmp_path):
    output_path = tmp_path / "result.json"
    args = argparse.Namespace(
        gt=str(FIXTURES / "tiny_gt.json"),
        pred_dir=str(FIXTURES / "tiny_preds"),
        output=str(output_path),
        iou_thresh=0.5,
        label="test-run",
    )

    run_command(args)

    payload = json.loads(output_path.read_text())
    assert payload["num_images"] == 2
    assert payload["num_predictions"] == 5
    assert payload["unmatched_prediction_stems"] == []

    metrics = payload["metrics"]

    # tiny: 1 GT (img1), matched perfectly, no stray tiny predictions.
    assert metrics["tiny"]["precision"] == pytest.approx(1.0)
    assert metrics["tiny"]["recall"] == pytest.approx(1.0)

    # small: 1 GT (img1), matched perfectly, PLUS one stray FP also
    # small-sized (img1's [150,150,170,170]) -> precision=1/2, recall=1/1.
    assert metrics["small"]["precision"] == pytest.approx(0.5)
    assert metrics["small"]["recall"] == pytest.approx(1.0)

    # medium: 2 GT (img1 + img2), BOTH missed; img2 has one stray
    # medium-sized FP -> precision=0/1, recall=0/2.
    assert metrics["medium"]["precision"] == pytest.approx(0.0)
    assert metrics["medium"]["recall"] == pytest.approx(0.0)
    assert metrics["medium"]["fn"] == 2

    # large: 1 GT (img2), matched perfectly.
    assert metrics["large"]["precision"] == pytest.approx(1.0)
    assert metrics["large"]["recall"] == pytest.approx(1.0)

    # all: 5 GT total, 3 TP, 2 FP, 2 FN -> precision=recall=3/5.
    assert metrics["all"]["precision"] == pytest.approx(0.6)
    assert metrics["all"]["recall"] == pytest.approx(0.6)

    # Counting: total predicted (5) == total GT (5) per image, so "all"
    # bucket count-error nets to 0 even though FP/FN are individually nonzero.
    assert metrics["all"]["mae"] == pytest.approx(0.0)
    assert metrics["all"]["rmse"] == pytest.approx(0.0)

    for bucket in ("tiny", "small", "medium", "large", "all"):
        assert 0.0 <= metrics[bucket]["mAP50"] <= 1.0 or metrics[bucket]["mAP50"] == -1.0
        assert 0.0 <= metrics[bucket]["mAP50-95"] <= 1.0 or metrics[bucket]["mAP50-95"] == -1.0


def test_compare_command_against_itself_has_zero_deltas(tmp_path, capsys):
    output_path = tmp_path / "result.json"
    run_command(
        argparse.Namespace(
            gt=str(FIXTURES / "tiny_gt.json"),
            pred_dir=str(FIXTURES / "tiny_preds"),
            output=str(output_path),
            iou_thresh=0.5,
            label="run-a",
        )
    )
    capsys.readouterr()  # discard run_command's own print

    compare_command(argparse.Namespace(result_a=str(output_path), result_b=str(output_path)))
    out = capsys.readouterr().out

    assert "run-a" in out
    for line in out.splitlines()[2:]:  # skip header + separator
        if not line.strip():
            continue
        assert "n/a" in line or line.rstrip().endswith("+0.0000")
