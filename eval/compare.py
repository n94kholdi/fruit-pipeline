"""``--compare a.json b.json``: a side-by-side delta table between two ``eval run`` outputs."""

from __future__ import annotations

from fruit_pipeline.eval.metrics import BUCKET_ORDER

METRIC_COLUMNS = ["mAP50", "mAP50-95", "precision", "recall", "mae", "rmse"]


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def _fmt_delta(a: float | int | None, b: float | int | None) -> str:
    if a is None or b is None:
        return "n/a"
    delta = b - a
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.4f}"


def build_delta_table(result_a: dict, result_b: dict, label_a: str = "A", label_b: str = "B") -> str:
    """Render a fixed-width text table comparing two ``eval run`` result dicts, bucket by bucket."""
    metrics_a = result_a["metrics"]
    metrics_b = result_b["metrics"]

    header = f"{'bucket':<8} {'metric':<10} {label_a:>10} {label_b:>10} {'delta':>10}"
    lines = [header, "-" * len(header)]

    for bucket in BUCKET_ORDER:
        bucket_a = metrics_a.get(bucket, {})
        bucket_b = metrics_b.get(bucket, {})
        for metric in METRIC_COLUMNS:
            val_a = bucket_a.get(metric)
            val_b = bucket_b.get(metric)
            lines.append(
                f"{bucket:<8} {metric:<10} {_fmt(val_a):>10} {_fmt(val_b):>10} {_fmt_delta(val_a, val_b):>10}"
            )

    return "\n".join(lines)
