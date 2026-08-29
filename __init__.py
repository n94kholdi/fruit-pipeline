"""Pretrained-only detection + segmentation pipeline for densely packed fruit images.

Pipeline stages: tiled detection (detect.py) -> merge (merge.py) ->
box-prompted segmentation (segment.py) -> visualization (visualize.py),
orchestrated by pipeline.py and exposed via cli.py.
"""

__version__ = "0.1.0"
