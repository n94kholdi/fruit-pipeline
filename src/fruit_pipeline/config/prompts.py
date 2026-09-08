"""Attribute-rich, configurable open-vocab prompt sets (YOLO-World / YOLOE text mode).

SDM-D's (arXiv 2411.16196) error analysis found most open-vocab errors come
from the CLIP text-matching stage, and that performance depends more on
descriptive visual attributes ("a round orange citrus fruit") than the bare
class name ("orange") — and that giving the model an explicit background/
other class to absorb non-fruit regions reduces false positives on those
regions. This module loads that prompt set from a YAML file instead of a
hardcoded list, and keeps the fruit/background split visible so callers can
drop background-category detections before they ever reach merging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

_BACKGROUND_NAMES = {"background", "other", "others", "none"}


@dataclass
class PromptConfig:
    fruit_prompts: list[str] = field(default_factory=list)
    background_prompts: list[str] = field(default_factory=list)

    @property
    def all_prompts(self) -> list[str]:
        """Fruit prompts first, background prompts last -- callers rely on this order

        (e.g. ``num_fruit_classes = len(fruit_prompts)``, so class indices
        ``< num_fruit_classes`` are fruit and everything after is
        background) to filter out background-category detections.
        """
        return [*self.fruit_prompts, *self.background_prompts]

    @property
    def num_fruit_classes(self) -> int:
        return len(self.fruit_prompts)


def load_prompt_config(path: str) -> PromptConfig:
    """Load a prompt YAML shaped like::

        classes:
          - name: fruit
            prompts: ["a round orange citrus fruit", "a ripe red apple"]
          - name: background
            prompts: ["background", "wooden crate wall"]

    Any class entry named background/other/others/none (case-insensitive) is
    treated as the background/others class; everything else is pooled into
    ``fruit_prompts``. A config needs at least one fruit prompt; the
    background class is optional.
    """
    with open(path) as f:
        data = yaml.safe_load(f) or {}

    fruit_prompts: list[str] = []
    background_prompts: list[str] = []
    for entry in data.get("classes", []):
        name = str(entry.get("name", "")).strip().lower()
        entry_prompts = [str(p) for p in entry.get("prompts", [])]
        if name in _BACKGROUND_NAMES:
            background_prompts.extend(entry_prompts)
        else:
            fruit_prompts.extend(entry_prompts)

    if not fruit_prompts:
        raise ValueError(f"Prompt config {path!r} has no non-background prompts under 'classes'")

    return PromptConfig(fruit_prompts=fruit_prompts, background_prompts=background_prompts)


DEFAULT_PROMPT_CONFIG_PATH = str(Path(__file__).parent / "prompts" / "default.yaml")
