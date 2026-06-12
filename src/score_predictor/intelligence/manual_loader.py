from __future__ import annotations

from pathlib import Path

import yaml

from .schemas import IntelligenceInput


def load_intelligence(path: str | Path) -> IntelligenceInput:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("Intelligence YAML must contain a mapping/object.")
    if "intelligence" in data and isinstance(data["intelligence"], dict):
        data = data["intelligence"]
    return IntelligenceInput(**data)

