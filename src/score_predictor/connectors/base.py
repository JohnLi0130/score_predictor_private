from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def merge_warnings(*groups: list[str] | None) -> list[str]:
    warnings: list[str] = []
    for group in groups:
        if not group:
            continue
        warnings.extend(item for item in group if item)
    return list(dict.fromkeys(warnings))


def ensure_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping/object.")
    return value

