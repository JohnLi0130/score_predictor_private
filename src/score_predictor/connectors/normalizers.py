from __future__ import annotations

from datetime import datetime
from typing import Any


def to_float_mapping(data: dict[str, Any], *, skip_keys: set[str] | None = None) -> dict:
    skip = skip_keys or set()
    result: dict[str, float] = {}
    for key, value in data.items():
        if key in skip:
            continue
        if value is None:
            continue
        result[str(key)] = float(value)
    return result


def parse_datetime(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    try:
        from dateutil import parser

        return parser.parse(normalized)
    except ModuleNotFoundError:
        return datetime.fromisoformat(normalized)

