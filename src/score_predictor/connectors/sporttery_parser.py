from __future__ import annotations

from .sporttery_manual import normalize_sporttery_manual


def parse_sporttery_manual_payload(data: dict) -> dict:
    return normalize_sporttery_manual(data)

