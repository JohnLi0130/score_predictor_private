from __future__ import annotations

from datetime import datetime, timezone
import json
import re
from pathlib import Path
from typing import Any


RAW_CACHE_DIR = Path("data/raw/the_odds_api")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def safe_filename_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return cleaned.strip("._") or "the_odds_api"


def write_raw_response(
    *,
    safe_match_id: str,
    kind: str,
    raw_json: Any,
    cache_dir: Path | str = RAW_CACHE_DIR,
    timestamp: str | None = None,
) -> Path:
    resolved_dir = Path(cache_dir)
    resolved_dir.mkdir(parents=True, exist_ok=True)
    stamp = timestamp or utc_timestamp()
    filename = (
        f"{safe_filename_part(safe_match_id)}_"
        f"{safe_filename_part(kind)}_"
        f"{safe_filename_part(stamp)}.json"
    )
    path = resolved_dir / filename
    path.write_text(
        json.dumps(raw_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path
