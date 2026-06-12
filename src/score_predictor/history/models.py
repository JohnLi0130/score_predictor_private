from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PredictionHistoryRecord:
    prediction_context_key: str
    prediction_id: str
    home_team: str
    away_team: str
    match_id: str | None = None
    event_id: str | None = None
    competition: str | None = None
    stage: str | None = None
    kickoff_time: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
