from __future__ import annotations

from .intelligence.schemas import IntelligenceInput


def build_audit(intel: IntelligenceInput | None) -> dict:
    if intel is None:
        return {
            "source_mode": "manual_only",
            "used_prediction_sources": False,
            "source_policy": "facts_only_no_prediction_articles",
            "sources": [],
            "excluded_sources": [],
        }

    return {
        "source_mode": intel.source_mode,
        "used_prediction_sources": False,
        "source_policy": "facts_only_no_prediction_articles",
        "sources": intel.sources,
        "excluded_sources": intel.excluded_sources,
    }
