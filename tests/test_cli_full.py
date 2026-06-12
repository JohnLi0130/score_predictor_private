from __future__ import annotations

import json
from pathlib import Path

from score_predictor.cli import main


def test_full_example_yaml_runs(capsys) -> None:
    project_root = Path(__file__).resolve().parents[1]
    example = project_root / "examples" / "match_full_example.yaml"

    exit_code = main(["predict", str(example), "--json-only"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert "final_lambda" in payload
    assert len(payload["top_scores"]) == 5
    assert "data_quality" in payload
    assert "warnings" in payload
    assert "audit" in payload
    assert "not betting advice" in payload["disclaimer"]


def test_separate_intel_file_runs(capsys) -> None:
    project_root = Path(__file__).resolve().parents[1]
    match_input = project_root / "examples" / "match_input_example.yaml"
    intel = project_root / "examples" / "match_intel_example.yaml"

    exit_code = main(["predict", str(match_input), "--intel", str(intel), "--json-only"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert "intelligence" in payload
    assert "friendly_match_total_goals_discount" in payload["warnings"]
