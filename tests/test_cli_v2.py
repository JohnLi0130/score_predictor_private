from __future__ import annotations

import json
from pathlib import Path

from score_predictor.cli import main


def test_market_command_runs(capsys) -> None:
    project_root = Path(__file__).resolve().parents[1]
    example = project_root / "examples" / "match_v2_sporttery_manual.yaml"

    exit_code = main(["market", str(example), "--json-only"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert "market_features" in payload
    assert "movement" in payload


def test_research_command_runs(capsys) -> None:
    project_root = Path(__file__).resolve().parents[1]
    example = project_root / "examples" / "match_research_config.yaml"

    exit_code = main(["research", str(example), "--json-only"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert "facts" in payload
    assert "market" in payload
    assert "audit" in payload


def test_predict_command_still_runs_with_research(capsys) -> None:
    project_root = Path(__file__).resolve().parents[1]
    match_input = project_root / "examples" / "match_full_example.yaml"
    research = project_root / "examples" / "match_research_config.yaml"

    exit_code = main(
        ["predict", str(match_input), "--research", str(research), "--json-only"]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert "final_lambda" in payload
    assert "research_bundle" in payload

