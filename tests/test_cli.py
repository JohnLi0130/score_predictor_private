from __future__ import annotations

import json
from pathlib import Path

from score_predictor.cli import main


def test_cli_reads_example_yaml_and_outputs_top_five(capsys) -> None:
    project_root = Path(__file__).resolve().parents[1]
    example = project_root / "examples" / "match_input_example.yaml"

    exit_code = main(["predict", str(example), "--json-only"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["match"] == "Argentina vs Iceland"
    assert len(payload["top_scores"]) == 5
    assert payload["max_probability_score"]["score"]
