from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from aib.narrative.cli.main import main
from conftest import FakeLLM


def test_main_help() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0


def test_datasets_list() -> None:
    assert main(["datasets", "list"]) == 0


def test_classify_help() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["classify", "--help"])
    assert exc_info.value.code == 0


def test_evaluate_help() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["evaluate", "--help"])
    assert exc_info.value.code == 0


def test_optimize_help() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["optimize", "--help"])
    assert exc_info.value.code == 0


def test_run_help() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["run", "--help"])
    assert exc_info.value.code == 0


def test_classify_with_mocked_model(capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeLLM('{"labels": ["resolution"], "scores": {"resolution": 0.9}, "evidence": []}')
    with patch("aib.narrative.cli.commands.classify.load_model", return_value=fake):
        code = main(["classify", "The hero returned home."])
    assert code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["labels"] == ["resolution"]


def test_evaluate_with_mocked_model(tmp_path: Path) -> None:
    fake = FakeLLM('{"labels": ["resolution"], "scores": {"resolution": 0.9}, "evidence": []}')
    run_dir = tmp_path / "evaluate"
    with patch("aib.narrative.cli.commands.evaluate.load_model", return_value=fake):
        code = main(
            [
                "evaluate",
                "--dataset",
                "freytag-sample",
                "--split",
                "val",
                "--run-dir",
                str(run_dir),
            ]
        )
    assert code == 0
