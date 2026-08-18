from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from aib.narrative.pipelines.experiment import run_full_experiment
from conftest import FakeLLM


def test_run_full_experiment_writes_summary(tmp_path: Path) -> None:
    response = '{"labels": ["resolution"], "scores": {"resolution": 0.9}, "evidence": ["resolved"]}'
    fake = FakeLLM(response)
    with patch("aib.narrative.pipelines.experiment.GEPAOptimizer") as optimizer_cls:
        from aib.narrative.optimization import OptimizationResult

        optimizer_cls.return_value.optimize.return_value = OptimizationResult(
            best_prompt="optimized prompt",
            best_score=0.9,
        )
        result = run_full_experiment(
            fake,
            fake,
            "freytag-sample",
            run_dir=str(tmp_path),
            max_metric_calls=3,
            skip_baseline=True,
        )

    summary = json.loads((tmp_path / "experiment_summary.json").read_text(encoding="utf-8"))
    assert summary["seed_prompt"]
    assert summary["optimization"]["best_prompt"] == "optimized prompt"
    assert result.best_prompt == "optimized prompt"
    assert result.summary_path == tmp_path / "experiment_summary.json"
