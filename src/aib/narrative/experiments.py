"""Experiment execution and reproducibility metadata."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from .artifacts import ArtifactManifest, ArtifactStore, LocalArtifactStore, manifest_for
from .evaluation import EvaluationResult, EvaluatorProtocol
from .models import Dataset, NarrativeExample, Prediction
from .tasks import TaskProtocol


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    name: str
    seed: int = 0
    model: Mapping[str, Any] = field(default_factory=lambda: {})
    settings: Mapping[str, Any] = field(default_factory=lambda: {})


@dataclass(frozen=True, slots=True)
class Experiment:
    config: ExperimentConfig
    task: TaskProtocol
    dataset: Dataset
    evaluator: EvaluatorProtocol
    metadata: Mapping[str, Any] = field(default_factory=lambda: {})


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    run_id: str
    status: str
    score: float | None
    metrics: Mapping[str, float]
    manifest: ArtifactManifest
    error: str | None = None


class ExperimentRunner:
    """Execute a task and persist all run-level outputs."""

    def __init__(self, artifact_store: ArtifactStore | None = None) -> None:
        self.artifact_store = artifact_store or LocalArtifactStore()

    def run(self, experiment: Experiment, model: object) -> ExperimentResult:
        run_id = self._run_id(experiment)
        run_path = self.artifact_store.create_run(experiment.config.name, run_id)
        self.artifact_store.write_json(run_path, "config.json", self._config_payload(experiment))
        self.artifact_store.write_json(
            run_path,
            "environment.json",
            {"python": sys.version, "platform": platform.platform()},
        )
        try:
            experiment.task.fit(model, experiment.dataset.examples)
            predictions = [
                experiment.task.run(model, example) for example in experiment.dataset.examples
            ]
            evaluation = experiment.evaluator.evaluate(experiment.dataset.examples, predictions)
            self.artifact_store.write_jsonl(
                run_path,
                "predictions.jsonl",
                [
                    self._prediction_payload(example, prediction)
                    for example, prediction in zip(
                        experiment.dataset.examples, predictions, strict=True
                    )
                ],
            )
            self.artifact_store.write_json(
                run_path, "metrics.json", self._evaluation_payload(evaluation)
            )
            manifest = manifest_for(run_path, run_id)
            return ExperimentResult(
                run_id, "completed", evaluation.score, evaluation.metrics, manifest
            )
        except Exception as exc:
            self.artifact_store.write_text(run_path, "error.txt", f"{type(exc).__name__}: {exc}\n")
            manifest = manifest_for(run_path, run_id)
            return ExperimentResult(run_id, "failed", None, {}, manifest, str(exc))

    def _run_id(self, experiment: Experiment) -> str:
        payload = {
            "name": experiment.config.name,
            "seed": experiment.config.seed,
            "model": experiment.config.model,
            "settings": experiment.config.settings,
            "dataset": experiment.dataset.fingerprint,
            "task": experiment.task.name,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()
        return digest[:16]

    def _config_payload(self, experiment: Experiment) -> dict[str, Any]:
        return {
            "config": asdict(experiment.config),
            "metadata": dict(experiment.metadata),
            "dataset": {
                "name": experiment.dataset.name,
                "fingerprint": experiment.dataset.fingerprint,
                "size": len(experiment.dataset.examples),
            },
            "task": experiment.task.name,
            "created_at": datetime.now(UTC).isoformat(),
        }

    def _prediction_payload(
        self, example: NarrativeExample, prediction: Prediction
    ) -> dict[str, Any]:
        return {
            "text": example.text,
            "expected": list(example.labels),
            "prediction": asdict(prediction),
        }

    def _evaluation_payload(self, evaluation: EvaluationResult) -> dict[str, Any]:
        return {
            "score": evaluation.score,
            "metrics": evaluation.metrics,
            "feedback": evaluation.feedback,
        }
