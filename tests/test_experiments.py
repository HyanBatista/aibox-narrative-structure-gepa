from pathlib import Path

from aib.narrative.artifacts import LocalArtifactStore
from aib.narrative.evaluation import F1Evaluator
from aib.narrative.experiments import Experiment, ExperimentConfig, ExperimentRunner
from aib.narrative.models import Category, Dataset, NarrativeExample
from aib.narrative.tasks import RhetoricalCategoryTask


def test_runner_writes_local_artifacts(tmp_path: Path, fake_llm: object) -> None:
    dataset = Dataset.from_examples([NarrativeExample("A story ends.", ("resolution",))])
    experiment = Experiment(
        config=ExperimentConfig(name="smoke"),
        task=RhetoricalCategoryTask([Category("resolution")]),
        dataset=dataset,
        evaluator=F1Evaluator(),
    )
    result = ExperimentRunner(LocalArtifactStore(tmp_path)).run(experiment, fake_llm)
    assert result.status == "completed"
    assert result.score == 1.0
    assert (tmp_path / "smoke" / result.run_id / "predictions.jsonl").exists()


def test_runner_preserves_failure(tmp_path: Path) -> None:
    class BrokenTask:
        name = "broken"

        def fit(self, model: object, examples: object) -> None:
            raise RuntimeError("broken")

        def run(self, model: object, example: object) -> object:
            raise AssertionError("not reached")

    dataset = Dataset.from_examples([NarrativeExample("text")])
    experiment = Experiment(
        config=ExperimentConfig(name="failure"),
        task=BrokenTask(),  # type: ignore[arg-type]
        dataset=dataset,
        evaluator=F1Evaluator(),
    )
    result = ExperimentRunner(LocalArtifactStore(tmp_path)).run(experiment, object())
    assert result.status == "failed"
    assert (tmp_path / "failure" / result.run_id / "error.txt").exists()
