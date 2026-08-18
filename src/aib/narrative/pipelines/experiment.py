"""Shared experiment pipeline logic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..artifacts import LocalArtifactStore
from ..data.registry import RegisteredDataset, load_dataset
from ..evaluation import DiagnosticF1Evaluator
from ..experiments import Experiment, ExperimentConfig, ExperimentRunner
from ..models import Dataset, GenerationOptions, LLMProtocol, NarrativeExample
from ..optimization import GEPAOptimizer, OptimizationConfig, OptimizationResult
from ..tasks import RhetoricalCategoryTask


@dataclass(frozen=True, slots=True)
class EvaluateResult:
    score: float
    run_id: str
    run_path: Path


@dataclass(frozen=True, slots=True)
class ExperimentPipelineResult:
    dataset_name: str
    baseline_f1: float | None
    optimized_f1: float
    best_prompt: str
    baseline_run_id: str | None
    optimized_run_id: str
    optimize_run_dir: Path


def _read_prompt(prompt_file: str | None) -> str | None:
    if prompt_file is None:
        return None
    return Path(prompt_file).read_text(encoding="utf-8").strip()


def _build_task(
    dataset: RegisteredDataset,
    prompt_file: str | None,
    temperature: float,
) -> RhetoricalCategoryTask:
    prompt = _read_prompt(prompt_file)
    return RhetoricalCategoryTask(
        dataset.categories,
        prompt=prompt,
        options=GenerationOptions(temperature=temperature, max_tokens=256),
    )


def _examples_for_split(dataset: RegisteredDataset, split: str) -> tuple[NarrativeExample, ...]:
    if split == "train":
        return dataset.train
    if split == "val":
        return dataset.val
    raise ValueError(f"Unknown split {split!r}. Expected 'train' or 'val'.")


def evaluate_dataset(
    model: LLMProtocol,
    dataset_name: str,
    *,
    split: str = "val",
    prompt_file: str | None = None,
    run_dir: str = "runs/evaluate",
    experiment_name: str = "rhetorical-classification",
    seed: int = 42,
    model_id: str = "Qwen/Qwen2.5-3B-Instruct",
    temperature: float = 0.0,
) -> EvaluateResult:
    dataset = load_dataset(dataset_name)
    examples = _examples_for_split(dataset, split)
    task = _build_task(dataset, prompt_file, temperature)
    experiment = Experiment(
        config=ExperimentConfig(
            name=experiment_name,
            seed=seed,
            model={"provider": "huggingface", "model_id": model_id},
        ),
        task=task,
        dataset=Dataset.from_examples(examples, name=f"{dataset_name}-{split}"),
        evaluator=DiagnosticF1Evaluator(),
    )
    result = ExperimentRunner(artifact_store=LocalArtifactStore(run_dir)).run(experiment, model)
    if result.status != "completed" or result.score is None:
        raise RuntimeError(result.error or "Evaluation failed.")
    return EvaluateResult(
        score=result.score,
        run_id=result.run_id,
        run_path=Path(result.manifest.path),
    )


def optimize_prompt(
    model: LLMProtocol,
    reflection_model: LLMProtocol,
    dataset_name: str,
    *,
    prompt_file: str | None = None,
    run_dir: str = "runs/optimize",
    max_metric_calls: int = 25,
    reflection_minibatch_size: int = 2,
    seed: int = 42,
    temperature: float = 0.0,
) -> OptimizationResult:
    dataset = load_dataset(dataset_name)
    task = _build_task(dataset, prompt_file, temperature)
    optimizer = GEPAOptimizer(
        reflection_model=reflection_model,
        config=OptimizationConfig(
            max_metric_calls=max_metric_calls,
            reflection_minibatch_size=reflection_minibatch_size,
            seed=seed,
            run_dir=run_dir,
        ),
    )
    result = optimizer.optimize(
        task=task,
        model=model,
        trainset=list(dataset.train),
        valset=list(dataset.val),
        evaluator=DiagnosticF1Evaluator(),
    )
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    Path(run_dir, "best_prompt.txt").write_text(result.best_prompt, encoding="utf-8")
    return result


def run_full_experiment(
    model: LLMProtocol,
    reflection_model: LLMProtocol,
    dataset_name: str,
    *,
    prompt_file: str | None = None,
    run_dir: str = "runs/experiment",
    max_metric_calls: int = 25,
    reflection_minibatch_size: int = 2,
    seed: int = 42,
    model_id: str = "Qwen/Qwen2.5-3B-Instruct",
    temperature: float = 0.0,
    skip_baseline: bool = False,
) -> ExperimentPipelineResult:
    base_dir = Path(run_dir)
    baseline_f1: float | None = None
    baseline_run_id: str | None = None

    if not skip_baseline:
        baseline = evaluate_dataset(
            model,
            dataset_name,
            split="val",
            run_dir=str(base_dir / "baseline"),
            experiment_name="baseline",
            seed=seed,
            model_id=model_id,
            temperature=temperature,
        )
        baseline_f1 = baseline.score
        baseline_run_id = baseline.run_id

    optimize_dir = base_dir / "optimize"
    optimization = optimize_prompt(
        model,
        reflection_model,
        dataset_name,
        prompt_file=prompt_file,
        run_dir=str(optimize_dir),
        max_metric_calls=max_metric_calls,
        reflection_minibatch_size=reflection_minibatch_size,
        seed=seed,
        temperature=temperature,
    )
    best_prompt_path = optimize_dir / "best_prompt.txt"

    optimized = evaluate_dataset(
        model,
        dataset_name,
        split="val",
        prompt_file=str(best_prompt_path),
        run_dir=str(base_dir / "optimized"),
        experiment_name="optimized",
        seed=seed,
        model_id=model_id,
        temperature=temperature,
    )

    return ExperimentPipelineResult(
        dataset_name=dataset_name,
        baseline_f1=baseline_f1,
        optimized_f1=optimized.score,
        best_prompt=optimization.best_prompt,
        baseline_run_id=baseline_run_id,
        optimized_run_id=optimized.run_id,
        optimize_run_dir=optimize_dir,
    )
