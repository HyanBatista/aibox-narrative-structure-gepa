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
from .logging import (
    log_metrics,
    log_prompt,
    log_run_location,
    log_section,
    write_experiment_summary,
)


@dataclass(frozen=True, slots=True)
class EvaluateResult:
    score: float
    metrics: dict[str, float]
    prompt: str
    run_id: str
    run_path: Path


@dataclass(frozen=True, slots=True)
class ExperimentPipelineResult:
    dataset_name: str
    seed_prompt: str
    baseline_f1: float | None
    baseline_metrics: dict[str, float] | None
    optimized_f1: float
    optimized_metrics: dict[str, float]
    optimization_best_score: float
    best_prompt: str
    baseline_run_id: str | None
    optimized_run_id: str
    optimize_run_dir: Path
    summary_path: Path


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
    evaluate_result = EvaluateResult(
        score=result.score,
        metrics=dict(result.metrics),
        prompt=task.prompt,
        run_id=result.run_id,
        run_path=Path(result.manifest.path),
    )
    log_section(f"Evaluation ({split})")
    log_prompt(task.prompt)
    log_metrics(evaluate_result.score, evaluate_result.metrics)
    log_run_location(evaluate_result.run_path)
    return evaluate_result


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
    reflection_temperature: float = 0.3,
) -> OptimizationResult:
    dataset = load_dataset(dataset_name)
    task = _build_task(dataset, prompt_file, temperature)
    log_section("GEPA optimization")
    log_prompt(task.prompt)
    optimizer = GEPAOptimizer(
        reflection_model=reflection_model,
        config=OptimizationConfig(
            max_metric_calls=max_metric_calls,
            reflection_minibatch_size=reflection_minibatch_size,
            seed=seed,
            run_dir=run_dir,
            reflection_temperature=reflection_temperature,
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
    Path(run_dir, "seed_prompt.txt").write_text(task.prompt, encoding="utf-8")
    log_metrics(result.best_score, {"f1": result.best_score})
    print(f"Best prompt saved to: {Path(run_dir) / 'best_prompt.txt'}")
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
    reflection_model_id: str | None = None,
    temperature: float = 0.0,
    reflection_temperature: float = 0.3,
    skip_baseline: bool = False,
) -> ExperimentPipelineResult:
    base_dir = Path(run_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(dataset_name)
    seed_task = _build_task(dataset, prompt_file, temperature)
    seed_prompt = seed_task.prompt

    print("=== Rhetorical Classification Experiment ===")
    print(f"Dataset: {dataset_name} ({len(dataset.train)} train / {len(dataset.val)} val)")
    print(f"Model:   {model_id}")
    if reflection_model_id and reflection_model_id != model_id:
        print(f"Reflect: {reflection_model_id}")
    log_section("Seed prompt")
    log_prompt(seed_prompt)

    baseline_f1: float | None = None
    baseline_metrics: dict[str, float] | None = None
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
        baseline_metrics = baseline.metrics
        baseline_run_id = baseline.run_id

    optimization = optimize_prompt(
        model,
        reflection_model,
        dataset_name,
        prompt_file=prompt_file,
        run_dir=str(base_dir / "optimize"),
        max_metric_calls=max_metric_calls,
        reflection_minibatch_size=reflection_minibatch_size,
        seed=seed,
        temperature=temperature,
        reflection_temperature=reflection_temperature,
    )
    best_prompt_path = base_dir / "optimize" / "best_prompt.txt"

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

    log_section("Evolved prompt")
    log_prompt(optimization.best_prompt)

    print("\n=== Experiment summary ===")
    if baseline_f1 is not None and baseline_metrics is not None:
        print("Baseline metrics:")
        log_metrics(baseline_f1, baseline_metrics)
    print("Optimized metrics:")
    log_metrics(optimized.score, optimized.metrics)
    if baseline_f1 is not None:
        print(f"Delta: {optimized.score - baseline_f1:+.4f}")

    summary_path = base_dir / "experiment_summary.json"
    write_experiment_summary(
        summary_path,
        {
            "dataset": dataset_name,
            "model_id": model_id,
            "reflection_model_id": reflection_model_id or model_id,
            "seed_prompt": seed_prompt,
            "baseline": (
                None
                if baseline_f1 is None
                else {
                    "f1": baseline_f1,
                    "metrics": baseline_metrics,
                    "run_id": baseline_run_id,
                }
            ),
            "optimization": {
                "best_f1": optimization.best_score,
                "best_prompt": optimization.best_prompt,
                "run_dir": str(base_dir / "optimize"),
            },
            "optimized": {
                "f1": optimized.score,
                "metrics": optimized.metrics,
                "run_id": optimized.run_id,
            },
        },
    )
    print(f"Summary: {summary_path}")

    return ExperimentPipelineResult(
        dataset_name=dataset_name,
        seed_prompt=seed_prompt,
        baseline_f1=baseline_f1,
        baseline_metrics=baseline_metrics,
        optimized_f1=optimized.score,
        optimized_metrics=optimized.metrics,
        optimization_best_score=optimization.best_score,
        best_prompt=optimization.best_prompt,
        baseline_run_id=baseline_run_id,
        optimized_run_id=optimized.run_id,
        optimize_run_dir=base_dir / "optimize",
        summary_path=summary_path,
    )
