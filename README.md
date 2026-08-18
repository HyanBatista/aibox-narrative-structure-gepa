# AIB Narrative Structure GEPA

An experiment-oriented Python library for rhetorical-category detection and GEPA prompt optimization.
The public namespace is `aib.narrative.*`.

## Development

This project uses Astral's tools:

```bash
uv sync --group dev
uv run ruff format .
uv run ruff check .
uv run pyright
uv run pytest
```

## A tracked experiment

```python
from aib.narrative import Category, Dataset, Experiment, ExperimentConfig
from aib.narrative import ExperimentRunner, F1Evaluator, NarrativeExample
from aib.narrative.tasks import RhetoricalCategoryTask

dataset = Dataset.from_examples(
    [NarrativeExample("The hero returns home.", labels=("resolution",))],
    name="demo",
)
task = RhetoricalCategoryTask(
    [Category("resolution", "The narrative resolves its central conflict.")]
)

experiment = Experiment(
    config=ExperimentConfig(name="demo", model={"provider": "example"}),
    task=task,
    dataset=dataset,
    evaluator=F1Evaluator(),
)
result = ExperimentRunner().run(experiment, model=my_llm)
print(result.status, result.score, result.manifest.path)
```

Models are intentionally protocol-based. An LLM implements `generate(messages, options)`,
while an estimator implements `fit(features, targets)` and `predict(features)`.

