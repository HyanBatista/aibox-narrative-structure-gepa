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

To run experiments with a local Hugging Face model:

```bash
uv sync --group dev --group huggingface
```

## CLI

Install the package, then use `aib-narrative`:

```bash
uv run aib-narrative datasets list
```

| Command | Purpose |
|---|---|
| `classify` | Assign rhetorical categories to narrative text |
| `evaluate` | Benchmark a prompt on a labeled dataset |
| `optimize` | Evolve the classification prompt with GEPA |
| `run` | Full pipeline: baseline → GEPA → re-evaluate |
| `datasets list` | Show registered datasets |

### Full GEPA experiment

```bash
# Pre-download model (cached under ~/.cache/huggingface)
uv run python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct')"

# One-command pipeline
uv run aib-narrative run --dataset freytag-sample --run-dir runs/experiment

# Or step-by-step
uv run aib-narrative evaluate --dataset freytag-sample --split val --run-dir runs/baseline
uv run aib-narrative optimize --dataset freytag-sample --run-dir runs/optimize
uv run aib-narrative evaluate --dataset freytag-sample --split val \
  --prompt-file runs/optimize/best_prompt.txt --run-dir runs/optimized

# Classify a single passage
uv run aib-narrative classify "The wolves retreated and peace returned to the village."
```

Default model: `Qwen/Qwen2.5-3B-Instruct` (auto-detects `cuda`, `mps`, or `cpu`).

## Programmatic usage

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
