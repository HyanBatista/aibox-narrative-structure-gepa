"""Dataset registry for rhetorical classification experiments."""

from __future__ import annotations

from dataclasses import dataclass

from ..models import Category, NarrativeExample
from .freytag_sample import FREYTAG_CATEGORIES, TRAIN_EXAMPLES, VAL_EXAMPLES


@dataclass(frozen=True, slots=True)
class RegisteredDataset:
    """A named dataset with train/val splits and category schema."""

    name: str
    description: str
    categories: tuple[Category, ...]
    train: tuple[NarrativeExample, ...]
    val: tuple[NarrativeExample, ...]


_DATASETS: dict[str, RegisteredDataset] = {
    "freytag-sample": RegisteredDataset(
        name="freytag-sample",
        description="Short narratives labeled with Freytag-inspired rhetorical categories.",
        categories=FREYTAG_CATEGORIES,
        train=TRAIN_EXAMPLES,
        val=VAL_EXAMPLES,
    ),
}


def list_datasets() -> tuple[RegisteredDataset, ...]:
    return tuple(_DATASETS[name] for name in sorted(_DATASETS))


def load_dataset(name: str) -> RegisteredDataset:
    try:
        return _DATASETS[name]
    except KeyError as exc:
        known = ", ".join(sorted(_DATASETS)) or "(none)"
        raise ValueError(f"Unknown dataset {name!r}. Known datasets: {known}.") from exc


def get_categories(name: str = "freytag-sample") -> tuple[Category, ...]:
    return load_dataset(name).categories
