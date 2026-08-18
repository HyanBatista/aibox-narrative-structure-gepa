"""Built-in datasets for rhetorical classification."""

from .freytag_sample import FREYTAG_CATEGORIES, TRAIN_EXAMPLES, VAL_EXAMPLES
from .registry import RegisteredDataset, get_categories, list_datasets, load_dataset

__all__ = [
    "FREYTAG_CATEGORIES",
    "RegisteredDataset",
    "TRAIN_EXAMPLES",
    "VAL_EXAMPLES",
    "get_categories",
    "list_datasets",
    "load_dataset",
]
