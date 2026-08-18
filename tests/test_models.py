import pytest

from aib.narrative.models import Category, Dataset, NarrativeExample, Prediction


def test_dataset_fingerprint_is_deterministic() -> None:
    examples = (NarrativeExample("text", ("setup",)),)
    assert Dataset(examples).fingerprint == Dataset(examples).fingerprint


def test_invalid_category_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        Category(" ")


def test_prediction_rejects_out_of_range_scores() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        Prediction(("setup",), {"setup": 1.1})
