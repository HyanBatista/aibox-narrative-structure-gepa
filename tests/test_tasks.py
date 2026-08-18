import pytest

from aib.narrative.models import Category
from aib.narrative.tasks import RhetoricalCategoryTask


def test_rhetorical_task_parses_prediction(fake_llm: object) -> None:
    task = RhetoricalCategoryTask([Category("resolution")])
    prediction = task.predict(fake_llm, "A story ends.")  # type: ignore[arg-type]
    assert prediction.labels == ("resolution",)
    assert prediction.scores["resolution"] == 0.9


def test_rhetorical_task_rejects_unknown_label() -> None:
    task = RhetoricalCategoryTask([Category("resolution")])
    with pytest.raises(ValueError, match="unknown"):
        task.parse_prediction('{"labels": ["setup"]}')


def test_rhetorical_task_accepts_markdown_json() -> None:
    task = RhetoricalCategoryTask([Category("resolution")])
    prediction = task.parse_prediction('```json\n{"labels": ["resolution"]}\n```')
    assert prediction.labels == ("resolution",)
