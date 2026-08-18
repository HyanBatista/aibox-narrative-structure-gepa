from aib.narrative.evaluation import F1Evaluator, multilabel_f1
from aib.narrative.models import NarrativeExample, Prediction


def test_multilabel_f1() -> None:
    assert multilabel_f1(("a", "b"), ("b", "c")) == 0.5


def test_f1_evaluator() -> None:
    result = F1Evaluator().evaluate([NarrativeExample("text", ("a",))], [Prediction(("a",))])
    assert result.score == 1.0
