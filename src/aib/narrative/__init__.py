"""Experiment primitives for narrative rhetorical-category detection."""

from .artifacts import ArtifactManifest, ArtifactStore, LocalArtifactStore
from .evaluation import (
    DiagnosticF1Evaluator,
    EvaluationResult,
    EvaluatorProtocol,
    F1Evaluator,
    multilabel_f1,
)
from .experiments import Experiment, ExperimentConfig, ExperimentResult, ExperimentRunner
from .llm import HuggingFaceLLM, detect_device
from .models import (
    Category,
    Dataset,
    EstimatorProtocol,
    GenerationOptions,
    LLMProtocol,
    ModelResponse,
    NarrativeExample,
    Prediction,
)
from .optimization import GEPAOptimizer, OptimizationConfig, OptimizationResult
from .tasks import RhetoricalCategoryTask, TaskProtocol

__all__ = [
    "ArtifactManifest",
    "ArtifactStore",
    "Category",
    "Dataset",
    "DiagnosticF1Evaluator",
    "EstimatorProtocol",
    "EvaluationResult",
    "EvaluatorProtocol",
    "F1Evaluator",
    "Experiment",
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentRunner",
    "GenerationOptions",
    "HuggingFaceLLM",
    "LLMProtocol",
    "LocalArtifactStore",
    "ModelResponse",
    "NarrativeExample",
    "Prediction",
    "RhetoricalCategoryTask",
    "TaskProtocol",
    "GEPAOptimizer",
    "OptimizationConfig",
    "OptimizationResult",
    "detect_device",
    "multilabel_f1",
]
