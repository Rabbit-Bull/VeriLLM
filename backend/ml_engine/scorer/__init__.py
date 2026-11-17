"""
VeriLLM Scoring and Quantification Engine.
Converts proposition-level verification results into mathematically rigorous,
transparent response-level factuality scores and benchmark dataset aggregations.
"""

from ml_engine.scorer.models import (
    ScoringConfig,
    ClaimSummary,
    ResponseCounts,
    ResponseRates,
    FactualityIndices,
    ResponseScore,
    ModelAggregateScore,
    DatasetScore
)
from ml_engine.scorer.response_scorer import ResponseScorer
from ml_engine.scorer.dataset_scorer import DatasetScorer
from ml_engine.scorer.scorer import FactualityScorer

__all__ = [
    "ScoringConfig",
    "ClaimSummary",
    "ResponseCounts",
    "ResponseRates",
    "FactualityIndices",
    "ResponseScore",
    "ModelAggregateScore",
    "DatasetScore",
    "ResponseScorer",
    "DatasetScorer",
    "FactualityScorer"
]

