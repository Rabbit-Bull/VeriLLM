"""
VeriLLM Top-Level Factuality Scoring Orchestrator.
Provides unified interfaces for single-response scoring, dataset-level aggregation,
and cross-model hallucination benchmarking.
"""

from typing import List, Optional, Union, Dict

from ml_engine.claim_verifier.models import VerificationReport, VerifiedClaim
from ml_engine.scorer.models import (
    ResponseScore,
    DatasetScore,
    ScoringConfig
)
from ml_engine.scorer.response_scorer import ResponseScorer
from ml_engine.scorer.dataset_scorer import DatasetScorer


class FactualityScorer:
    """
    Main entry point for the VeriLLM Scoring and Quantification Engine.
    Coordinates response-level evaluations, micro/macro aggregations, and comparative benchmarks.
    """

    def __init__(self, config: Optional[ScoringConfig] = None):
        self.config = config or ScoringConfig()
        self.response_scorer = ResponseScorer(config=self.config)
        self.dataset_scorer = DatasetScorer(config=self.config)

    def score_response(
        self,
        report_or_claims: Union[VerificationReport, List[VerifiedClaim]],
        response_id: str = "resp_001",
        model_id: Optional[str] = None,
        document_name: Optional[str] = None,
        execution_time_seconds: Optional[float] = None
    ) -> ResponseScore:
        """
        Score a single response from its verified claims or VerificationReport.
        """
        return self.response_scorer.score_response(
            report_or_claims=report_or_claims,
            response_id=response_id,
            model_id=model_id,
            document_name=document_name,
            execution_time_seconds=execution_time_seconds
        )

    def score_dataset(
        self,
        responses_or_reports: List[Union[ResponseScore, VerificationReport]],
        dataset_name: Optional[str] = None
    ) -> DatasetScore:
        """
        Aggregate multiple response scores or reports into pooled dataset metrics.
        """
        return self.dataset_scorer.score_dataset(
            responses_or_reports=responses_or_reports,
            dataset_name=dataset_name
        )

    def compare_models(
        self,
        responses_by_model: Dict[str, List[Union[ResponseScore, VerificationReport]]],
        dataset_name: Optional[str] = "Multi-Model Comparison"
    ) -> DatasetScore:
        """
        Evaluate and compare multiple generative LLMs on identical or separate test cases.

        Args:
            responses_by_model: Dictionary mapping model identifiers (e.g. 'gpt-4o', 'claude-3-5-sonnet')
                                to their respective lists of ResponseScore or VerificationReport objects.
            dataset_name: Label for the comparison study.

        Returns:
            DatasetScore containing pooled metrics and model-by-model comparisons.
        """
        all_scores: List[ResponseScore] = []
        for model_id, items in responses_by_model.items():
            for i, item in enumerate(items):
                if isinstance(item, ResponseScore):
                    # Ensure model_id is set
                    item.model_id = model_id
                    all_scores.append(item)
                elif isinstance(item, VerificationReport):
                    resp_id = f"{model_id}_resp_{i+1:03d}"
                    score = self.response_scorer.score_response(
                        item,
                        response_id=resp_id,
                        model_id=model_id
                    )
                    all_scores.append(score)
                else:
                    raise ValueError(f"Unsupported item type for model comparison: {type(item)}")

        return self.dataset_scorer.score_dataset(all_scores, dataset_name=dataset_name)

