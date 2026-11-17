"""
VeriLLM Dataset-Level Scoring and Aggregation Engine.
Aggregates multiple response scores across benchmarks, datasets, and generative models.
Implements micro-pooled rates (avoiding Simpson's paradox) alongside macro-averages
and cross-model comparisons.
"""

from typing import List, Optional, Dict, Union
from collections import defaultdict

from ml_engine.claim_verifier.models import VerificationReport
from ml_engine.scorer.models import (
    ResponseScore,
    ModelAggregateScore,
    DatasetScore,
    ScoringConfig
)
from ml_engine.scorer.response_scorer import ResponseScorer


class DatasetScorer:
    """
    Aggregates evaluated responses across an entire benchmark dataset or multi-model evaluation.
    """

    def __init__(self, config: Optional[ScoringConfig] = None):
        self.config = config or ScoringConfig()
        self.response_scorer = ResponseScorer(config=self.config)

    def score_dataset(
        self,
        responses_or_reports: List[Union[ResponseScore, VerificationReport]],
        dataset_name: Optional[str] = None
    ) -> DatasetScore:
        """
        Aggregate a collection of response evaluations into a unified DatasetScore.

        Args:
            responses_or_reports: List of ResponseScore or VerificationReport objects.
            dataset_name: Optional label for the benchmark or dataset.

        Returns:
            DatasetScore with micro-pooled rates, macro-averages, and per-model breakdowns.
        """
        # Convert any VerificationReport objects to ResponseScore
        response_scores: List[ResponseScore] = []
        for i, item in enumerate(responses_or_reports):
            if isinstance(item, ResponseScore):
                response_scores.append(item)
            elif isinstance(item, VerificationReport):
                resp_id = f"resp_{i+1:03d}"
                response_scores.append(self.response_scorer.score_response(item, response_id=resp_id))
            else:
                raise ValueError(f"Unsupported item type in dataset evaluation: {type(item)}")

        total_responses = len(response_scores)
        empty_responses = sum(1 for r in response_scores if r.is_empty_response)
        valid_responses = [r for r in response_scores if r.has_factual_claims]
        zero_factual_responses = total_responses - len(valid_responses)

        # -------------------------------------------------------------
        # Step 1: Micro-Pooling (Sum counts before dividing)
        # Prevents Simpson's Paradox / weighted average fallacy
        # -------------------------------------------------------------
        total_extracted_claims = sum(r.counts.total_claims for r in response_scores)
        total_factual_claims = sum(r.counts.factual_claims for r in response_scores)
        total_non_factual_claims = sum(r.counts.non_factual_claims for r in response_scores)

        pooled_supported = sum(r.counts.supported_count for r in response_scores)
        pooled_contradicted = sum(r.counts.contradicted_count for r in response_scores)
        pooled_unsupported = sum(r.counts.unsupported_count for r in response_scores)
        pooled_unverifiable = sum(r.counts.unverifiable_count for r in response_scores)
        pooled_non_factual = sum(r.counts.non_factual_count for r in response_scores)

        if total_factual_claims > 0:
            N = total_factual_claims
            micro_esr = round((pooled_supported / N) * 100.0, 2)
            micro_ccr = round((pooled_contradicted / N) * 100.0, 2)
            micro_usr = round((pooled_unsupported / N) * 100.0, 2)
            micro_uvr = round((pooled_unverifiable / N) * 100.0, 2)
            micro_ur = round(((pooled_unsupported + pooled_unverifiable) / N) * 100.0, 2)
            micro_gfi = micro_esr

            lambda_w = self.config.contradiction_penalty_weight
            penalized_val = max(0.0, (pooled_supported - (lambda_w * pooled_contradicted)) / N) * 100.0
            micro_cpfs = round(penalized_val, 2)

            invariants_verified = abs(micro_esr + micro_ccr + micro_usr + micro_uvr - 100.0) <= 0.05
        else:
            micro_esr = None
            micro_ccr = None
            micro_usr = None
            micro_uvr = None
            micro_ur = None
            micro_gfi = None
            micro_cpfs = None
            invariants_verified = True

        # -------------------------------------------------------------
        # Step 2: Macro-Averaging (Unweighted mean of valid response rates)
        # -------------------------------------------------------------
        num_valid = len(valid_responses)
        if num_valid > 0:
            macro_esr = round(sum(r.rates.evidence_support_rate for r in valid_responses) / num_valid, 2)
            macro_ccr = round(sum(r.rates.confirmed_contradiction_rate for r in valid_responses) / num_valid, 2)
            macro_usr = round(sum(r.rates.unsupported_rate for r in valid_responses) / num_valid, 2)
            macro_uvr = round(sum(r.rates.unverifiable_rate for r in valid_responses) / num_valid, 2)
            macro_ur = round(sum(r.rates.combined_uncertainty_rate for r in valid_responses) / num_valid, 2)
        else:
            macro_esr = None
            macro_ccr = None
            macro_usr = None
            macro_uvr = None
            macro_ur = None

        # -------------------------------------------------------------
        # Step 3: Model-by-Model Breakdown
        # -------------------------------------------------------------
        responses_by_model: Dict[str, List[ResponseScore]] = defaultdict(list)
        for r in response_scores:
            if r.model_id:
                responses_by_model[r.model_id].append(r)

        models_breakdown: Dict[str, ModelAggregateScore] = {}
        for m_id, m_responses in responses_by_model.items():
            m_total_resp = len(m_responses)
            m_valid_resp = [r for r in m_responses if r.has_factual_claims]
            m_num_valid = len(m_valid_resp)

            m_factual_n = sum(r.counts.factual_claims for r in m_responses)
            m_non_factual_n = sum(r.counts.non_factual_claims for r in m_responses)
            m_supp = sum(r.counts.supported_count for r in m_responses)
            m_contra = sum(r.counts.contradicted_count for r in m_responses)
            m_unsup = sum(r.counts.unsupported_count for r in m_responses)
            m_unver = sum(r.counts.unverifiable_count for r in m_responses)
            m_non_fact_c = sum(r.counts.non_factual_count for r in m_responses)

            if m_factual_n > 0:
                m_micro_esr = round((m_supp / m_factual_n) * 100.0, 2)
                m_micro_ccr = round((m_contra / m_factual_n) * 100.0, 2)
                m_micro_usr = round((m_unsup / m_factual_n) * 100.0, 2)
                m_micro_uvr = round((m_unver / m_factual_n) * 100.0, 2)
                m_micro_ur = round(((m_unsup + m_unver) / m_factual_n) * 100.0, 2)
                m_micro_gfi = m_micro_esr
                lambda_w = self.config.contradiction_penalty_weight
                m_micro_cpfs = round(max(0.0, (m_supp - lambda_w * m_contra) / m_factual_n) * 100.0, 2)
            else:
                m_micro_esr = None
                m_micro_ccr = None
                m_micro_usr = None
                m_micro_uvr = None
                m_micro_ur = None
                m_micro_gfi = None
                m_micro_cpfs = None

            if m_num_valid > 0:
                m_macro_esr = round(sum(r.rates.evidence_support_rate for r in m_valid_resp) / m_num_valid, 2)
                m_macro_ccr = round(sum(r.rates.confirmed_contradiction_rate for r in m_valid_resp) / m_num_valid, 2)
                m_macro_usr = round(sum(r.rates.unsupported_rate for r in m_valid_resp) / m_num_valid, 2)
                m_macro_uvr = round(sum(r.rates.unverifiable_rate for r in m_valid_resp) / m_num_valid, 2)
                m_macro_ur = round(sum(r.rates.combined_uncertainty_rate for r in m_valid_resp) / m_num_valid, 2)
            else:
                m_macro_esr = None
                m_macro_ccr = None
                m_macro_usr = None
                m_macro_uvr = None
                m_macro_ur = None

            models_breakdown[m_id] = ModelAggregateScore(
                model_id=m_id,
                total_responses=m_total_resp,
                responses_with_factual_claims=m_num_valid,
                total_factual_claims=m_factual_n,
                total_non_factual_claims=m_non_factual_n,
                supported_count=m_supp,
                contradicted_count=m_contra,
                unsupported_count=m_unsup,
                unverifiable_count=m_unver,
                non_factual_count=m_non_fact_c,
                micro_evidence_support_rate=m_micro_esr,
                micro_confirmed_contradiction_rate=m_micro_ccr,
                micro_unsupported_rate=m_micro_usr,
                micro_unverifiable_rate=m_micro_uvr,
                micro_combined_uncertainty_rate=m_micro_ur,
                micro_grounded_factuality_index=m_micro_gfi,
                micro_penalized_factuality_score=m_micro_cpfs,
                macro_evidence_support_rate=m_macro_esr,
                macro_confirmed_contradiction_rate=m_macro_ccr,
                macro_unsupported_rate=m_macro_usr,
                macro_unverifiable_rate=m_macro_uvr,
                macro_combined_uncertainty_rate=m_macro_ur
            )

        # Documentation and notes
        notes = [
            f"Dataset evaluated over {total_responses} responses containing {total_extracted_claims} total propositions.",
            f"Micro-pooled metrics aggregate all {total_factual_claims} empirical factual claims across responses before calculating rates, preventing Simpson's paradox.",
            f"Macro-averaged metrics report the unweighted average across the {num_valid} responses that contained factual claims.",
            f"{zero_factual_responses} responses had zero testable factual claims and were excluded from factuality rate denominators."
        ]

        return DatasetScore(
            dataset_name=dataset_name,
            total_responses=total_responses,
            responses_with_factual_claims=num_valid,
            responses_zero_factual_claims=zero_factual_responses,
            empty_responses=empty_responses,
            total_extracted_claims=total_extracted_claims,
            total_factual_claims=total_factual_claims,
            total_non_factual_claims=total_non_factual_claims,
            pooled_supported_count=pooled_supported,
            pooled_contradicted_count=pooled_contradicted,
            pooled_unsupported_count=pooled_unsupported,
            pooled_unverifiable_count=pooled_unverifiable,
            pooled_non_factual_count=pooled_non_factual,
            micro_evidence_support_rate=micro_esr,
            micro_confirmed_contradiction_rate=micro_ccr,
            micro_unsupported_rate=micro_usr,
            micro_unverifiable_rate=micro_uvr,
            micro_combined_uncertainty_rate=micro_ur,
            micro_grounded_factuality_index=micro_gfi,
            micro_penalized_factuality_score=micro_cpfs,
            macro_evidence_support_rate=macro_esr,
            macro_confirmed_contradiction_rate=macro_ccr,
            macro_unsupported_rate=macro_usr,
            macro_unverifiable_rate=macro_uvr,
            macro_combined_uncertainty_rate=macro_ur,
            models_breakdown=models_breakdown,
            mathematical_invariants_verified=invariants_verified,
            aggregation_notes=notes
        )

