"""
VeriLLM Response-Level Scoring and Quantification Engine.
Computes mathematically rigorous factuality rates, overall indices (GFI, CPFS, HSI),
and provenance traces for individual LLM responses.
"""

import re
from typing import List, Optional, Union, Dict, Set
from pydantic import BaseModel

from ml_engine.claim_verifier.models import (
    VerificationReport,
    VerifiedClaim,
    VerificationCategory
)
from ml_engine.scorer.models import (
    ResponseScore,
    ResponseCounts,
    ResponseRates,
    FactualityIndices,
    ClaimSummary,
    ScoringConfig
)


def _normalize_claim_text(text: str) -> str:
    """Normalize proposition text for duplicate detection."""
    clean = text.lower().strip()
    clean = re.sub(r'[\s\.\,\;\:\!\?]+$', '', clean)
    clean = re.sub(r'\s+', ' ', clean)
    return clean


class ResponseScorer:
    """
    Evaluates verified propositions of a single LLM response and produces
    structured, mathematically consistent factuality metrics.
    """

    def __init__(self, config: Optional[ScoringConfig] = None):
        self.config = config or ScoringConfig()

    def score_response(
        self,
        report_or_claims: Union[VerificationReport, List[VerifiedClaim]],
        response_id: str = "resp_001",
        model_id: Optional[str] = None,
        document_name: Optional[str] = None,
        execution_time_seconds: Optional[float] = None
    ) -> ResponseScore:
        """
        Compute factuality metrics, indices, and traceability summaries for a single response.

        Args:
            report_or_claims: Either a VerificationReport or a list of VerifiedClaim objects.
            response_id: Unique identifier for the evaluated response.
            model_id: Optional identifier for the LLM that generated the text (e.g. 'gpt-4o').
            document_name: Optional reference document name.
            execution_time_seconds: Optional verification duration in seconds.

        Returns:
            ResponseScore containing partitioned counts, rates, indices, and invariant checks.
        """
        # Unpack input
        if isinstance(report_or_claims, VerificationReport):
            raw_claims = report_or_claims.claims
            doc_name = document_name or report_or_claims.document_name
            exec_time = execution_time_seconds if execution_time_seconds is not None else report_or_claims.execution_time_seconds
        else:
            raw_claims = report_or_claims
            doc_name = document_name
            exec_time = execution_time_seconds

        is_empty = len(raw_claims) == 0

        # Detect duplicates
        seen_propositions: Set[str] = set()
        duplicate_count = 0
        claims_to_evaluate: List[VerifiedClaim] = []

        for c in raw_claims:
            norm = _normalize_claim_text(c.claim_text)
            if norm in seen_propositions:
                duplicate_count += 1
                if not self.config.deduplicate_claims:
                    claims_to_evaluate.append(c)
            else:
                seen_propositions.add(norm)
                claims_to_evaluate.append(c)

        # Partition claims into empirical factual vs non-factual
        factual_claims: List[VerifiedClaim] = []
        non_factual_claims: List[VerifiedClaim] = []

        for c in claims_to_evaluate:
            # A claim is non-factual if tagged is_factual=False or classified as NON_FACTUAL
            if not c.is_factual or c.classification == VerificationCategory.NON_FACTUAL:
                non_factual_claims.append(c)
            else:
                factual_claims.append(c)

        total_claims_count = len(claims_to_evaluate)
        N = len(factual_claims)
        non_factual_count_val = len(non_factual_claims)

        # Count factual category occurrences
        supported_count = sum(1 for c in factual_claims if c.classification == VerificationCategory.SUPPORTED)
        contradicted_count = sum(1 for c in factual_claims if c.classification == VerificationCategory.CONTRADICTED)
        unsupported_count = sum(1 for c in factual_claims if c.classification == VerificationCategory.UNSUPPORTED)
        unverifiable_count = sum(1 for c in factual_claims if c.classification == VerificationCategory.UNVERIFIABLE)

        counts = ResponseCounts(
            total_claims=total_claims_count,
            factual_claims=N,
            non_factual_claims=non_factual_count_val,
            supported_count=supported_count,
            contradicted_count=contradicted_count,
            unsupported_count=unsupported_count,
            unverifiable_count=unverifiable_count,
            non_factual_count=non_factual_count_val,
            duplicate_claims_count=duplicate_count
        )

        # Build claim summaries for traceability
        claim_summaries: List[ClaimSummary] = []
        for c in raw_claims:
            citation_str = None
            if c.primary_evidence and c.primary_evidence.chunk:
                chunk = c.primary_evidence.chunk
                citation_str = f"{chunk.document_name} (Page {chunk.page_number}, chunk: {chunk.chunk_id})"

            claim_summaries.append(ClaimSummary(
                claim_id=c.claim_id,
                claim_text=c.claim_text,
                is_factual=c.is_factual,
                classification=c.classification.value if hasattr(c.classification, 'value') else str(c.classification),
                citation=citation_str,
                minicheck_probability=c.scores.minicheck_probability if c.scores else None,
                nli_entailment_prob=c.scores.nli_entailment_prob if c.scores else None,
                nli_contradiction_prob=c.scores.nli_contradiction_prob if c.scores else None,
                nli_neutral_prob=c.scores.nli_neutral_prob if c.scores else None,
                custom_deberta_supported_prob=c.scores.custom_deberta_supported_prob if c.scores else None,
                custom_deberta_contradiction_prob=c.scores.custom_deberta_contradiction_prob if c.scores else None,
                custom_deberta_unsupported_prob=c.scores.custom_deberta_unsupported_prob if c.scores else None,
                custom_deberta_predicted_label=c.scores.custom_deberta_predicted_label if c.scores else None,
                reasoning_trace=c.reasoning_trace
            ))

        # Handle zero factual claims
        if N == 0:
            rates = ResponseRates()
            indices = FactualityIndices()
            desc = (
                f"Evaluated 0 empirical factual claims ({non_factual_count_val} non-factual statements excluded). "
                "All factuality rates and indices are undefined (N/A)."
            )
            return ResponseScore(
                response_id=response_id,
                model_id=model_id,
                document_name=doc_name,
                counts=counts,
                rates=rates,
                indices=indices,
                has_factual_claims=False,
                is_empty_response=is_empty,
                contains_duplicates=duplicate_count > 0,
                mathematical_invariants_verified=True,
                denominator_description=desc,
                claims=claim_summaries,
                execution_time_seconds=exec_time
            )

        # Calculate exact percentage rates
        esr = round((supported_count / N) * 100.0, 2)
        ccr = round((contradicted_count / N) * 100.0, 2)
        usr = round((unsupported_count / N) * 100.0, 2)
        uvr = round((unverifiable_count / N) * 100.0, 2)
        ur = round(((unsupported_count + unverifiable_count) / N) * 100.0, 2)

        rates = ResponseRates(
            evidence_support_rate=esr,
            confirmed_contradiction_rate=ccr,
            unsupported_rate=usr,
            unverifiable_rate=uvr,
            combined_uncertainty_rate=ur
        )

        # Calculate overall factuality indices
        # 1. Grounded Factuality Index (GFI): Equal to Evidence Support Rate
        gfi = esr

        # 2. Contradiction-Penalized Factuality Score (CPFS): max(0.0, (supported - lambda * contradicted) / N) * 100
        lambda_weight = self.config.contradiction_penalty_weight
        penalized_val = max(0.0, (supported_count - (lambda_weight * contradicted_count)) / N) * 100.0
        cpfs = round(penalized_val, 2)

        # 3. Hallucination Severity Index (HSI): Equal to Confirmed Contradiction Rate
        hsi = ccr

        indices = FactualityIndices(
            grounded_factuality_index=gfi,
            contradiction_penalized_factuality=cpfs,
            hallucination_severity_index=hsi
        )

        # Invariant Verification: ESR + CCR + USR + UVR == 100.00% (within 0.05 rounding tolerance)
        partition_sum = esr + ccr + usr + uvr
        invariants_verified = abs(partition_sum - 100.0) <= 0.05

        desc = (
            f"Metrics evaluated over {N} empirical factual claims "
            f"({non_factual_count_val} non-factual statements excluded from factuality denominator)."
        )

        return ResponseScore(
            response_id=response_id,
            model_id=model_id,
            document_name=doc_name,
            counts=counts,
            rates=rates,
            indices=indices,
            has_factual_claims=True,
            is_empty_response=is_empty,
            contains_duplicates=duplicate_count > 0,
            mathematical_invariants_verified=invariants_verified,
            denominator_description=desc,
            claims=claim_summaries,
            execution_time_seconds=exec_time
        )

