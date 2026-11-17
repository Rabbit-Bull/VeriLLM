"""
VeriLLM Claim Verifier Pipeline Orchestrator.
Coordinates claim extraction, evidence matching, MiniCheck factual checking,
NLI 3-way inference, deterministic decision synthesis, and metric computation.
"""

import logging
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Union
from ml_engine.claim_extractor.models import ExtractedClaim
from ml_engine.evidence_retriever.models import ClaimEvidenceMatch, RetrievedPassage
from ml_engine.claim_verifier.models import (
    VerificationCategory,
    VerifiedClaim,
    VerificationMetrics,
    VerificationReport
)
from ml_engine.claim_verifier.minicheck_verifier import MiniCheckVerifier
from ml_engine.claim_verifier.nli_verifier import NLIVerifier
from ml_engine.claim_verifier.decision_strategy import VerificationDecisionStrategy
from ml_engine.claim_verifier.custom_deberta_verifier import (
    CustomDebertaPrediction,
    CustomDebertaVerifier,
)

logger = logging.getLogger(__name__)


def _attach_shadow_scores(
    verified_claim: VerifiedClaim,
    prediction: Optional[CustomDebertaPrediction],
) -> VerifiedClaim:
    """Return a safe copy of verified_claim populated with Custom DeBERTa shadow fields."""
    if prediction is None:
        return verified_claim

    updated_scores = verified_claim.scores.model_copy(
        update={
            "custom_deberta_supported_prob": prediction.supported_proxy_prob,
            "custom_deberta_contradiction_prob": prediction.contradicted_prob,
            "custom_deberta_unsupported_prob": prediction.unsupported_prob,
            "custom_deberta_predicted_label": prediction.predicted_label,
        }
    )
    return verified_claim.model_copy(update={"scores": updated_scores})


class ClaimVerifier:
    """
    End-to-end factual verification engine.
    """

    def __init__(
        self,
        minicheck_model: str = "roberta-large",
        nli_model: str = "cross-encoder/nli-deberta-v3-xsmall",
        evidence_floor_threshold: float = 0.25,
        entailment_threshold: float = 0.50,
        contradiction_threshold: float = 0.50,
        conflict_threshold: float = 0.70,
        use_custom_model: bool = False,
        custom_model_path: Optional[Union[str, Path]] = None,
    ):
        self.minicheck = MiniCheckVerifier(model_name=minicheck_model)
        self.nli = NLIVerifier(model_name=nli_model)
        self.decision_strategy = VerificationDecisionStrategy(
            evidence_floor_threshold=evidence_floor_threshold,
            entailment_threshold=entailment_threshold,
            contradiction_threshold=contradiction_threshold,
            conflict_threshold=conflict_threshold
        )
        self.use_custom_model = use_custom_model
        self.custom_model_path = custom_model_path
        # Lazily initialized only when shadow mode is enabled and eligible claims exist.
        self.custom_deberta: Optional[CustomDebertaVerifier] = None
        self._custom_model_unavailable: bool = False

    def _get_custom_deberta_verifier(self) -> Optional[CustomDebertaVerifier]:
        """Lazily initialize the optional CustomDebertaVerifier using its path-keyed cache."""
        if not self.use_custom_model:
            return None

        if self.custom_deberta is not None:
            return self.custom_deberta

        if self._custom_model_unavailable:
            return None

        try:
            self.custom_deberta = CustomDebertaVerifier(
                model_path=self.custom_model_path
            )
        except FileNotFoundError as exc:
            if self.custom_model_path is None:
                logger.warning(
                    "Default Custom DeBERTa checkpoint is unavailable; "
                    "continuing with baseline verification only: %s",
                    exc,
                )
                self._custom_model_unavailable = True
                return None
            raise

        return self.custom_deberta

    def _run_custom_deberta_shadow(
        self,
        pairs: List[Tuple[str, str]],
        eval_indices: List[int],
    ) -> Dict[int, CustomDebertaPrediction]:
        """Run batched shadow-mode Custom DeBERTa scoring without affecting baseline flow."""
        if not self.use_custom_model or not pairs:
            return {}

        custom_deberta = self._get_custom_deberta_verifier()
        if custom_deberta is None:
            return {}

        predictions = custom_deberta.predict_pairs(list(pairs))
        if len(predictions) != len(eval_indices):
            raise ValueError(
                f"CustomDebertaVerifier returned {len(predictions)} predictions "
                f"for {len(eval_indices)} input pairs."
            )

        return {
            original_idx: prediction
            for original_idx, prediction in zip(eval_indices, predictions)
        }

    def verify_claims(
        self,
        claims: List[ExtractedClaim],
        evidence_matches: List[ClaimEvidenceMatch],
        document_name: Optional[str] = None
    ) -> VerificationReport:
        """
        Verifies all extracted claims against their retrieved evidence passages.

        Args:
            claims: List of ExtractedClaim objects from Phase 3.
            evidence_matches: List of ClaimEvidenceMatch objects from Phase 4.
            document_name: Optional name of the reference document.

        Returns:
            VerificationReport containing verified claims, reasoning traces, and metrics.
        """
        start_time = time.time()

        if not claims:
            # Handle empty response edge case
            metrics = VerificationMetrics(
                total_claims=0,
                factual_claims=0,
                non_factual_claims=0,
                supported_count=0,
                contradicted_count=0,
                unsupported_count=0,
                unverifiable_count=0,
                non_factual_count=0,
                has_factual_claims=False,
                evidence_support_rate=None,
                confirmed_contradiction_rate=None,
                unsupported_rate=None,
                unverifiable_rate=None,
                uncertainty_rate=None,
                denominator_description="No claims to evaluate. Factuality metrics are undefined (N/A)."
            )
            return VerificationReport(
                document_name=document_name,
                total_claims_processed=0,
                claims=[],
                metrics=metrics,
                execution_time_seconds=0.0
            )

        # Build evidence lookup table: claim_id -> ClaimEvidenceMatch
        evidence_by_claim_id: Dict[str, ClaimEvidenceMatch] = {
            m.claim_id: m for m in evidence_matches
        }

        # Step 1: Collect pairs that require ML model scoring
        # (Exclude non-factual claims or claims with zero evidence from GPU/CPU inference)
        mc_docs_to_eval = []
        mc_claims_to_eval = []
        nli_pairs_to_eval = []
        eval_indices = []
        deberta_pairs_to_eval: List[Tuple[str, str]] = []
        deberta_eval_indices: List[int] = []

        for idx, claim in enumerate(claims):
            match = evidence_by_claim_id.get(claim.claim_id)
            primary_evidence = match.best_passage if match else None

            if claim.is_factual and primary_evidence and primary_evidence.chunk.text.strip():
                mc_docs_to_eval.append(primary_evidence.chunk.text)
                mc_claims_to_eval.append(claim.text)
                nli_pairs_to_eval.append((primary_evidence.chunk.text, claim.text))
                eval_indices.append(idx)

                if self.use_custom_model and claim.text.strip():
                    deberta_pairs_to_eval.append((primary_evidence.chunk.text, claim.text))
                    deberta_eval_indices.append(idx)

        # Step 2: Batch inference for MiniCheck and NLI
        mc_results_map: Dict[int, Tuple[int, float]] = {}
        nli_results_map: Dict[int, Tuple[float, float, float]] = {}

        if mc_docs_to_eval:
            mc_scores = self.minicheck.score_pairs(mc_docs_to_eval, mc_claims_to_eval)
            nli_scores = self.nli.predict_pairs(nli_pairs_to_eval)

            for original_idx, mc_s, nli_s in zip(eval_indices, mc_scores, nli_scores):
                mc_results_map[original_idx] = mc_s
                nli_results_map[original_idx] = nli_s

        # Step 2b: Optional shadow-mode Custom DeBERTa batch inference
        deberta_results_map = self._run_custom_deberta_shadow(
            deberta_pairs_to_eval, deberta_eval_indices
        )

        # Step 3: Check secondary evidence passages for conflict detection where applicable
        conflict_scores_map: Dict[int, List[Tuple[float, float, float]]] = {}
        for original_idx in eval_indices:
            claim = claims[original_idx]
            match = evidence_by_claim_id.get(claim.claim_id)
            if match and len(match.retrieved_passages) >= 2:
                # Only check for conflict among relevant passages (similarity >= 0.45)
                rel_passages = [
                    p for p in match.retrieved_passages[:3]
                    if p.chunk.text.strip() and p.similarity_score >= 0.45
                ]
                if len(rel_passages) >= 2:
                    sec_pairs = [
                        (p.chunk.text, claim.text)
                        for p in rel_passages
                    ]
                    sec_scores = self.nli.predict_pairs(sec_pairs)
                    conflict_scores_map[original_idx] = sec_scores

        # Step 4: Run decision strategy for each claim
        verified_claims: List[VerifiedClaim] = []

        for idx, claim in enumerate(claims):
            match = evidence_by_claim_id.get(claim.claim_id)
            primary_evidence = match.best_passage if match else None
            all_evidence = match.retrieved_passages if match else []

            mc_score = mc_results_map.get(idx)
            nli_score = nli_results_map.get(idx)
            sec_nli = conflict_scores_map.get(idx)

            verified = self.decision_strategy.evaluate_claim(
                claim_id=claim.claim_id,
                claim_text=claim.text,
                source_sentence=claim.source_sentence,
                start_char=claim.start_char,
                end_char=claim.end_char,
                claim_start_char=claim.claim_start_char,
                claim_end_char=claim.claim_end_char,
                offset_type=claim.offset_type,
                is_factual=claim.is_factual,
                primary_evidence=primary_evidence,
                all_evidence=all_evidence,
                minicheck_score=mc_score,
                nli_scores=nli_score,
                secondary_nli_scores=sec_nli
            )
            verified_claims.append(
                _attach_shadow_scores(verified, deberta_results_map.get(idx))
            )

        # Step 5: Calculate Factuality and Uncertainty Metrics
        total_claims = len(verified_claims)
        factual_claims = sum(1 for c in verified_claims if c.is_factual)
        non_factual_claims = total_claims - factual_claims

        # Mutually exclusive counts among empirical factual claims
        supported_count = sum(
            1 for c in verified_claims 
            if c.is_factual and c.classification == VerificationCategory.SUPPORTED
        )
        contradicted_count = sum(
            1 for c in verified_claims 
            if c.is_factual and c.classification == VerificationCategory.CONTRADICTED
        )
        unsupported_count = sum(
            1 for c in verified_claims 
            if c.is_factual and c.classification == VerificationCategory.UNSUPPORTED
        )
        unverifiable_count = sum(
            1 for c in verified_claims 
            if c.is_factual and c.classification == VerificationCategory.UNVERIFIABLE
        )
        non_factual_count = sum(
            1 for c in verified_claims 
            if not c.is_factual or c.classification == VerificationCategory.NON_FACTUAL
        )

        if factual_claims > 0:
            support_rate = round((supported_count / factual_claims) * 100.0, 2)
            contradiction_rate = round((contradicted_count / factual_claims) * 100.0, 2)
            unsupported_rate = round((unsupported_count / factual_claims) * 100.0, 2)
            unverifiable_rate = round((unverifiable_count / factual_claims) * 100.0, 2)
            uncertainty_rate = round(((unsupported_count + unverifiable_count) / factual_claims) * 100.0, 2)
            has_factual = True
            denominator_desc = (
                f"Metrics evaluated over {factual_claims} factual claims "
                f"({non_factual_claims} non-factual statements excluded from factuality denominator)."
            )
        else:
            support_rate = None
            contradiction_rate = None
            unsupported_rate = None
            unverifiable_rate = None
            uncertainty_rate = None
            has_factual = False
            denominator_desc = (
                f"No empirical factual claims to evaluate ({non_factual_claims} non-factual statements excluded). "
                "Factuality metrics are undefined (N/A)."
            )

        metrics = VerificationMetrics(
            total_claims=total_claims,
            factual_claims=factual_claims,
            non_factual_claims=non_factual_claims,
            supported_count=supported_count,
            contradicted_count=contradicted_count,
            unsupported_count=unsupported_count,
            unverifiable_count=unverifiable_count,
            non_factual_count=non_factual_count,
            has_factual_claims=has_factual,
            evidence_support_rate=support_rate,
            confirmed_contradiction_rate=contradiction_rate,
            unsupported_rate=unsupported_rate,
            unverifiable_rate=unverifiable_rate,
            uncertainty_rate=uncertainty_rate,
            denominator_description=denominator_desc
        )

        duration = time.time() - start_time

        return VerificationReport(
            document_name=document_name,
            total_claims_processed=total_claims,
            claims=verified_claims,
            metrics=metrics,
            execution_time_seconds=round(duration, 3)
        )
