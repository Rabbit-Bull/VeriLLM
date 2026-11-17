"""
VeriLLM Verification Decision Strategy.
Synthesizes MiniCheck scores, 3-way NLI probabilities, retrieval similarity,
and evidence consistency into transparent 4-way classifications with reasoning traces.
"""

from typing import Optional, List, Tuple
from ml_engine.claim_verifier.models import (
    VerificationCategory,
    ModelVerificationScore,
    VerifiedClaim
)
from ml_engine.evidence_retriever.models import RetrievedPassage


class VerificationDecisionStrategy:
    """
    Deterministic rule-based decision synthesizer.
    Combines model probabilities and retrieval metadata to classify claims into:
    SUPPORTED, CONTRADICTED, UNSUPPORTED, or UNVERIFIABLE.
    """

    def __init__(
        self,
        evidence_floor_threshold: float = 0.25,
        entailment_threshold: float = 0.50,
        contradiction_threshold: float = 0.50,
        conflict_threshold: float = 0.70
    ):
        """
        Args:
            evidence_floor_threshold: Minimum cosine similarity required to consider evidence relevant.
            entailment_threshold: Probability threshold above which a claim is deemed SUPPORTED.
            contradiction_threshold: Probability threshold above which a claim is deemed CONTRADICTED.
            conflict_threshold: Threshold above which two conflicting passages trigger UNVERIFIABLE.
        """
        self.evidence_floor_threshold = evidence_floor_threshold
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold
        self.conflict_threshold = conflict_threshold

    def evaluate_claim(
        self,
        claim_id: str,
        claim_text: str,
        source_sentence: str,
        start_char: int,
        end_char: int,
        claim_start_char: Optional[int],
        claim_end_char: Optional[int],
        offset_type: str,
        is_factual: bool,
        primary_evidence: Optional[RetrievedPassage],
        all_evidence: List[RetrievedPassage],
        minicheck_score: Optional[Tuple[int, float]],
        nli_scores: Optional[Tuple[float, float, float]],
        secondary_nli_scores: Optional[List[Tuple[float, float, float]]] = None
    ) -> VerifiedClaim:
        """
        Applies the decision matrix to classify a single claim.
        """
        # Unpack model scores
        mc_pred, mc_prob = minicheck_score if minicheck_score is not None else (None, None)
        p_entail, p_contra, p_neutral = nli_scores if nli_scores is not None else (None, None, None)
        sim_score = primary_evidence.similarity_score if primary_evidence else None

        score_model = ModelVerificationScore(
            minicheck_prediction=mc_pred,
            minicheck_probability=mc_prob,
            nli_entailment_prob=p_entail,
            nli_contradiction_prob=p_contra,
            nli_neutral_prob=p_neutral,
            retrieval_similarity=sim_score
        )

        # -------------------------------------------------------------
        # Rule 1: Non-factual statements (opinions, greetings, meta-text)
        # -------------------------------------------------------------
        if not is_factual:
            return VerifiedClaim(
                claim_id=claim_id,
                claim_text=claim_text,
                source_sentence=source_sentence,
                start_char=start_char,
                end_char=end_char,
                claim_start_char=claim_start_char,
                claim_end_char=claim_end_char,
                offset_type=offset_type,
                is_factual=is_factual,
                classification=VerificationCategory.NON_FACTUAL,
                scores=score_model,
                primary_evidence=primary_evidence,
                all_evidence=all_evidence,
                decision_rule="NON_FACTUAL_ASSERTION",
                reasoning_trace=(
                    "Statement is subjective, opinion-based, or conversational meta-text "
                    "and does not assert an empirical factual proposition."
                )
            )

        # -------------------------------------------------------------
        # Rule 2: Empty evidence
        # -------------------------------------------------------------
        if primary_evidence is None or not primary_evidence.chunk.text.strip():
            return VerifiedClaim(
                claim_id=claim_id,
                claim_text=claim_text,
                source_sentence=source_sentence,
                start_char=start_char,
                end_char=end_char,
                claim_start_char=claim_start_char,
                claim_end_char=claim_end_char,
                offset_type=offset_type,
                is_factual=is_factual,
                classification=VerificationCategory.UNVERIFIABLE,
                scores=score_model,
                primary_evidence=None,
                all_evidence=[],
                decision_rule="EMPTY_EVIDENCE",
                reasoning_trace=(
                    "No reference evidence was provided or retrieved to verify this claim."
                )
            )

        # -------------------------------------------------------------
        # Rule 3: Evidence below minimum similarity floor
        # -------------------------------------------------------------
        if sim_score is not None and sim_score < self.evidence_floor_threshold:
            return VerifiedClaim(
                claim_id=claim_id,
                claim_text=claim_text,
                source_sentence=source_sentence,
                start_char=start_char,
                end_char=end_char,
                claim_start_char=claim_start_char,
                claim_end_char=claim_end_char,
                offset_type=offset_type,
                is_factual=is_factual,
                classification=VerificationCategory.UNVERIFIABLE,
                scores=score_model,
                primary_evidence=primary_evidence,
                all_evidence=all_evidence,
                decision_rule="EVIDENCE_SIMILARITY_BELOW_FLOOR",
                reasoning_trace=(
                    f"Top retrieved passage has low semantic similarity ({sim_score:.4f} < {self.evidence_floor_threshold:.2f}). "
                    "The reference document lacks sufficient context regarding this proposition."
                )
            )

        # -------------------------------------------------------------
        # Rule 4: Conflicting evidence passages
        # -------------------------------------------------------------
        if secondary_nli_scores and len(secondary_nli_scores) >= 2:
            has_strong_entailment = any(s[0] >= self.conflict_threshold for s in secondary_nli_scores)
            has_strong_contradiction = any(s[1] >= self.conflict_threshold for s in secondary_nli_scores)
            if has_strong_entailment and has_strong_contradiction and (mc_prob is None or mc_prob < 0.85):
                return VerifiedClaim(
                    claim_id=claim_id,
                    claim_text=claim_text,
                    source_sentence=source_sentence,
                    start_char=start_char,
                    end_char=end_char,
                    claim_start_char=claim_start_char,
                    claim_end_char=claim_end_char,
                    offset_type=offset_type,
                    is_factual=is_factual,
                    classification=VerificationCategory.UNVERIFIABLE,
                    scores=score_model,
                    primary_evidence=primary_evidence,
                    all_evidence=all_evidence,
                    decision_rule="CONFLICTING_EVIDENCE_PASSAGES",
                    reasoning_trace=(
                        "Retrieved candidate passages provide directly conflicting assertions "
                        f"(one affirms with p >= {self.conflict_threshold:.2f}, while another contradicts with p >= {self.conflict_threshold:.2f})."
                    )
                )

        # Ensure model scores are available
        effective_mc_prob = mc_prob if mc_prob is not None else 0.0
        effective_p_entail = p_entail if p_entail is not None else 0.0
        effective_p_contra = p_contra if p_contra is not None else 0.0
        effective_p_neutral = p_neutral if p_neutral is not None else 1.0

        page_str = f"Page {primary_evidence.chunk.page_number}" if primary_evidence.chunk.page_number else "N/A"
        doc_name = primary_evidence.chunk.document_name

        # -------------------------------------------------------------
        # Rule 5: SUPPORTED
        # -------------------------------------------------------------
        # Case A: Strong MiniCheck factual support (>=0.80) with no dominant contradiction (p_contra < 0.50)
        # Case B: Threshold entailment: MiniCheck or NLI entailment >= 0.50 with low contradiction (p_contra < 0.40)
        is_supported = (
            (effective_mc_prob >= 0.80 and effective_p_contra < 0.50) or
            ((effective_mc_prob >= self.entailment_threshold or effective_p_entail >= self.entailment_threshold) and effective_p_contra < 0.40)
        )
        if is_supported:
            return VerifiedClaim(
                claim_id=claim_id,
                claim_text=claim_text,
                source_sentence=source_sentence,
                start_char=start_char,
                end_char=end_char,
                claim_start_char=claim_start_char,
                claim_end_char=claim_end_char,
                offset_type=offset_type,
                is_factual=is_factual,
                classification=VerificationCategory.SUPPORTED,
                scores=score_model,
                primary_evidence=primary_evidence,
                all_evidence=all_evidence,
                decision_rule="EVIDENCE_ENTAILMENT",
                reasoning_trace=(
                    f"Evidence in '{doc_name}' ({page_str}) entails the claim. "
                    f"[MiniCheck support p={effective_mc_prob:.4f}, NLI entailment={effective_p_entail:.4f}, contradiction={effective_p_contra:.4f}]"
                )
            )

        # -------------------------------------------------------------
        # Rule 6: CONTRADICTED
        # -------------------------------------------------------------
        # NLI contradiction is strong and exceeds neutral, while MiniCheck support is low
        if effective_p_contra >= self.contradiction_threshold and effective_p_contra > effective_p_neutral and effective_mc_prob < self.entailment_threshold:
            return VerifiedClaim(
                claim_id=claim_id,
                claim_text=claim_text,
                source_sentence=source_sentence,
                start_char=start_char,
                end_char=end_char,
                claim_start_char=claim_start_char,
                claim_end_char=claim_end_char,
                offset_type=offset_type,
                is_factual=is_factual,
                classification=VerificationCategory.CONTRADICTED,
                scores=score_model,
                primary_evidence=primary_evidence,
                all_evidence=all_evidence,
                decision_rule="EVIDENCE_CONTRADICTION",
                reasoning_trace=(
                    f"Evidence in '{doc_name}' ({page_str}) directly refutes the claim. "
                    f"[NLI contradiction={effective_p_contra:.4f}, MiniCheck support p={effective_mc_prob:.4f}, NLI neutral={effective_p_neutral:.4f}]"
                )
            )

        # -------------------------------------------------------------
        # Rule 7: UNSUPPORTED
        # -------------------------------------------------------------
        # Default: MiniCheck < 0.50 and NLI contradiction is not dominant (neutral or unmentioned)
        return VerifiedClaim(
            claim_id=claim_id,
            claim_text=claim_text,
            source_sentence=source_sentence,
            start_char=start_char,
            end_char=end_char,
            claim_start_char=claim_start_char,
            claim_end_char=claim_end_char,
            offset_type=offset_type,
            is_factual=is_factual,
            classification=VerificationCategory.UNSUPPORTED,
            scores=score_model,
            primary_evidence=primary_evidence,
            all_evidence=all_evidence,
            decision_rule="EVIDENCE_UNMENTIONED_NEUTRAL",
            reasoning_trace=(
                f"Retrieved passage in '{doc_name}' ({page_str}) discusses related context but does not contain "
                f"factual assertions confirming or refuting the claim. "
                f"[NLI neutral={effective_p_neutral:.4f}, MiniCheck support p={effective_mc_prob:.4f}, NLI contradiction={effective_p_contra:.4f}]"
            )
        )
