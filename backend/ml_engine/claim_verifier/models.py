"""
VeriLLM Claim Verification Data Models.
Defines strongly-typed Pydantic schemas for verification results, model scores,
probabilistic reasoning traces, and factuality metrics.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from ml_engine.evidence_retriever.models import RetrievedPassage


class VerificationCategory(str, Enum):
    """
    Standard classification categories for proposition verification.
    The first four apply exclusively to empirical factual claims;
    NON_FACTUAL applies to subjective opinions, greetings, and conversational remarks.
    """
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNSUPPORTED = "UNSUPPORTED"
    UNVERIFIABLE = "UNVERIFIABLE"
    NON_FACTUAL = "NON_FACTUAL"


class ModelVerificationScore(BaseModel):
    """
    Detailed model outputs from MiniCheck, NLI Cross-Encoder, and optional Custom DeBERTa.
    Separates raw model outputs from the synthesized final classification.
    """
    minicheck_prediction: Optional[int] = Field(
        default=None,
        description="Binary prediction from MiniCheck (1 = supported, 0 = unsupported/contradicted)."
    )
    minicheck_probability: Optional[float] = Field(
        default=None,
        description="Support probability from MiniCheck in range [0.0, 1.0]."
    )
    nli_entailment_prob: Optional[float] = Field(
        default=None,
        description="Entailment probability from 3-way NLI cross-encoder in range [0.0, 1.0]."
    )
    nli_contradiction_prob: Optional[float] = Field(
        default=None,
        description="Contradiction probability from 3-way NLI cross-encoder in range [0.0, 1.0]."
    )
    nli_neutral_prob: Optional[float] = Field(
        default=None,
        description="Neutral/unmentioned probability from 3-way NLI cross-encoder in range [0.0, 1.0]."
    )
    retrieval_similarity: Optional[float] = Field(
        default=None,
        description="Cosine similarity score of the primary evidence passage."
    )
    custom_deberta_supported_prob: Optional[float] = Field(
        default=None,
        description="Normalized softmax score for SUPPORTED_PROXY from fine-tuned RAGTruth DeBERTa (not a calibrated truth probability)."
    )
    custom_deberta_contradiction_prob: Optional[float] = Field(
        default=None,
        description="Normalized softmax score for CONTRADICTED from fine-tuned RAGTruth DeBERTa (not a calibrated truth probability)."
    )
    custom_deberta_unsupported_prob: Optional[float] = Field(
        default=None,
        description="Normalized softmax score for UNSUPPORTED from fine-tuned RAGTruth DeBERTa (not a calibrated truth probability)."
    )
    custom_deberta_predicted_label: Optional[str] = Field(
        default=None,
        description="Predicted class label from fine-tuned RAGTruth DeBERTa (CONTRADICTED, SUPPORTED_PROXY, or UNSUPPORTED)."
    )


class VerifiedClaim(BaseModel):
    """
    Represents an atomic claim evaluated against reference evidence.
    Includes provenance, character offsets, model scores, decision rule, and reasoning trace.
    """
    claim_id: str = Field(..., description="Unique claim identifier.")
    claim_text: str = Field(..., description="The atomic factual proposition.")
    source_sentence: str = Field(..., description="Original sentence in candidate response.")
    start_char: int = Field(..., description="Start character offset of source sentence.")
    end_char: int = Field(..., description="End character offset of source sentence.")
    claim_start_char: Optional[int] = Field(
        default=None,
        description="Exact start index of sub-clause in response where localized."
    )
    claim_end_char: Optional[int] = Field(
        default=None,
        description="Exact end index of sub-clause in response where localized."
    )
    offset_type: str = Field(
        default="EXACT",
        description="Offset fidelity type (EXACT, CLAUSE_SPAN, EMBEDDED_CLAUSE, PRONOUN_RESOLVED)."
    )
    is_factual: bool = Field(
        default=True,
        description="Whether statement asserts an empirical fact (False for opinion/greeting)."
    )
    classification: VerificationCategory = Field(
        ...,
        description="Final verification classification (SUPPORTED, CONTRADICTED, UNSUPPORTED, UNVERIFIABLE, NON_FACTUAL)."
    )
    scores: ModelVerificationScore = Field(
        default_factory=ModelVerificationScore,
        description="Model prediction scores."
    )
    primary_evidence: Optional[RetrievedPassage] = Field(
        default=None,
        description="Top-1 highest-scoring retrieved passage used for verification."
    )
    all_evidence: List[RetrievedPassage] = Field(
        default_factory=list,
        description="All retrieved candidate passages evaluated."
    )
    decision_rule: str = Field(
        ...,
        description="Identifier of the deterministic rule that triggered the classification."
    )
    reasoning_trace: str = Field(
        ...,
        description="Human-readable explanation of why this claim was classified into this category."
    )


class VerificationMetrics(BaseModel):
    """
    Aggregated factuality and hallucination metrics computed exclusively from empirical factual claims.
    Denominator and treatment of non-factual claims are explicitly documented.
    """
    total_claims: int = Field(..., description="Total count of all extracted claims.")
    factual_claims: int = Field(..., description="Count of empirical factual claims evaluated (N).")
    non_factual_claims: int = Field(..., description="Count of subjective/conversational claims excluded from factual denominator.")
    supported_count: int = Field(..., description="Number of factual claims verified as SUPPORTED.")
    contradicted_count: int = Field(..., description="Number of factual claims verified as CONTRADICTED.")
    unsupported_count: int = Field(..., description="Number of factual claims verified as UNSUPPORTED.")
    unverifiable_count: int = Field(..., description="Number of factual claims verified as UNVERIFIABLE.")
    non_factual_count: int = Field(..., description="Number of claims classified as NON_FACTUAL.")
    has_factual_claims: bool = Field(
        default=True,
        description="True if factual_claims > 0; False if response contained zero factual claims."
    )
    evidence_support_rate: Optional[float] = Field(
        default=None,
        description="Percentage of factual claims supported: (supported / factual_claims) * 100. None if factual_claims == 0."
    )
    confirmed_contradiction_rate: Optional[float] = Field(
        default=None,
        description="Percentage of factual claims contradicted: (contradicted / factual_claims) * 100. None if factual_claims == 0."
    )
    unsupported_rate: Optional[float] = Field(
        default=None,
        description="Percentage of factual claims unmentioned: (unsupported / factual_claims) * 100. None if factual_claims == 0."
    )
    unverifiable_rate: Optional[float] = Field(
        default=None,
        description="Percentage of factual claims unverifiable: (unverifiable / factual_claims) * 100. None if factual_claims == 0."
    )
    uncertainty_rate: Optional[float] = Field(
        default=None,
        description="Combined uncertainty percentage: ((unsupported + unverifiable) / factual_claims) * 100. None if factual_claims == 0."
    )
    denominator_description: str = Field(
        ...,
        description="Detailed explanation of the metric denominator."
    )


class VerificationReport(BaseModel):
    """
    Complete analysis container for an LLM response evaluated against reference material.
    """
    document_name: Optional[str] = Field(default=None, description="Name of reference document.")
    total_claims_processed: int = Field(..., description="Total number of claims processed.")
    claims: List[VerifiedClaim] = Field(default_factory=list, description="All verified claims.")
    metrics: VerificationMetrics = Field(..., description="Calculated factuality metrics.")
    execution_time_seconds: float = Field(..., description="Total verification duration in seconds.")
