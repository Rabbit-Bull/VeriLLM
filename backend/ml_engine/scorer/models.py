"""
VeriLLM Scoring and Quantification Engine Data Models.
Defines strongly typed Pydantic schemas for response-level factuality scores,
dataset-level micro/macro aggregations, model comparison breakdowns, and scoring configurations.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ScoringConfig(BaseModel):
    """
    Configuration options for the factuality scoring engine.
    """
    contradiction_penalty_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight lambda applied to contradicted claims in the Contradiction-Penalized Factuality Score (default: 1.0)."
    )
    deduplicate_claims: bool = Field(
        default=False,
        description="If True, calculate rates after deduplicating exact/normalized claim propositions."
    )


class ClaimSummary(BaseModel):
    """
    Traceable claim summary preserving verification metadata for API and UI consumers.
    """
    claim_id: str = Field(..., description="Unique claim identifier.")
    claim_text: str = Field(..., description="The atomic factual proposition.")
    is_factual: bool = Field(..., description="Whether proposition asserts an empirical fact.")
    classification: str = Field(..., description="Assigned category (SUPPORTED, CONTRADICTED, UNSUPPORTED, UNVERIFIABLE, NON_FACTUAL).")
    citation: Optional[str] = Field(default=None, description="Document name, page, and chunk provenance.")
    minicheck_probability: Optional[float] = Field(default=None, description="MiniCheck support probability.")
    nli_entailment_prob: Optional[float] = Field(default=None, description="NLI entailment probability.")
    nli_contradiction_prob: Optional[float] = Field(default=None, description="NLI contradiction probability.")
    nli_neutral_prob: Optional[float] = Field(default=None, description="NLI neutral probability.")
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
    reasoning_trace: str = Field(..., description="Human-readable decision explanation.")


class ResponseCounts(BaseModel):
    """
    Partitioned counts of claims extracted from a single response.
    """
    total_claims: int = Field(..., description="Total extracted propositions (including non-factual).")
    factual_claims: int = Field(..., description="Count of empirical factual propositions evaluated (N).")
    non_factual_claims: int = Field(..., description="Count of opinions, greetings, and conversational remarks.")
    supported_count: int = Field(..., description="Count of empirical factual claims verified as SUPPORTED.")
    contradicted_count: int = Field(..., description="Count of empirical factual claims verified as CONTRADICTED.")
    unsupported_count: int = Field(..., description="Count of empirical factual claims verified as UNSUPPORTED.")
    unverifiable_count: int = Field(..., description="Count of empirical factual claims verified as UNVERIFIABLE.")
    non_factual_count: int = Field(..., description="Count of propositions categorized as NON_FACTUAL.")
    duplicate_claims_count: int = Field(default=0, description="Count of duplicate propositions detected in this response.")


class ResponseRates(BaseModel):
    """
    Mutually exclusive and collectively exhaustive rate metrics over empirical factual claims (N).
    All values are in range [0.0, 100.0] or None when N = 0.
    """
    evidence_support_rate: Optional[float] = Field(
        default=None,
        description="Percentage of factual claims directly supported by evidence: (supported / N) * 100."
    )
    confirmed_contradiction_rate: Optional[float] = Field(
        default=None,
        description="Percentage of factual claims directly refuted by evidence: (contradicted / N) * 100."
    )
    unsupported_rate: Optional[float] = Field(
        default=None,
        description="Percentage of factual claims unmentioned in evidence: (unsupported / N) * 100."
    )
    unverifiable_rate: Optional[float] = Field(
        default=None,
        description="Percentage of factual claims unverifiable due to missing/conflicting evidence: (unverifiable / N) * 100."
    )
    combined_uncertainty_rate: Optional[float] = Field(
        default=None,
        description="Combined epistemic uncertainty rate: ((unsupported + unverifiable) / N) * 100."
    )


class FactualityIndices(BaseModel):
    """
    Overall factuality indices with documented mathematical formulations.
    Not calibrated Bayesian probabilities of universal truth; measures document-grounded consistency.
    """
    grounded_factuality_index: Optional[float] = Field(
        default=None,
        description="GFI: Percentage of factual claims confirmed by evidence (equivalent to Evidence Support Rate, in [0.0, 100.0])."
    )
    contradiction_penalized_factuality: Optional[float] = Field(
        default=None,
        description="CPFS: max(0.0, (supported - lambda * contradicted) / N) * 100, penalizing active hallucinations."
    )
    hallucination_severity_index: Optional[float] = Field(
        default=None,
        description="HSI: Percentage of claims actively contradicted (equivalent to Confirmed Contradiction Rate, in [0.0, 100.0])."
    )


class ResponseScore(BaseModel):
    """
    Comprehensive scoring and quantification container for a single evaluated LLM response.
    """
    response_id: str = Field(..., description="Unique identifier for the evaluated response.")
    model_id: Optional[str] = Field(default=None, description="Identifier of the LLM that generated the response (e.g. gpt-4o, llama-3-8b).")
    document_name: Optional[str] = Field(default=None, description="Reference document name.")
    counts: ResponseCounts = Field(..., description="Raw proposition and category counts.")
    rates: ResponseRates = Field(..., description="Calculated rate metrics over factual propositions.")
    indices: FactualityIndices = Field(..., description="Overall factuality and hallucination severity indices.")
    has_factual_claims: bool = Field(..., description="True if factual_claims > 0; False if response contained zero testable facts.")
    is_empty_response: bool = Field(..., description="True if response text contained zero extracted propositions.")
    contains_duplicates: bool = Field(default=False, description="True if identical or duplicate propositions were detected.")
    mathematical_invariants_verified: bool = Field(
        ...,
        description="True if Support Rate + Contradiction Rate + Unsupported Rate + Unverifiable Rate == 100.00% (within 0.01 tolerance) when N > 0."
    )
    denominator_description: str = Field(..., description="Detailed explanation of the denominator used for calculations.")
    claims: List[ClaimSummary] = Field(default_factory=list, description="Traceable summaries of all evaluated propositions.")
    execution_time_seconds: Optional[float] = Field(default=None, description="Verification latency in seconds.")


class ModelAggregateScore(BaseModel):
    """
    Aggregated performance metrics for a specific generative model across multiple responses.
    """
    model_id: str = Field(..., description="Identifier of the generative model.")
    total_responses: int = Field(..., description="Total responses evaluated for this model.")
    responses_with_factual_claims: int = Field(..., description="Number of responses containing at least one factual proposition.")
    total_factual_claims: int = Field(..., description="Total empirical factual propositions evaluated across all responses (N_pooled).")
    total_non_factual_claims: int = Field(..., description="Total non-factual claims excluded from factuality rates.")
    supported_count: int = Field(..., description="Total factual claims verified as SUPPORTED.")
    contradicted_count: int = Field(..., description="Total factual claims verified as CONTRADICTED.")
    unsupported_count: int = Field(..., description="Total factual claims verified as UNSUPPORTED.")
    unverifiable_count: int = Field(..., description="Total factual claims verified as UNVERIFIABLE.")
    non_factual_count: int = Field(..., description="Total claims categorized as NON_FACTUAL.")
    
    # Micro-pooled metrics (claim-weighted, mathematically consistent across responses)
    micro_evidence_support_rate: Optional[float] = Field(default=None, description="Pooled Evidence Support Rate (supported_sum / N_sum * 100).")
    micro_confirmed_contradiction_rate: Optional[float] = Field(default=None, description="Pooled Contradiction Rate (contradicted_sum / N_sum * 100).")
    micro_unsupported_rate: Optional[float] = Field(default=None, description="Pooled Unsupported Rate (unsupported_sum / N_sum * 100).")
    micro_unverifiable_rate: Optional[float] = Field(default=None, description="Pooled Unverifiable Rate (unverifiable_sum / N_sum * 100).")
    micro_combined_uncertainty_rate: Optional[float] = Field(default=None, description="Pooled Uncertainty Rate ((unsupported_sum + unverifiable_sum) / N_sum * 100).")
    micro_grounded_factuality_index: Optional[float] = Field(default=None, description="Pooled Grounded Factuality Index.")
    micro_penalized_factuality_score: Optional[float] = Field(default=None, description="Pooled Contradiction-Penalized Factuality Score.")
    
    # Macro-averaged metrics (unweighted mean of response-level rates)
    macro_evidence_support_rate: Optional[float] = Field(default=None, description="Unweighted average of response support rates.")
    macro_confirmed_contradiction_rate: Optional[float] = Field(default=None, description="Unweighted average of response contradiction rates.")
    macro_unsupported_rate: Optional[float] = Field(default=None, description="Unweighted average of response unsupported rates.")
    macro_unverifiable_rate: Optional[float] = Field(default=None, description="Unweighted average of response unverifiable rates.")
    macro_combined_uncertainty_rate: Optional[float] = Field(default=None, description="Unweighted average of response uncertainty rates.")


class DatasetScore(BaseModel):
    """
    Dataset-level scoring aggregation container.
    Provides pooled (micro) statistics, unweighted response-level (macro) statistics,
    and cross-model comparisons.
    """
    dataset_name: Optional[str] = Field(default=None, description="Name or identifier of the evaluated dataset.")
    total_responses: int = Field(..., description="Total evaluated responses in the dataset.")
    responses_with_factual_claims: int = Field(..., description="Responses containing at least one testable factual claim.")
    responses_zero_factual_claims: int = Field(..., description="Responses with zero factual claims (opinions, greetings only).")
    empty_responses: int = Field(..., description="Responses with zero extracted propositions.")
    total_extracted_claims: int = Field(..., description="Total propositions extracted across all responses.")
    total_factual_claims: int = Field(..., description="Total empirical factual propositions evaluated across all responses (N_dataset).")
    total_non_factual_claims: int = Field(..., description="Total non-factual claims excluded from factuality denominators.")
    
    # Pooled proposition counts
    pooled_supported_count: int = Field(..., description="Total supported claims across the dataset.")
    pooled_contradicted_count: int = Field(..., description="Total contradicted claims across the dataset.")
    pooled_unsupported_count: int = Field(..., description="Total unsupported claims across the dataset.")
    pooled_unverifiable_count: int = Field(..., description="Total unverifiable claims across the dataset.")
    pooled_non_factual_count: int = Field(..., description="Total non-factual claims across the dataset.")
    
    # Micro-pooled rates (Primary: prevents Simpson's paradox)
    micro_evidence_support_rate: Optional[float] = Field(
        default=None,
        description="Dataset Micro-Pooled Support Rate: (sum(supported) / sum(N)) * 100."
    )
    micro_confirmed_contradiction_rate: Optional[float] = Field(
        default=None,
        description="Dataset Micro-Pooled Contradiction Rate: (sum(contradicted) / sum(N)) * 100."
    )
    micro_unsupported_rate: Optional[float] = Field(
        default=None,
        description="Dataset Micro-Pooled Unsupported Rate: (sum(unsupported) / sum(N)) * 100."
    )
    micro_unverifiable_rate: Optional[float] = Field(
        default=None,
        description="Dataset Micro-Pooled Unverifiable Rate: (sum(unverifiable) / sum(N)) * 100."
    )
    micro_combined_uncertainty_rate: Optional[float] = Field(
        default=None,
        description="Dataset Micro-Pooled Uncertainty Rate: ((sum(unsupported) + sum(unverifiable)) / sum(N)) * 100."
    )
    micro_grounded_factuality_index: Optional[float] = Field(
        default=None,
        description="Dataset Grounded Factuality Index (Micro-Pooled)."
    )
    micro_penalized_factuality_score: Optional[float] = Field(
        default=None,
        description="Dataset Contradiction-Penalized Factuality Score (Micro-Pooled)."
    )
    
    # Macro-averaged rates (Secondary: response-level unweighted average)
    macro_evidence_support_rate: Optional[float] = Field(
        default=None,
        description="Dataset Macro-Averaged Support Rate (mean of response rates across responses with N > 0)."
    )
    macro_confirmed_contradiction_rate: Optional[float] = Field(
        default=None,
        description="Dataset Macro-Averaged Contradiction Rate (mean of response rates across responses with N > 0)."
    )
    macro_unsupported_rate: Optional[float] = Field(
        default=None,
        description="Dataset Macro-Averaged Unsupported Rate (mean of response rates across responses with N > 0)."
    )
    macro_unverifiable_rate: Optional[float] = Field(
        default=None,
        description="Dataset Macro-Averaged Unverifiable Rate (mean of response rates across responses with N > 0)."
    )
    macro_combined_uncertainty_rate: Optional[float] = Field(
        default=None,
        description="Dataset Macro-Averaged Uncertainty Rate (mean of response rates across responses with N > 0)."
    )
    
    # Grouped breakdowns
    models_breakdown: Dict[str, ModelAggregateScore] = Field(
        default_factory=dict,
        description="Per-model aggregated scores when model identifiers are provided."
    )
    mathematical_invariants_verified: bool = Field(
        ...,
        description="True if micro_support + micro_contradicted + micro_unsupported + micro_unverifiable == 100.00%."
    )
    aggregation_notes: List[str] = Field(
        default_factory=list,
        description="Methodological explanations, caveats, and denominator documentation."
    )

