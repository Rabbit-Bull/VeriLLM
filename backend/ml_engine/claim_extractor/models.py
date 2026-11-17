"""
VeriLLM Claim Extraction Data Models.
Defines strongly-typed Pydantic schemas for atomic claims and extraction results.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ExtractedClaim(BaseModel):
    """
    Represents an atomic, self-contained proposition extracted from an LLM response.
    Preserves exact traceability back to the original response text.
    """
    claim_id: str = Field(
        ..., 
        description="Unique identifier for the claim (e.g., 'claim_001')."
    )
    text: str = Field(
        ..., 
        description="The standalone, atomic factual proposition."
    )
    source_sentence: str = Field(
        ..., 
        description="The original sentence from which this claim was extracted."
    )
    start_char: int = Field(
        ..., 
        description="Character start index of the source sentence in the raw response."
    )
    end_char: int = Field(
        ..., 
        description="Character end index of the source sentence in the raw response."
    )
    claim_start_char: Optional[int] = Field(
        default=None,
        description="Exact start index of the specific claim/clause in raw text, where localized."
    )
    claim_end_char: Optional[int] = Field(
        default=None,
        description="Exact end index of the specific claim/clause in raw text, where localized."
    )
    offset_type: str = Field(
        default="EXACT",
        description="Fidelity of claim offset: EXACT (matches raw span), CLAUSE_SPAN (sub-clause in sentence), EMBEDDED_CLAUSE (unwrapped from rhetorical question), PRONOUN_RESOLVED (subject resolved), or SENTENCE_LEVEL."
    )
    is_factual: bool = Field(
        default=True, 
        description="True if the statement asserts an empirical fact; False if opinion/question/conversational."
    )
    claim_type: str = Field(
        default="FACTUAL_PROPOSITION",
        description="Type classification: FACTUAL_PROPOSITION, NUMERICAL_ASSERTION, SUBJECTIVE_OPINION, QUESTION, CONVERSATIONAL."
    )
    has_negation: bool = Field(
        default=False, 
        description="Whether the claim contains explicit negation tokens (not, never, no, neither)."
    )
    has_numbers: bool = Field(
        default=False, 
        description="Whether the claim contains numerical values, dates, percentages, or measurements."
    )
    named_entities: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of detected named entities with 'text' and 'label' (e.g. PERSON, DATE, GPE)."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional extraction metadata (e.g. resolved coreference pronouns, clause origin)."
    )


class ExtractionResult(BaseModel):
    """
    Container for the full decomposition output of an LLM response.
    """
    raw_text: str = Field(..., description="The original raw input text.")
    claims: List[ExtractedClaim] = Field(
        default_factory=list, 
        description="All extracted propositions in presentation order."
    )
    total_sentences: int = Field(default=0, description="Total sentences detected in raw text.")
    factual_claims_count: int = Field(default=0, description="Number of actionable factual claims.")
    non_factual_count: int = Field(default=0, description="Number of non-factual or subjective sentences.")

    @property
    def factual_claims(self) -> List[ExtractedClaim]:
        """Convenience accessor returning only factual propositions ready for verification."""
        return [c for c in self.claims if c.is_factual]

