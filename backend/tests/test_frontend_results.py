"""
Unit tests for VeriLLM Frontend Results View (Step 10.4 & Step 10.5).
Tests classification normalization, claim filtering logic, evidence and provenance presentation,
calibrated probability rendering, visual analytics charts, metric guide, and empty/active states.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import altair as alt
import pytest

# Ensure project root and backend are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml_engine.claim_verifier.models import (
    ModelVerificationScore,
    VerificationCategory,
    VerifiedClaim,
)
from ml_engine.evidence_retriever.models import DocumentChunk, RetrievedPassage
from ml_engine.pipeline import PipelineResult
from ml_engine.scorer.models import (
    FactualityIndices,
    ResponseCounts,
    ResponseRates,
    ResponseScore,
)
from frontend.views.results import (
    build_calibrated_probabilities_chart,
    build_classification_distribution_chart,
    filter_claims_by_category,
    get_classification_str,
    render_claim_card,
    render_metric_interpretation_guide,
    render_results_view,
)


def _make_dummy_claim(
    claim_id: str,
    text: str,
    classification: VerificationCategory,
    is_factual: bool = True,
    primary_passage: Optional[RetrievedPassage] = None,
    all_evidence: Optional[list] = None,
    reasoning_trace: str = "Test trace",
    decision_rule: str = "TEST_RULE",
) -> VerifiedClaim:
    """Helper to construct a typed VerifiedClaim for testing."""
    return VerifiedClaim(
        claim_id=claim_id,
        claim_text=text,
        source_sentence=text,
        start_char=0,
        end_char=len(text),
        is_factual=is_factual,
        classification=classification,
        scores=ModelVerificationScore(),
        primary_evidence=primary_passage,
        all_evidence=all_evidence or ([] if not primary_passage else [primary_passage]),
        decision_rule=decision_rule,
        reasoning_trace=reasoning_trace,
    )


def _make_dummy_passage(
    chunk_id: str = "chunk_001",
    document_name: str = "doc.pdf",
    page_number: Optional[int] = 1,
    text: str = "Evidence text snippet.",
    similarity: float = 0.85,
) -> RetrievedPassage:
    """Helper to construct a typed RetrievedPassage for testing."""
    chunk = DocumentChunk(
        chunk_id=chunk_id,
        document_name=document_name,
        page_number=page_number,
        chunk_index=0,
        text=text,
        start_char=0,
        end_char=len(text),
        word_count=len(text.split()),
    )
    return RetrievedPassage(
        chunk=chunk,
        similarity_score=similarity,
        rank=1,
    )


class TestClassificationNormalization:
    """Tests safely extracting normalized category strings from VerifiedClaim."""

    def test_supported_category(self):
        claim = _make_dummy_claim("c1", "Fact A", VerificationCategory.SUPPORTED)
        assert get_classification_str(claim) == "SUPPORTED"

    def test_contradicted_category(self):
        claim = _make_dummy_claim("c2", "Fact B", VerificationCategory.CONTRADICTED)
        assert get_classification_str(claim) == "CONTRADICTED"

    def test_unsupported_category(self):
        claim = _make_dummy_claim("c3", "Fact C", VerificationCategory.UNSUPPORTED)
        assert get_classification_str(claim) == "UNSUPPORTED"

    def test_unverifiable_category(self):
        claim = _make_dummy_claim("c4", "Fact D", VerificationCategory.UNVERIFIABLE)
        assert get_classification_str(claim) == "UNVERIFIABLE"

    def test_non_factual_category(self):
        claim = _make_dummy_claim("c5", "Hello world!", VerificationCategory.NON_FACTUAL, is_factual=False)
        assert get_classification_str(claim) == "NON_FACTUAL"

    def test_is_factual_false_forces_non_factual(self):
        claim = _make_dummy_claim("c6", "In my opinion it is good.", VerificationCategory.SUPPORTED, is_factual=False)
        assert get_classification_str(claim) == "NON_FACTUAL"


class TestClaimFilteringLogic:
    """Tests filtering claim lists by user selection."""

    @pytest.fixture
    def sample_claims(self):
        return [
            _make_dummy_claim("c1", "Claim 1", VerificationCategory.SUPPORTED),
            _make_dummy_claim("c2", "Claim 2", VerificationCategory.SUPPORTED),
            _make_dummy_claim("c3", "Claim 3", VerificationCategory.CONTRADICTED),
            _make_dummy_claim("c4", "Claim 4", VerificationCategory.UNSUPPORTED),
            _make_dummy_claim("c5", "Claim 5", VerificationCategory.UNVERIFIABLE),
            _make_dummy_claim("c6", "Claim 6", VerificationCategory.NON_FACTUAL, is_factual=False),
        ]

    def test_filter_all(self, sample_claims):
        filtered = filter_claims_by_category(sample_claims, "All (6)")
        assert len(filtered) == 6

    def test_filter_supported(self, sample_claims):
        filtered = filter_claims_by_category(sample_claims, "Supported (2)")
        assert len(filtered) == 2
        assert all(get_classification_str(c) == "SUPPORTED" for c in filtered)

    def test_filter_contradicted(self, sample_claims):
        filtered = filter_claims_by_category(sample_claims, "Contradicted (1)")
        assert len(filtered) == 1
        assert filtered[0].claim_id == "c3"

    def test_filter_unsupported(self, sample_claims):
        filtered = filter_claims_by_category(sample_claims, "Unsupported (1)")
        assert len(filtered) == 1
        assert filtered[0].claim_id == "c4"

    def test_filter_unverifiable(self, sample_claims):
        filtered = filter_claims_by_category(sample_claims, "Unverifiable (1)")
        assert len(filtered) == 1
        assert filtered[0].claim_id == "c5"

    def test_filter_non_factual(self, sample_claims):
        filtered = filter_claims_by_category(sample_claims, "Non-Factual (1)")
        assert len(filtered) == 1
        assert filtered[0].claim_id == "c6"

    def test_filter_no_matches_returns_empty(self, sample_claims):
        supported_only = [sample_claims[0], sample_claims[1]]
        filtered = filter_claims_by_category(supported_only, "Contradicted (0)")
        assert len(filtered) == 0

    def test_filter_empty_list_returns_empty(self):
        assert filter_claims_by_category([], "All (0)") == []


class TestClaimCardRendering:
    """Tests claim card HTML generation across evidence and provenance scenarios."""

    @patch("streamlit.markdown")
    @patch("streamlit.expander")
    @patch("streamlit.caption")
    def test_render_supported_claim_with_evidence(self, mock_caption, mock_expander, mock_markdown):
        passage = _make_dummy_passage(
            chunk_id="chunk_apollo_01",
            document_name="history.pdf",
            page_number=3,
            text="Apollo 11 landed on July 20, 1969.",
            similarity=0.912,
        )
        claim = _make_dummy_claim(
            claim_id="c_01",
            text="Apollo 11 landed in July 1969.",
            classification=VerificationCategory.SUPPORTED,
            primary_passage=passage,
            reasoning_trace="[Calibrated: Support=0.95] Strong entailment.",
        )
        probs = {"Support": 0.95, "Contradiction": 0.02, "Unsupported": 0.03}

        render_claim_card(claim, index=0, calibrated_probs=probs)

        calls_str = " ".join(str(c) for c in mock_markdown.call_args_list)
        assert "SUPPORTED" in calls_str
        assert "history.pdf" in calls_str
        assert "Page 3" in calls_str
        assert "chunk_apollo_01" in calls_str
        assert "0.912" in calls_str
        assert "Apollo 11 landed on July 20, 1969." in calls_str
        assert "95.0%" in calls_str

    @patch("streamlit.markdown")
    @patch("streamlit.expander")
    @patch("streamlit.caption")
    def test_render_unverifiable_claim_missing_evidence(self, mock_caption, mock_expander, mock_markdown):
        claim = _make_dummy_claim(
            claim_id="c_02",
            text="Alien life was detected on Europa.",
            classification=VerificationCategory.UNVERIFIABLE,
            primary_passage=None,
            reasoning_trace="No evidence above similarity floor 0.20.",
        )

        render_claim_card(claim, index=1, calibrated_probs=None)

        calls_str = " ".join(str(c) for c in mock_markdown.call_args_list)
        assert "UNVERIFIABLE" in calls_str
        assert "Safe Abstention" in calls_str
        assert "unverified, not proven false" in calls_str

    @patch("streamlit.markdown")
    @patch("streamlit.expander")
    @patch("streamlit.caption")
    def test_render_contradicted_claim_with_evidence(self, mock_caption, mock_expander, mock_markdown):
        passage = _make_dummy_passage(
            chunk_id="chunk_chem_01",
            document_name="elements.txt",
            page_number=None,
            text="Water is composed of hydrogen and oxygen.",
            similarity=0.885,
        )
        claim = _make_dummy_claim(
            claim_id="c_03",
            text="Water contains nitrogen and helium.",
            classification=VerificationCategory.CONTRADICTED,
            primary_passage=passage,
            reasoning_trace="[Calibrated: Contradiction=0.91] Direct refutation.",
        )

        render_claim_card(claim, index=2, calibrated_probs={"Contradiction": 0.91, "Support": 0.01, "Unsupported": 0.08})

        calls_str = " ".join(str(c) for c in mock_markdown.call_args_list)
        assert "CONTRADICTED" in calls_str
        assert "Directly Refuting" in calls_str
        assert "elements.txt" in calls_str
        assert "91.0%" in calls_str


class TestVisualAnalyticsCharts:
    """Tests Altair chart construction for distribution and calibrated probabilities."""

    def test_build_classification_distribution_chart(self):
        mock_result = MagicMock(spec=PipelineResult)
        mock_score = MagicMock(spec=ResponseScore)
        mock_score.counts = ResponseCounts(
            total_claims=4,
            factual_claims=4,
            non_factual_claims=0,
            supported_count=2,
            contradicted_count=1,
            unsupported_count=1,
            unverifiable_count=0,
            non_factual_count=0,
        )
        mock_score.rates = ResponseRates(
            evidence_support_rate=50.0,
            confirmed_contradiction_rate=25.0,
            unsupported_rate=25.0,
            unverifiable_rate=0.0,
            combined_uncertainty_rate=25.0,
        )
        mock_result.score = mock_score

        chart = build_classification_distribution_chart(mock_result)
        assert isinstance(chart, alt.Chart)

    def test_build_calibrated_probabilities_chart_success(self):
        mock_result = MagicMock(spec=PipelineResult)
        c1 = _make_dummy_claim("c1", "Claim 1", VerificationCategory.SUPPORTED)
        c2 = _make_dummy_claim("c2", "Claim 2", VerificationCategory.CONTRADICTED)
        mock_result.claims = [c1, c2]
        mock_result.metadata = {
            "calibrated_probabilities": {
                "c1": {"Support": 0.95, "Contradiction": 0.01, "Unsupported": 0.04},
                "c2": {"Support": 0.02, "Contradiction": 0.88, "Unsupported": 0.10},
            }
        }

        chart = build_calibrated_probabilities_chart(mock_result)
        assert isinstance(chart, alt.Chart)

    def test_build_calibrated_probabilities_chart_empty(self):
        mock_result = MagicMock(spec=PipelineResult)
        mock_result.claims = []
        mock_result.metadata = {}

        chart = build_calibrated_probabilities_chart(mock_result)
        assert chart is None

    @patch("streamlit.expander")
    @patch("streamlit.markdown")
    def test_render_metric_interpretation_guide(self, mock_markdown, mock_expander):
        render_metric_interpretation_guide()
        mock_expander.assert_called_once()
        calls_str = " ".join(str(c) for c in mock_markdown.call_args_list)
        assert "Grounded Factuality Index" in calls_str
        assert "Contradiction-Penalized Factuality" in calls_str
        assert "UNVERIFIABLE (Safe Abstention)" in calls_str
        assert "not a confirmed hallucination" in calls_str


class TestResultsViewEmptyAndActiveStates:
    """Tests the top-level render_results_view routing and visual dashboard integration."""

    @patch("streamlit.session_state", {})
    @patch("streamlit.markdown")
    def test_render_empty_state_when_no_result(self, mock_markdown):
        render_results_view()
        calls_str = " ".join(str(c) for c in mock_markdown.call_args_list)
        assert "No Verification Results Available Yet" in calls_str

    @patch("streamlit.session_state")
    @patch("streamlit.markdown")
    @patch("streamlit.columns")
    @patch("streamlit.radio")
    @patch("streamlit.expander")
    @patch("streamlit.altair_chart")
    def test_render_active_results_state(self, mock_chart, mock_expander, mock_radio, mock_columns, mock_markdown, mock_session):
        mock_result = MagicMock(spec=PipelineResult)
        mock_result.document_name = "test_doc.pdf"
        mock_result.regime = "retrieval"
        mock_result.execution_time_seconds = 4.25
        mock_result.warnings = []
        mock_result.metadata = {
            "calibrated_probabilities": {
                "c1": {"Support": 0.95, "Contradiction": 0.02, "Unsupported": 0.03},
                "c2": {"Support": 0.92, "Contradiction": 0.03, "Unsupported": 0.05},
            }
        }
        mock_result.to_dict.return_value = {}

        mock_score = MagicMock(spec=ResponseScore)
        mock_score.counts = ResponseCounts(
            total_claims=2,
            factual_claims=2,
            non_factual_claims=0,
            supported_count=2,
            contradicted_count=0,
            unsupported_count=0,
            unverifiable_count=0,
            non_factual_count=0,
        )
        mock_score.rates = ResponseRates(
            evidence_support_rate=100.0,
            confirmed_contradiction_rate=0.0,
            unsupported_rate=0.0,
            unverifiable_rate=0.0,
            combined_uncertainty_rate=0.0,
        )
        mock_score.indices = FactualityIndices(
            grounded_factuality_index=100.0,
            contradiction_penalized_factuality=100.0,
            hallucination_severity_index=0.0,
        )
        mock_result.score = mock_score

        c1 = _make_dummy_claim("c1", "Claim 1", VerificationCategory.SUPPORTED)
        c2 = _make_dummy_claim("c2", "Claim 2", VerificationCategory.SUPPORTED)
        mock_result.claims = [c1, c2]

        mock_session.get.side_effect = lambda k, default=None: mock_result if k == "verification_result" else default
        mock_columns.side_effect = lambda n, **kwargs: [MagicMock() for _ in range(n if isinstance(n, int) else len(n))]
        mock_radio.return_value = "All (2)"

        render_results_view()

        calls_str = " ".join(str(c) for c in mock_markdown.call_args_list)
        assert "Verification Summary" in calls_str
        assert "Visual Verification Analytics" in calls_str
        assert "Detailed Claim-Level Inspector" in calls_str
        assert mock_chart.called
