"""
VeriLLM Overview / Landing Page View.
Presents the project foundation, architectural overview, verification workflow,
dual verification regimes, and navigation action to begin analysis.
"""

from typing import Callable, Optional
import streamlit as st
from frontend.components.header import render_page_header


def render_overview_view(on_navigate_verify: Optional[Callable[[], None]] = None) -> None:
    """Renders the comprehensive VeriLLM landing page."""
    render_page_header(
        title="Document-Grounded Hallucination Detection",
        subtitle=(
            "An empirical verification engine for LLM-generated responses. Evaluates atomic factual claims "
            "against source documents using fine-tuned DeBERTa-v3 cross-encoders, dense passage retrieval, "
            "and mathematically calibrated decision rules."
        ),
        badge="VeriLLM System • Local Offline Edition",
    )

    # Primary Action CTA
    col_cta, col_meta = st.columns([1, 2])
    with col_cta:
        if st.button("Start Verification →", type="primary", use_container_width=True):
            if on_navigate_verify:
                on_navigate_verify()
            elif hasattr(st, "switch_page"):
                try:
                    st.session_state["nav_selection"] = "Verify Response"
                    st.rerun()
                except Exception:
                    pass

    with col_meta:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 12px; height: 100%; padding-left: 8px;">
                <span class="status-pill">
                    <span class="status-dot-active"></span>
                    <span>Fine-tuned DeBERTa-v3 Backbone</span>
                </span>
                <span class="status-pill">
                    <span class="status-dot-active"></span>
                    <span>Stage 1 Frozen C2 Calibration</span>
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # Section 1: Core Principles
    # -------------------------------------------------------------
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(
            """
            <div class="verillm-card">
                <div class="verillm-card-header">
                    <span>Document-Grounded Verification</span>
                    <span class="status-pill" style="font-size: 0.72rem;">Principle 01</span>
                </div>
                <p class="verillm-card-body">
                    Traditional evaluation relies on subjective heuristics or ungrounded model-as-a-judge prompting. 
                    VeriLLM treats hallucination detection as an <strong>empirical, document-grounded NLI task</strong>. 
                    Every claim is validated strictly against reference text. Statements not mentioned in the source 
                    are identified as unverified rather than assumed to be true.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="verillm-card">
                <div class="verillm-card-header">
                    <span>Calibrated Posteriors & Abstention</span>
                    <span class="status-pill" style="font-size: 0.72rem;">Principle 02</span>
                </div>
                <p class="verillm-card-body">
                    Raw neural network softmax outputs are notoriously overconfident under distribution shift. 
                    VeriLLM applies <strong>prior-bias shifted temperature calibration (C2)</strong> fitted during research 
                    validation. The system enforces strict epistemic abstention: claims with insufficient or irrelevant evidence 
                    are categorized as <em>UNVERIFIABLE</em> rather than false hallucinations.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------
    # Section 2: Verification Workflow
    # -------------------------------------------------------------
    st.markdown("### End-to-End Verification Pipeline")
    st.markdown(
        "<p style='color: #9CA3AF; font-size: 0.95rem; margin-top: -8px; margin-bottom: 20px;'>"
        "The automated pipeline executes four modular processing stages without external cloud dependencies."
        "</p>",
        unsafe_allow_html=True,
    )

    step_cols = st.columns(4, gap="medium")

    with step_cols[0]:
        st.markdown(
            """
            <div class="verillm-card" style="height: 100%;">
                <div class="step-item">
                    <span class="step-number">1</span>
                    <div class="step-text">
                        <h4>Document Ingestion</h4>
                        <p>PDF or plain text is parsed, cleaned, and partitioned into page-preserved passages via sliding window.</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with step_cols[1]:
        st.markdown(
            """
            <div class="verillm-card" style="height: 100%;">
                <div class="step-item">
                    <span class="step-number">2</span>
                    <div class="step-text">
                        <h4>Claim Extraction</h4>
                        <p>Candidate LLM response is decomposed into standalone atomic propositions using coreference resolution.</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with step_cols[2]:
        st.markdown(
            """
            <div class="verillm-card" style="height: 100%;">
                <div class="step-item">
                    <span class="step-number">3</span>
                    <div class="step-text">
                        <h4>Evidence Grounding</h4>
                        <p>Top candidate passages are retrieved via dense vector similarity (MiniLM) or paired directly as full context.</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with step_cols[3]:
        st.markdown(
            """
            <div class="verillm-card" style="height: 100%;">
                <div class="step-item">
                    <span class="step-number">4</span>
                    <div class="step-text">
                        <h4>Calibrated Scoring</h4>
                        <p>Fine-tuned DeBERTa evaluates NLI pairs under frozen C2 thresholds to calculate GFI, CPFS, and HSI indices.</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------
    # Section 3: Dual Verification Regimes
    # -------------------------------------------------------------
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.markdown("### Operational Verification Regimes")
    st.markdown(
        "<p style='color: #9CA3AF; font-size: 0.95rem; margin-top: -8px; margin-bottom: 20px;'>"
        "VeriLLM provides two mathematically distinct regimes optimized for different document lengths and contexts."
        "</p>",
        unsafe_allow_html=True,
    )

    reg_col1, reg_col2 = st.columns(2, gap="large")

    with reg_col1:
        st.markdown(
            """
            <div class="verillm-card">
                <div class="verillm-card-header">
                    <span>Direct Context Regime</span>
                    <span class="status-pill" style="color: #60A5FA;">Short Documents (&lt;350 words)</span>
                </div>
                <p class="verillm-card-body" style="margin-bottom: 12px;">
                    Pairs each extracted proposition directly with the complete cleaned reference document. 
                    Offers exhaustive cross-attention across the full context window.
                </p>
                <div style="background-color: #0B0F17; border-radius: 6px; padding: 10px 14px; font-size: 0.82rem; color: #9CA3AF; border: 1px solid #1F2937;">
                    <div>• <strong>Calibration:</strong> Frozen C2 (T = 1.769592, b = [-0.67, +0.71, -0.04])</div>
                    <div>• <strong>Thresholds:</strong> &tau;<sub>contra</sub> = 0.35, &tau;<sub>sup</sub> = 0.70</div>
                    <div>• <strong>Boundary:</strong> Documents &gt;512 tokens are automatically flagged for truncation.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with reg_col2:
        st.markdown(
            """
            <div class="verillm-card">
                <div class="verillm-card-header">
                    <span>Retrieval-Augmented Regime</span>
                    <span class="status-pill" style="color: #34D399;">Multi-Page / Long Documents</span>
                </div>
                <p class="verillm-card-body" style="margin-bottom: 12px;">
                    Indexes document passages into dense vector space and retrieves top-k relevant candidate passages per claim. 
                    Scales to multi-page PDFs without context-window truncation.
                </p>
                <div style="background-color: #0B0F17; border-radius: 6px; padding: 10px 14px; font-size: 0.82rem; color: #9CA3AF; border: 1px solid #1F2937;">
                    <div>• <strong>Calibration:</strong> Frozen C2 (T = 2.504110, b = [-0.73, +1.23, -0.50])</div>
                    <div>• <strong>Thresholds:</strong> &tau;<sub>contra</sub> = 0.30, &tau;<sub>sup</sub> = 0.70</div>
                    <div>• <strong>Evidence Floor:</strong> Similarity &lt; 0.20 strictly abstains to <em>UNVERIFIABLE</em>.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------
    # Section 4: Local Execution & Privacy Banner
    # -------------------------------------------------------------
    st.markdown(
        """
        <div style="background-color: #0E1420; border: 1px solid #1F2937; border-radius: 8px; padding: 16px 20px; margin-top: 10px; display: flex; align-items: center; justify-content: space-between;">
            <div>
                <span style="font-weight: 600; color: #F3F4F6; font-size: 0.92rem;">Local & Offline Execution Guarantee</span>
                <p style="margin: 4px 0 0 0; color: #6B7280; font-size: 0.85rem;">
                    All DeBERTa inference, sentence embeddings, and PDF parsing execute locally on CPU. No data is sent to external APIs or cloud services.
                </p>
            </div>
            <div class="status-pill">
                <span class="status-dot-active"></span>
                <span>Ready</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

