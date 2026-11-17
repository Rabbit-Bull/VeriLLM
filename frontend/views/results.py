"""
VeriLLM Results View.
Renders verification results and visual analytics from real VeriLLMPipeline execution,
including response-level factuality KPI cards, classification distribution charts,
calibrated posterior probability dashboard, metric interpretation guide,
claim-level proposition cards with citation provenance, and raw payload inspector.
"""

from typing import Any, Callable, Dict, List, Optional
import altair as alt
import pandas as pd
import streamlit as st
from frontend.components.header import render_page_header
from ml_engine.pipeline import PipelineResult
from ml_engine.claim_verifier.models import VerificationCategory, VerifiedClaim


def get_classification_str(claim: VerifiedClaim) -> str:
    """Safely extracts normalized uppercase classification string from VerifiedClaim."""
    if not claim.is_factual:
        return "NON_FACTUAL"
    cat = claim.classification
    if hasattr(cat, "value"):
        return str(cat.value).upper()
    return str(cat).upper()


def normalize_calibrated_probabilities(raw: Dict[str, float]) -> Dict[str, float]:
    """Normalizes raw probability dictionary keys across both abbreviations (sup, contra, unsup) and full names.

    Ensures seamless compatibility between pipeline metadata ('sup', 'contra', 'unsup') and UI labels
    ('Support', 'Contradiction', 'Unsupported').
    """
    p_sup = float(raw.get("Support", raw.get("sup", 0.0)))
    p_contra = float(raw.get("Contradiction", raw.get("contra", 0.0)))
    default_unsup = max(0.0, round(1.0 - p_sup - p_contra, 4))
    p_unsup = float(raw.get("Unsupported", raw.get("unsup", default_unsup)))
    return {
        "Support": p_sup,
        "Contradiction": p_contra,
        "Unsupported": p_unsup,
    }


def filter_claims_by_category(
    claims: List[VerifiedClaim], filter_choice: str
) -> List[VerifiedClaim]:
    """Filters a list of VerifiedClaim objects according to user-selected classification category."""
    choice_prefix = filter_choice.split()[0].upper()

    if choice_prefix == "ALL":
        return claims
    elif choice_prefix == "SUPPORTED":
        return [c for c in claims if get_classification_str(c) == "SUPPORTED"]
    elif choice_prefix == "CONTRADICTED":
        return [c for c in claims if get_classification_str(c) == "CONTRADICTED"]
    elif choice_prefix == "UNSUPPORTED":
        return [c for c in claims if get_classification_str(c) == "UNSUPPORTED"]
    elif choice_prefix == "UNVERIFIABLE":
        return [c for c in claims if get_classification_str(c) == "UNVERIFIABLE"]
    elif choice_prefix.startswith("NON"):
        return [c for c in claims if get_classification_str(c) == "NON_FACTUAL"]
    return claims


def build_classification_distribution_chart(result: PipelineResult) -> Optional[alt.Chart]:
    """Constructs an Altair bar chart displaying the factual proposition classification distribution.

    Visualizes SUPPORTED, CONTRADICTED, UNSUPPORTED, and UNVERIFIABLE counts over empirical claims.
    Non-factual claims are intentionally excluded to match factual scoring denominators.
    """
    counts = result.score.counts
    df = pd.DataFrame([
        {"Category": "Supported", "Count": counts.supported_count, "Rate": f"{result.score.rates.evidence_support_rate:.1f}%" if result.score.rates.evidence_support_rate is not None else "N/A"},
        {"Category": "Contradicted", "Count": counts.contradicted_count, "Rate": f"{result.score.rates.confirmed_contradiction_rate:.1f}%" if result.score.rates.confirmed_contradiction_rate is not None else "N/A"},
        {"Category": "Unsupported", "Count": counts.unsupported_count, "Rate": f"{result.score.rates.unsupported_rate:.1f}%" if result.score.rates.unsupported_rate is not None else "N/A"},
        {"Category": "Unverifiable", "Count": counts.unverifiable_count, "Rate": f"{result.score.rates.unverifiable_rate:.1f}%" if result.score.rates.unverifiable_rate is not None else "N/A"},
    ])

    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadius=4)
        .encode(
            x=alt.X("Category:N", sort=None, axis=alt.Axis(labelColor="#D1D5DB", title=None, labelAngle=0)),
            y=alt.Y("Count:Q", axis=alt.Axis(labelColor="#D1D5DB", title="Claims", tickMinStep=1)),
            color=alt.Color(
                "Category:N",
                scale=alt.Scale(
                    domain=["Supported", "Contradicted", "Unsupported", "Unverifiable"],
                    range=["#10B981", "#EF4444", "#F59E0B", "#6B7280"],
                ),
                legend=None,
            ),
            tooltip=["Category:N", "Count:Q", "Rate:N"],
        )
        .properties(height=210)
        .configure_view(strokeOpacity=0)
        .configure_axis(gridColor="#1F2937")
    )
    return chart


def build_calibrated_probabilities_chart(result: PipelineResult) -> Optional[alt.Chart]:
    """Constructs an Altair horizontal stacked bar chart of calibrated DeBERTa posterior probabilities."""
    cal_probs_map: Dict[str, Dict[str, float]] = result.metadata.get("calibrated_probabilities", {})
    if not cal_probs_map:
        return None

    rows = []
    for idx, claim in enumerate(result.claims):
        if claim.claim_id in cal_probs_map:
            probs = normalize_calibrated_probabilities(cal_probs_map[claim.claim_id])
            short_text = (
                claim.claim_text[:30] + "..."
                if len(claim.claim_text) > 30
                else claim.claim_text
            )
            label = f"#{idx+1}: {short_text}"
            for cat, col_name, color in [
                ("Support", "Support", "#10B981"),
                ("Contradiction", "Contradiction", "#EF4444"),
                ("Unsupported", "Unsupported", "#F59E0B"),
            ]:
                val = float(probs.get(cat, 0.0))
                rows.append({
                    "Claim": label,
                    "Class": col_name,
                    "Probability": val,
                    "Percent": f"{val * 100:.1f}%",
                })

    if not rows:
        return None

    df = pd.DataFrame(rows)
    chart_height = max(130, len(cal_probs_map) * 44)

    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadius=3)
        .encode(
            y=alt.Y("Claim:N", sort=None, axis=alt.Axis(labelColor="#D1D5DB", title=None, labelLimit=260)),
            x=alt.X(
                "Probability:Q",
                axis=alt.Axis(format="%", labelColor="#D1D5DB", title="Posterior Probability"),
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color(
                "Class:N",
                scale=alt.Scale(
                    domain=["Support", "Contradiction", "Unsupported"],
                    range=["#10B981", "#EF4444", "#F59E0B"],
                ),
                legend=alt.Legend(
                    title="Calibrated Class",
                    orient="top",
                    labelColor="#D1D5DB",
                    titleColor="#FFFFFF",
                ),
            ),
            tooltip=["Claim:N", "Class:N", "Percent:N"],
        )
        .properties(height=chart_height)
        .configure_view(strokeOpacity=0)
        .configure_axis(gridColor="#1F2937")
    )
    return chart


def render_metric_interpretation_guide() -> None:
    """Renders a structured, compact reference guide explaining metric formulations and epistemic semantics."""
    with st.expander("📘 Metric & Classification Interpretation Guide", expanded=False):
        st.markdown(
            """
            <div style="font-size: 0.88rem; color: #D1D5DB; line-height: 1.6;">
                <div style="margin-bottom: 12px;">
                    <strong style="color: #10B981;">Grounded Factuality Index (GFI):</strong>
                    Measures the percentage of empirical factual claims directly substantiated by reference context 
                    <code>GFI = (supported / N) * 100</code>. GFI quantifies <em>document-grounded consistency</em>; 
                    it is not a universal truth guarantee.
                </div>
                <div style="margin-bottom: 12px;">
                    <strong style="color: #3B82F6;">Contradiction-Penalized Factuality (CPFS):</strong>
                    Applies an active contradiction penalty (<code>&lambda; = 1.0</code>): 
                    <code>max(0.0, (supported - &lambda; * contradicted) / N) * 100</code>. 
                    Ensures candidate answers containing active hallucinations receive heavily degraded scores.
                </div>
                <div style="margin-bottom: 12px;">
                    <strong style="color: #EF4444;">Hallucination Severity Index (HSI):</strong>
                    Measures the proportion of empirical claims actively refuted by the evidence 
                    <code>HSI = (contradicted / N) * 100</code> (equivalent to Confirmed Contradiction Rate, CCR).
                </div>
                <div style="margin-bottom: 12px;">
                    <strong style="color: #6B7280;">UNVERIFIABLE (Safe Abstention):</strong>
                    Triggered when candidate passage similarity is below the safety threshold 
                    (<code>&tau;floor = 0.20</code>) or reference text is absent. The verifier <strong>abstains</strong>; 
                    this reflects epistemic uncertainty, <strong>not a confirmed hallucination</strong>.
                </div>
                <div>
                    <strong style="color: #F59E0B;">UNSUPPORTED (Unmentioned):</strong>
                    Candidate passages were retrieved above the similarity floor, but the reference ground 
                    did not establish support. This indicates the assertion is unmentioned; 
                    it does <strong>not automatically mean the claim is false</strong>.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_claim_card(
    claim: VerifiedClaim,
    index: int,
    calibrated_probs: Optional[Dict[str, float]] = None,
) -> None:
    """Renders a single factual or non-factual claim card with evidence and provenance."""
    cat_str = get_classification_str(claim)

    # Style mapping
    if cat_str == "SUPPORTED":
        badge_class = "badge-supported"
        border_color = "#10B981"
        badge_label = "SUPPORTED"
        box_class = "evidence-quote-box-supported"
        evidence_title = "Grounded Supporting Evidence (Passage Rank 1)"
    elif cat_str == "CONTRADICTED":
        badge_class = "badge-contradicted"
        border_color = "#EF4444"
        badge_label = "CONTRADICTED"
        box_class = "evidence-quote-box-contradicted"
        evidence_title = "Directly Refuting Reference Passage (Passage Rank 1)"
    elif cat_str == "UNSUPPORTED":
        badge_class = "badge-unsupported"
        border_color = "#F59E0B"
        badge_label = "UNSUPPORTED"
        box_class = "evidence-quote-box-unsupported"
        evidence_title = "Closest Candidate Passage (Passage Rank 1 — Claim Not Supported)"
    elif cat_str == "UNVERIFIABLE":
        badge_class = "badge-unverifiable"
        border_color = "#6B7280"
        badge_label = "UNVERIFIABLE"
        box_class = "evidence-quote-box-unverifiable"
        evidence_title = "Low-Similarity Passage (Passage Rank 1 — Below Similarity Floor)"
    else:
        badge_class = "badge-nonfactual"
        border_color = "#6366F1"
        badge_label = "NON-FACTUAL"
        box_class = "evidence-quote-box-unverifiable"
        evidence_title = "Non-Factual Remark"

    # Card Container
    st.markdown(
        f"""
        <div class="claim-card" style="border-left: 4px solid {border_color};">
            <div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #1F2937; padding-bottom: 8px; margin-bottom: 10px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 0.82rem; font-weight: 700; color: #9CA3AF; text-transform: uppercase; letter-spacing: 0.05em;">Claim #{index + 1}</span>
                    <span style="font-size: 0.75rem; color: #6B7280;">({claim.claim_id})</span>
                </div>
                <span class="claim-badge {badge_class}">{badge_label}</span>
            </div>
            <div style="font-size: 1.02rem; font-weight: 500; color: #FFFFFF; line-height: 1.5; margin-bottom: 8px;">
                &ldquo;{claim.claim_text}&rdquo;
            </div>
            {f'<div style="font-size: 0.8rem; color: #9CA3AF; margin-bottom: 10px;">Source sentence: <em>&ldquo;{claim.source_sentence}&rdquo;</em></div>' if (claim.source_sentence and claim.source_sentence.strip() != claim.claim_text.strip()) else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Probabilities & Confidence Bar
    if calibrated_probs:
        norm_p = normalize_calibrated_probabilities(calibrated_probs)
        p_sup = norm_p["Support"]
        p_contra = norm_p["Contradiction"]
        p_unsup = norm_p["Unsupported"]
        st.markdown(
            f"""
            <div style="margin-top: -10px; margin-bottom: 10px; padding: 0 4px; display: flex; flex-wrap: wrap; gap: 6px;">
                <span class="prob-metric-pill" style="border-color: #059669; color: #34D399;">Support: <strong>{p_sup:.1%}</strong></span>
                <span class="prob-metric-pill" style="border-color: #DC2626; color: #F87171;">Contradiction: <strong>{p_contra:.1%}</strong></span>
                <span class="prob-metric-pill" style="border-color: #D97706; color: #FBBF24;">Unsupported: <strong>{p_unsup:.1%}</strong></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Evidence & Provenance
    if claim.primary_evidence and claim.primary_evidence.chunk:
        chunk = claim.primary_evidence.chunk
        sim_val = claim.primary_evidence.similarity_score
        sim_label = (
            "1.000 (Direct Context)"
            if chunk.chunk_id.endswith("_direct")
            else f"{sim_val:.3f}"
        )
        page_str = f"Page {chunk.page_number} • " if chunk.page_number is not None else ""

        st.markdown(
            f"""
            <div class="evidence-quote-box {box_class}" style="margin-top: -4px; margin-bottom: 8px;">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
                    <span style="font-weight: 600; font-size: 0.82rem; color: #E5E7EB;">{evidence_title}</span>
                    <span style="font-size: 0.75rem; color: #9CA3AF;">Cosine Sim: <strong>{sim_label}</strong></span>
                </div>
                <div style="font-style: italic; color: #D1D5DB; margin-bottom: 8px;">
                    &ldquo;{chunk.text}&rdquo;
                </div>
                <div style="font-size: 0.75rem; color: #6B7280; border-top: 1px solid #1F2937; padding-top: 6px;">
                    Provenance: Document: <strong>{chunk.document_name}</strong> • {page_str}Chunk ID: <code>{chunk.chunk_id}</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if cat_str == "UNVERIFIABLE":
            st.caption(
                "ℹ️ **Abstention Note:** Passage similarity is below the 0.20 safety floor. "
                "The system abstains from verifying; this does not confirm the claim is false."
            )
        elif cat_str == "UNSUPPORTED":
            st.caption(
                "ℹ️ **Evidence Note:** The reference text does not corroborate this proposition. "
                "The claim is unmentioned in the ground-truth document."
            )

    else:
        # Missing / Absent Evidence
        if cat_str == "UNVERIFIABLE":
            st.markdown(
                """
                <div class="evidence-quote-box evidence-quote-box-unverifiable" style="margin-top: -4px; margin-bottom: 8px;">
                    <div style="font-weight: 600; font-size: 0.82rem; color: #9CA3AF; margin-bottom: 4px;">Evidence Absent (Safe Abstention)</div>
                    <div style="color: #9CA3AF;">
                        No relevant passage could be retrieved from the reference document (or document context is absent).
                        The verifier safely abstains; this claim is <strong>unverified, not proven false</strong>.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif cat_str == "UNSUPPORTED":
            st.markdown(
                """
                <div class="evidence-quote-box evidence-quote-box-unsupported" style="margin-top: -4px; margin-bottom: 8px;">
                    <div style="font-weight: 600; font-size: 0.82rem; color: #F59E0B; margin-bottom: 4px;">No Supporting Evidence Found</div>
                    <div style="color: #9CA3AF;">
                        No corroborating passage exists in the reference context. The asserted fact is unmentioned in the source ground.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif cat_str == "NON_FACTUAL":
            st.markdown(
                """
                <div class="evidence-quote-box evidence-quote-box-unverifiable" style="margin-top: -4px; margin-bottom: 8px;">
                    <div style="font-weight: 600; font-size: 0.82rem; color: #818CF8; margin-bottom: 4px;">Non-Empirical Proposition</div>
                    <div style="color: #9CA3AF;">
                        Conversational greeting, subjective opinion, or rhetorical statement. Excluded from factuality scoring.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Expandable Technical Trace
    with st.expander(f"View Technical Trace & Reasoning ({claim.claim_id})", expanded=False):
        st.markdown(f"**Deterministic Decision Rule:** ` {claim.decision_rule} `")
        st.markdown(f"**Reasoning Explanation:** {claim.reasoning_trace}")
        st.markdown(
            f"**Candidate Response Offsets:** ` [{claim.start_char}:{claim.end_char}] ` "
            f"(Fidelity: **{claim.offset_type}**)"
        )

        # Show secondary candidate passages if multiple were retrieved
        if claim.all_evidence and len(claim.all_evidence) > 1:
            st.markdown("---")
            st.markdown(f"**All Candidate Passages Evaluated ({len(claim.all_evidence)} total):**")
            for p in claim.all_evidence:
                st.markdown(
                    f"- **Rank {p.rank}** (`{p.chunk.chunk_id}` • Sim: `{p.similarity_score:.3f}`): "
                    f"_{p.chunk.text[:140]}..._"
                )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)


def render_results_view(on_navigate_verify: Optional[Callable[[], None]] = None) -> None:
    """Renders the Results & Metrics view with actual pipeline results or empty state."""
    render_page_header(
        title="Verification Results & Metrics",
        subtitle=(
            "Inspect structured claim-level factuality classifications, citation provenance, "
            "calibrated posterior probabilities, and mathematical response-level hallucination indices."
        ),
        badge="Analytics Dashboard",
    )

    # Check for active verification result in session state
    result: Optional[PipelineResult] = (
        st.session_state.get("verification_result")
        or st.session_state.get("pipeline_result")
    )
    has_results = result is not None

    if not has_results:
        # ---------------------------------------------------------------------
        # Clean Empty State
        # ---------------------------------------------------------------------
        st.markdown(
            """
            <div class="placeholder-box" style="padding: 60px 24px;">
                <div style="width: 48px; height: 48px; border-radius: 12px; background-color: #1E293B; border: 1px solid #334155; display: inline-flex; align-items: center; justify-content: center; margin-bottom: 16px;">
                    <span style="font-size: 1.5rem; color: #60A5FA;">📊</span>
                </div>
                <h3>No Verification Results Available Yet</h3>
                <p>
                    Execute a verification workflow from the <strong>Verify Response</strong> tab to inspect 
                    grounded factuality rates (GFI), contradiction penalties (CPFS), and claim-level NLI traces.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_left, col_btn, col_right = st.columns([1, 1, 1])
        with col_btn:
            st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
            if st.button("Go to Verify Response →", type="secondary", use_container_width=True, key="btn_empty_verify"):
                if on_navigate_verify:
                    on_navigate_verify()

        st.markdown("<div style='height: 32px;'></div>", unsafe_allow_html=True)

        # Overview of Planned Visualization Architecture
        st.markdown("### Planned Results Visualization Structure")
        st.markdown(
            "<p style='color: #9CA3AF; font-size: 0.95rem; margin-top: -8px; margin-bottom: 18px;'>"
            "When active verification data is supplied, this view renders:"
            "</p>",
            unsafe_allow_html=True,
        )

        col_a, col_b, col_c = st.columns(3, gap="medium")

        with col_a:
            st.markdown(
                """
                <div class="verillm-card">
                    <div class="verillm-card-header">
                        <span>Response Factuality Indices</span>
                    </div>
                    <p class="verillm-card-body">
                        Visual scorecards for <strong>Grounded Factuality Index (GFI)</strong>, 
                        <strong>Contradiction-Penalized Factuality (CPFS)</strong>, and 
                        <strong>Hallucination Severity Index (HSI)</strong>.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_b:
            st.markdown(
                """
                <div class="verillm-card">
                    <div class="verillm-card-header">
                        <span>Proposition Breakdown</span>
                    </div>
                    <p class="verillm-card-body">
                        Partitioned distribution bars of <em>SUPPORTED</em>, <em>CONTRADICTED</em>, 
                        <em>UNSUPPORTED</em>, <em>UNVERIFIABLE</em>, and non-factual claims.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_c:
            st.markdown(
                """
                <div class="verillm-card">
                    <div class="verillm-card-header">
                        <span>Claim & Provenance Traces</span>
                    </div>
                    <p class="verillm-card-body">
                        Interactive claim explorer with citation snippets, similarity scores, 
                        and calibrated posterior probabilities from the DeBERTa model.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    else:
        # =====================================================================
        # Active Result View with Visual Analytics Dashboard (Step 10.5)
        # =====================================================================
        regime_label = (
            "Retrieval Augmented"
            if result.regime.lower() == "retrieval"
            else "Direct Context"
        )
        timestamp = st.session_state.get("last_verification_time", "Recent Execution")
        staged_inp = st.session_state.get("prepared_inputs") or {}
        workflow_label = staged_inp.get("workflow_mode", "Manual Verification")
        gen_meta = staged_inp.get("generation_metadata")
        gen_info_str = ""
        if gen_meta and gen_meta.get("model_id"):
            mod = gen_meta["model_id"]
            lat = gen_meta.get("execution_time_seconds", 0.0)
            gen_info_str = f" • Generated by <strong>{mod}</strong> ({lat:.2f}s)"

        # 1. Provenance Banner
        st.markdown(
            f"""
            <div class="summary-box" style="margin-top: 0; margin-bottom: 20px;">
                <div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #1F2937; padding-bottom: 8px; margin-bottom: 10px;">
                    <div>
                        <span style="font-weight: 600; color: #FFFFFF; font-size: 1.05rem;">Verification Summary</span>
                        <span style="color: #9CA3AF; font-size: 0.85rem; margin-left: 10px;">• Executed in <strong>{result.execution_time_seconds:.2f}s</strong></span>
                    </div>
                    <span class="status-pill"><span class="status-dot-active"></span>DeBERTa-v3 Calibrated</span>
                </div>
                <div style="font-size: 0.88rem; color: #D1D5DB;">
                    Document: <strong>{result.document_name}</strong> • Regime: <strong>{regime_label}</strong> • Workflow: <strong>{workflow_label}</strong>{gen_info_str} • Timestamp: <strong>{timestamp}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 2. Key Metric KPI Cards (with descriptive tooltips)
        gfi = result.score.indices.grounded_factuality_index
        cpfs = result.score.indices.contradiction_penalized_factuality
        hsi = result.score.indices.hallucination_severity_index

        gfi_str = f"{gfi:.1f}%" if gfi is not None else "N/A"
        cpfs_str = f"{cpfs:.1f}%" if cpfs is not None else "N/A"
        hsi_str = f"{hsi:.1f}%" if hsi is not None else "N/A"
        factual_ratio = f"{result.score.counts.factual_claims} / {result.score.counts.total_claims}"

        col_m1, col_m2, col_m3, col_m4 = st.columns(4, gap="medium")

        with col_m1:
            st.markdown(
                f"""
                <div class="summary-item">
                    <div class="summary-item-label" title="Grounded Factuality Index: Percentage of factual claims confirmed by reference evidence.">
                        Grounded Factuality (GFI) ℹ️
                    </div>
                    <div style="font-size: 1.45rem; font-weight: 700; color: #10B981;">{gfi_str}</div>
                    <div style="font-size: 0.78rem; color: #9CA3AF; margin-top: 4px;">Supported / Factual claims</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_m2:
            st.markdown(
                f"""
                <div class="summary-item">
                    <div class="summary-item-label" title="Contradiction-Penalized Factuality: max(0, (Supported - lambda * Contradicted) / N) * 100 with lambda = 1.0.">
                        Penalized Factuality (CPFS) ℹ️
                    </div>
                    <div style="font-size: 1.45rem; font-weight: 700; color: #3B82F6;">{cpfs_str}</div>
                    <div style="font-size: 0.78rem; color: #9CA3AF; margin-top: 4px;">Active contradiction penalty</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_m3:
            hsi_color = "#EF4444" if (hsi is not None and hsi > 0) else "#10B981"
            st.markdown(
                f"""
                <div class="summary-item">
                    <div class="summary-item-label" title="Hallucination Severity Index: Percentage of empirical claims actively refuted by the evidence.">
                        Hallucination Severity (HSI) ℹ️
                    </div>
                    <div style="font-size: 1.45rem; font-weight: 700; color: {hsi_color};">{hsi_str}</div>
                    <div style="font-size: 0.78rem; color: #9CA3AF; margin-top: 4px;">Actively refuted propositions</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_m4:
            st.markdown(
                f"""
                <div class="summary-item">
                    <div class="summary-item-label" title="Factual claims evaluated (N) out of total extracted propositions (including non-factual remarks).">
                        Factual Claims (N) ℹ️
                    </div>
                    <div style="font-size: 1.45rem; font-weight: 700; color: #F3F4F6;">{factual_ratio}</div>
                    <div style="font-size: 0.78rem; color: #9CA3AF; margin-top: 4px;">Empirical vs total extracted</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Detailed Rate Metrics Bar (ESR, CCR, USR, UVR)
        rates = result.score.rates
        esr_str = f"{rates.evidence_support_rate:.1f}%" if rates.evidence_support_rate is not None else "N/A"
        ccr_str = f"{rates.confirmed_contradiction_rate:.1f}%" if rates.confirmed_contradiction_rate is not None else "N/A"
        usr_str = f"{rates.unsupported_rate:.1f}%" if rates.unsupported_rate is not None else "N/A"
        uvr_str = f"{rates.unverifiable_rate:.1f}%" if rates.unverifiable_rate is not None else "N/A"
        cur_str = f"{rates.combined_uncertainty_rate:.1f}%" if rates.combined_uncertainty_rate is not None else "N/A"

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="display: flex; flex-wrap: wrap; gap: 8px; padding: 8px 12px; background-color: #0E1420; border: 1px solid #1F2937; border-radius: 6px; font-size: 0.8rem; color: #9CA3AF;">
                <span>Detailed Rates:</span>
                <span class="prob-metric-pill" style="border-color: #059669; color: #34D399;">ESR: <strong>{esr_str}</strong></span>
                <span class="prob-metric-pill" style="border-color: #DC2626; color: #F87171;">CCR: <strong>{ccr_str}</strong></span>
                <span class="prob-metric-pill" style="border-color: #D97706; color: #FBBF24;">USR: <strong>{usr_str}</strong></span>
                <span class="prob-metric-pill" style="border-color: #4B5563; color: #9CA3AF;">UVR: <strong>{uvr_str}</strong></span>
                <span class="prob-metric-pill" style="border-color: #374151; color: #D1D5DB;">Combined Uncertainty: <strong>{cur_str}</strong></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 3. Operational Warnings (if any)
        if result.warnings:
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            for w in result.warnings:
                st.warning(w, icon="⚠️")

        # =====================================================================
        # 4. Visual Analytics: Classification Distribution & Composition (Step 10.5)
        # =====================================================================
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        st.markdown("### Visual Verification Analytics")

        col_viz_chart, col_viz_comp = st.columns([3, 2], gap="medium")

        with col_viz_chart:
            st.markdown("#### Factual Claim Distribution")
            if result.score.counts.factual_claims > 0:
                dist_chart = build_classification_distribution_chart(result)
                if dist_chart is not None:
                    st.altair_chart(dist_chart, use_container_width=True)
            else:
                st.info("No empirical factual claims in candidate response (all propositions are non-factual).", icon="ℹ️")

        with col_viz_comp:
            st.markdown("#### Proposition Composition")
            tot_propositions = max(1, result.score.counts.total_claims)
            fact_pct = (result.score.counts.factual_claims / tot_propositions) * 100
            nonfact_pct = (result.score.counts.non_factual_count / tot_propositions) * 100

            st.markdown(
                f"""
                <div class="summary-box" style="margin-top: 0; padding: 14px 16px;">
                    <div style="font-size: 0.85rem; color: #D1D5DB; margin-bottom: 10px;">
                        Composition of <strong>{result.score.counts.total_claims}</strong> extracted propositions:
                    </div>
                    <div style="margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between; font-size: 0.82rem; margin-bottom: 2px;">
                            <span style="color: #60A5FA;">Factual Propositions (N)</span>
                            <span style="font-weight: 600; color: #FFFFFF;">{result.score.counts.factual_claims} ({fact_pct:.1f}%)</span>
                        </div>
                        <div style="width: 100%; height: 6px; background-color: #1F2937; border-radius: 3px; overflow: hidden;">
                            <div style="width: {fact_pct:.1f}%; height: 100%; background-color: #3B82F6;"></div>
                        </div>
                    </div>
                    <div style="margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; font-size: 0.82rem; margin-bottom: 2px;">
                            <span style="color: #818CF8;">Non-Factual Remarks</span>
                            <span style="font-weight: 600; color: #FFFFFF;">{result.score.counts.non_factual_count} ({nonfact_pct:.1f}%)</span>
                        </div>
                        <div style="width: 100%; height: 6px; background-color: #1F2937; border-radius: 3px; overflow: hidden;">
                            <div style="width: {nonfact_pct:.1f}%; height: 100%; background-color: #6366F1;"></div>
                        </div>
                    </div>
                    <div style="font-size: 0.76rem; color: #6B7280; line-height: 1.4; border-top: 1px solid #1F2937; padding-top: 8px;">
                        Factual propositions assert testable claims. Non-factual remarks (opinions, greetings) are partitioned out of factuality rates.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # =====================================================================
        # 5. Calibrated Posterior Probability Dashboard (Step 10.5)
        # =====================================================================
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px;">
                <h4 style="margin: 0; font-size: 1.05rem; color: #FFFFFF;">Calibrated Probability Distribution</h4>
                <span class="status-pill" style="font-size: 0.72rem;">Frozen C2 Calibrator</span>
            </div>
            <p style="color: #9CA3AF; font-size: 0.85rem; margin-top: 0; margin-bottom: 10px;">
                Visualizes the calibrated posterior class probabilities for each evaluated proposition. 
                These values reflect model posterior beliefs conditioned on evidence, not absolute guarantees of real-world truth.
            </p>
            """,
            unsafe_allow_html=True,
        )

        prob_chart = build_calibrated_probabilities_chart(result)
        if prob_chart is not None:
            st.altair_chart(prob_chart, use_container_width=True)
        else:
            st.info("No calibrated probability values available for this response (e.g. non-factual or unverified claims).", icon="ℹ️")

        # =====================================================================
        # 6. Metric & Verification Interpretation Guide (Step 10.5)
        # =====================================================================
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        render_metric_interpretation_guide()

        # =====================================================================
        # 7. Detailed Claim-Level Inspector (Step 10.4)
        # =====================================================================
        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px;">
                <h3 style="margin: 0; font-size: 1.15rem; color: #FFFFFF;">Detailed Claim-Level Inspector</h3>
                <span class="status-pill" style="font-size: 0.72rem;">Atomic Propositions</span>
            </div>
            <p style="color: #9CA3AF; font-size: 0.88rem; margin-top: 0; margin-bottom: 12px;">
                Inspect each proposition's factual verification classification, evidence citations, retrieval similarity, 
                and calibrated DeBERTa posterior probabilities.
            </p>
            """,
            unsafe_allow_html=True,
        )

        all_claims = result.claims
        total_claims_count = len(all_claims)
        s_cnt = result.score.counts.supported_count
        c_cnt = result.score.counts.contradicted_count
        u_cnt = result.score.counts.unsupported_count
        uv_cnt = result.score.counts.unverifiable_count
        nf_cnt = result.score.counts.non_factual_count

        filter_options = [
            f"All ({total_claims_count})",
            f"Supported ({s_cnt})",
            f"Contradicted ({c_cnt})",
            f"Unsupported ({u_cnt})",
            f"Unverifiable ({uv_cnt})",
            f"Non-Factual ({nf_cnt})",
        ]

        filter_choice = st.radio(
            "Filter Claims by Classification",
            options=filter_options,
            index=0,
            horizontal=True,
            label_visibility="collapsed",
            key="claims_classification_filter",
        )

        filtered_claims = filter_claims_by_category(all_claims, filter_choice)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        if not all_claims:
            st.info("No atomic claims were extracted from this candidate response.", icon="ℹ️")
        elif not filtered_claims:
            st.info(f"No claims match the filter '{filter_choice}'.", icon="ℹ️")
        else:
            # Map of calibrated probabilities if present in metadata
            cal_probs_map: Dict[str, Dict[str, float]] = result.metadata.get(
                "calibrated_probabilities", {}
            )

            for idx, claim in enumerate(filtered_claims):
                claim_cal_probs = cal_probs_map.get(claim.claim_id)
                render_claim_card(claim, index=idx, calibrated_probs=claim_cal_probs)

        # 8. Raw Result Inspector
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        with st.expander("Inspect Raw Pipeline Result (JSON)", expanded=False):
            st.json(result.to_dict())

        # 9. Navigation Actions
        col_act_left, col_act_right = st.columns([1, 1], gap="medium")
        with col_act_left:
            if st.button("← Verify Another Response", type="primary", use_container_width=True, key="btn_verify_another"):
                if on_navigate_verify:
                    on_navigate_verify()

        with col_act_right:
            if st.button("Clear Results", type="secondary", use_container_width=True, key="btn_clear_results"):
                st.session_state["verification_result"] = None
                st.session_state["last_verification_time"] = None
                st.rerun()
