"""
VeriLLM Frontend Styles Module.
Provides clean CSS injections and styling utilities for a professional Modern AI SaaS dashboard theme.
Restrained color palette: Dark Navy / Charcoal (#0B0F17, #111827), Slate Blue Accent (#3B82F6), Cool Grey (#9CA3AF).
"""

import streamlit as st

CUSTOM_CSS = """
<style>
/* -------------------------------------------------------------
 * Global Typography & Layout Tuning
 * ------------------------------------------------------------- */
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #F3F4F6;
    background-color: #0B0F17;
}

/* Header & Title Adjustments */
h1, h2, h3, h4, h5, h6 {
    color: #FFFFFF;
    font-weight: 600;
    letter-spacing: -0.02em;
}

/* Hide default streamlit top header line decoration */
[data-testid="stDecoration"] {
    display: none;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #0E1420;
    border-right: 1px solid #1F2937;
}

[data-testid="stSidebar"] hr {
    border-color: #1F2937;
}

/* -------------------------------------------------------------
 * Custom VeriLLM Component Classes
 * ------------------------------------------------------------- */

/* Hero Banner */
.verillm-hero {
    padding: 16px 0 28px 0;
    border-bottom: 1px solid #1F2937;
    margin-bottom: 28px;
}

.verillm-hero-badge {
    display: inline-block;
    padding: 4px 12px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    background-color: rgba(59, 130, 246, 0.12);
    color: #60A5FA;
    border: 1px solid rgba(59, 130, 246, 0.28);
    border-radius: 9999px;
    margin-bottom: 14px;
}

.verillm-hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: -0.03em;
    line-height: 1.2;
    margin: 0 0 10px 0;
}

.verillm-hero-subtitle {
    font-size: 1.05rem;
    color: #9CA3AF;
    line-height: 1.6;
    margin: 0;
    max-width: 820px;
}

/* Cards & Containers */
.verillm-card {
    background-color: #111827;
    border: 1px solid #1F2937;
    border-radius: 10px;
    padding: 22px;
    margin-bottom: 18px;
    transition: border-color 0.15s ease-in-out;
}

.verillm-card:hover {
    border-color: #374151;
}

.verillm-card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 1.05rem;
    font-weight: 600;
    color: #F9FAFB;
    margin-bottom: 10px;
}

.verillm-card-body {
    font-size: 0.9rem;
    color: #9CA3AF;
    line-height: 1.55;
    margin: 0;
}

/* Feature & Step Items */
.step-item {
    display: flex;
    align-items: flex-start;
    margin-bottom: 16px;
}

.step-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 26px;
    height: 26px;
    border-radius: 6px;
    background-color: #1E293B;
    border: 1px solid #334155;
    color: #60A5FA;
    font-size: 0.8rem;
    font-weight: 700;
    margin-right: 14px;
    flex-shrink: 0;
    margin-top: 2px;
}

.step-text h4 {
    margin: 0 0 4px 0;
    font-size: 0.95rem;
    color: #F3F4F6;
}

.step-text p {
    margin: 0;
    font-size: 0.875rem;
    color: #9CA3AF;
    line-height: 1.5;
}

/* Status Indicators */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.78rem;
    color: #9CA3AF;
    background-color: #111827;
    border: 1px solid #1F2937;
    padding: 4px 10px;
    border-radius: 6px;
}

.status-dot-active {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background-color: #10B981;
}

.status-dot-neutral {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background-color: #6B7280;
}

/* Input & Summary Helpers */
.input-counter {
    font-size: 0.78rem;
    color: #6B7280;
    text-align: right;
    margin-top: -6px;
    margin-bottom: 8px;
}

.summary-box {
    background-color: #111827;
    border: 1px solid #1F2937;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 16px 0;
}

.summary-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-top: 10px;
}

.summary-item {
    background-color: #0B0F17;
    border: 1px solid #1F2937;
    border-radius: 6px;
    padding: 10px 14px;
}

.summary-item-label {
    font-size: 0.75rem;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-weight: 600;
    margin-bottom: 4px;
}

.summary-item-value {
    font-size: 0.92rem;
    color: #F3F4F6;
    font-weight: 500;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* Placeholder Container */
.placeholder-box {
    border: 2px dashed #1F2937;
    border-radius: 12px;
    padding: 48px 24px;
    text-align: center;
    background-color: #0E1420;
    margin-top: 16px;
}

.placeholder-box h3 {
    font-size: 1.15rem;
    color: #E5E7EB;
    margin: 12px 0 6px 0;
}

.placeholder-box p {
    font-size: 0.9rem;
    color: #6B7280;
    max-width: 520px;
    margin: 0 auto;
    line-height: 1.5;
}

/* Claim Inspector Cards */
.claim-card {
    background-color: #111827;
    border: 1px solid #1F2937;
    border-radius: 8px;
    padding: 18px 20px;
    margin-bottom: 16px;
    transition: border-color 0.15s ease-in-out;
}

.claim-card:hover {
    border-color: #374151;
}

.claim-badge {
    display: inline-flex;
    align-items: center;
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    padding: 3px 10px;
    border-radius: 6px;
    text-transform: uppercase;
}

.badge-supported {
    background-color: #064E3B;
    border: 1px solid #059669;
    color: #34D399;
}

.badge-contradicted {
    background-color: #450A0A;
    border: 1px solid #DC2626;
    color: #F87171;
}

.badge-unsupported {
    background-color: #451A03;
    border: 1px solid #D97706;
    color: #FBBF24;
}

.badge-unverifiable {
    background-color: #1F2937;
    border: 1px solid #4B5563;
    color: #9CA3AF;
}

.badge-nonfactual {
    background-color: #1E1B4B;
    border: 1px solid #4F46E5;
    color: #818CF8;
}

.evidence-quote-box {
    background-color: #0B0F17;
    border: 1px solid #1F2937;
    border-radius: 6px;
    padding: 12px 16px;
    margin-top: 10px;
    font-size: 0.88rem;
    line-height: 1.55;
    color: #D1D5DB;
}

.evidence-quote-box-supported {
    border-left: 3px solid #10B981;
}

.evidence-quote-box-contradicted {
    border-left: 3px solid #EF4444;
}

.evidence-quote-box-unsupported {
    border-left: 3px solid #F59E0B;
}

.evidence-quote-box-unverifiable {
    border-left: 3px solid #6B7280;
}

.prob-metric-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 0.78rem;
    background-color: #1F2937;
    border: 1px solid #374151;
    color: #E5E7EB;
    padding: 2px 8px;
    border-radius: 4px;
    margin-right: 8px;
}
</style>
"""


def apply_custom_styles() -> None:
    """Injects custom CSS into the Streamlit session for theme refinements."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

