"""
VeriLLM Header Component.
Renders consistent top-level branding and system status indicators.
"""

import streamlit as st


def render_sidebar_branding() -> None:
    """Renders clean sidebar branding, version tag, and system indicators."""
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 4px 0 16px 0;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 28px; height: 28px; border-radius: 6px; background-color: #2563EB; display: flex; align-items: center; justify-content: center; font-weight: 700; color: #FFFFFF; font-size: 0.95rem;">V</div>
                    <div>
                        <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF; letter-spacing: -0.02em;">VeriLLM</div>
                        <div style="font-size: 0.72rem; color: #6B7280; font-weight: 500;">Factuality & Hallucination Engine</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="status-pill" style="margin-bottom: 18px;">
                <span class="status-dot-active"></span>
                <span>Local CPU Inference • Offline Mode</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_page_header(title: str, subtitle: str, badge: str = "VeriLLM Research System") -> None:
    """Renders a standardized page header across application views."""
    st.markdown(
        f"""
        <div class="verillm-hero">
            <div class="verillm-hero-badge">{badge}</div>
            <h1 class="verillm-hero-title">{title}</h1>
            <p class="verillm-hero-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

