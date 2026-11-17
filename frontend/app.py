"""
VeriLLM Streamlit Application Entrypoint.
Sets up page configuration, custom design system, sidebar navigation,
and routes between Overview, Verify Response, and Results views.
"""

import sys
from pathlib import Path

# Anchor project root to sys.path so modules import reliably
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import streamlit as st

from frontend.utils.styles import apply_custom_styles
from frontend.components.header import render_sidebar_branding
from frontend.views.overview import render_overview_view
from frontend.views.verify import render_verify_view
from frontend.views.results import render_results_view

# -----------------------------------------------------------------------------
# Streamlit Application Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="VeriLLM • Hallucination Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply Modern AI SaaS dark theme styling
apply_custom_styles()

# Render persistent sidebar header & status badge
render_sidebar_branding()

# -----------------------------------------------------------------------------
# Application Navigation (Overview, Verify Response, Results)
# -----------------------------------------------------------------------------
def view_overview():
    render_overview_view(on_navigate_verify=lambda: st.switch_page(page_verify))

def view_verify():
    render_verify_view(on_navigate_results=lambda: st.switch_page(page_results))

def view_results():
    render_results_view(on_navigate_verify=lambda: st.switch_page(page_verify))

page_overview = st.Page(
    view_overview,
    title="Overview",
    icon=":material/dashboard:",
    default=True,
)

page_verify = st.Page(
    view_verify,
    title="Verify Response",
    icon=":material/fact_check:",
)

page_results = st.Page(
    view_results,
    title="Results",
    icon=":material/analytics:",
)

nav = st.navigation(
    {"Navigation": [page_overview, page_verify, page_results]},
    position="sidebar",
)

# Persistent sidebar metadata footer
with st.sidebar:
    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.78rem; color: #6B7280; padding: 4px 0;">
            <div><strong>VeriLLM System</strong> • v1.0.0</div>
            <div>Model: DeBERTa-v3 + Frozen C2</div>
            <div>Mode: Direct / Retrieval (Dual)</div>
            <div style="margin-top: 6px; color: #4B5563;">Offline Local Research Stack</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Execute active route
nav.run()

