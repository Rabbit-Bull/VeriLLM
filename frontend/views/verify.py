"""
VeriLLM Verify Response View (Functional Input Interface).
Supports dual verification workflows:
1. Manual Verification: Document ingestion (PDF/TXT/direct text), optional prompt, manual candidate response.
2. Live Gemini Verification: Document ingestion, mandatory user prompt, Gemini live generation,
   candidate review/editing, stale-response guard, and handoff to the local VeriLLMPipeline.

Crucial Architectural Invariants:
- Reference documents remain strictly local on CPU and are NEVER transmitted to external APIs.
- Gemini NEVER determines whether its own response is hallucinated.
- Both workflows converge on the SAME VeriLLMPipeline execution and Results visualization.
"""

from pathlib import Path
import time
from typing import Any, Callable, Dict, Optional, Tuple
import streamlit as st
from frontend.components.header import render_page_header
from frontend.utils.pipeline_service import execute_verification
from frontend.utils.generation_service import (
    check_gemini_availability,
    generate_candidate_response,
)


def format_file_size(size_bytes: int) -> str:
    """Formats bytes into human-readable B, KB, or MB."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def validate_document_file(
    file_name: str, file_bytes: bytes
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """Validates an uploaded reference document file (PDF or TXT).

    Args:
        file_name: Name of the uploaded file.
        file_bytes: Raw byte contents of the file.

    Returns:
        Tuple of (is_valid, error_message, metadata_dict).
    """
    file_size = len(file_bytes) if file_bytes is not None else 0
    size_display = format_file_size(file_size)

    if file_bytes is None or file_size == 0:
        return False, f"Uploaded file '{file_name}' is empty (0 bytes). Please upload a valid document.", {
            "name": file_name,
            "type": "Empty File",
            "size_display": size_display,
            "text": None,
            "bytes": file_bytes,
        }

    ext = Path(file_name).suffix.lower()

    if ext == ".pdf":
        doc_type = "PDF Document"
        if not file_bytes.startswith(b"%PDF"):
            return False, f"File '{file_name}' has a .pdf extension but lacks a valid PDF header (%PDF).", {
                "name": file_name,
                "type": doc_type,
                "size_display": size_display,
                "text": None,
                "bytes": file_bytes,
            }
        return True, None, {
            "name": file_name,
            "type": doc_type,
            "size_display": size_display,
            "text": None,
            "bytes": file_bytes,
        }

    elif ext == ".txt":
        doc_type = "Plain Text (TXT)"
        try:
            decoded_text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return False, f"File '{file_name}' could not be decoded as UTF-8 text.", {
                "name": file_name,
                "type": doc_type,
                "size_display": size_display,
                "text": None,
                "bytes": file_bytes,
            }

        if not decoded_text.strip():
            return False, f"Uploaded text file '{file_name}' contains only whitespace.", {
                "name": file_name,
                "type": doc_type,
                "size_display": size_display,
                "text": "",
                "bytes": file_bytes,
            }

        words = len(decoded_text.split())
        return True, None, {
            "name": file_name,
            "type": doc_type,
            "size_display": size_display,
            "text": decoded_text,
            "bytes": file_bytes,
            "words": words,
        }

    else:
        return False, f"Unsupported file extension '{ext}'. Only .pdf and .txt are accepted.", {
            "name": file_name,
            "type": "Unsupported",
            "size_display": size_display,
            "text": None,
            "bytes": file_bytes,
        }


def validate_document_text(text: str) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """Validates raw pasted reference document text.

    Args:
        text: Plain-text string input.

    Returns:
        Tuple of (is_valid, error_message, metadata_dict).
    """
    raw_str = text or ""
    words = len(raw_str.split())
    chars = len(raw_str)
    size_display = f"{words} words"

    if not raw_str.strip():
        return False, "Reference text is empty or contains only whitespace.", {
            "name": "Pasted Reference Text",
            "type": "Direct Text Entry",
            "size_display": size_display,
            "words": words,
            "chars": chars,
            "text": "",
        }

    return True, None, {
        "name": "Pasted Reference Text",
        "type": "Direct Text Entry",
        "size_display": size_display,
        "words": words,
        "chars": chars,
        "text": raw_str,
    }


def validate_candidate_response(response_text: str) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """Validates candidate LLM response text.

    Args:
        response_text: Candidate response string input.

    Returns:
        Tuple of (is_valid, error_message, metadata_dict).
    """
    raw_str = response_text or ""
    words = len(raw_str.split())
    chars = len(raw_str)

    if not raw_str:
        return False, "Candidate LLM response is required.", {
            "words": 0,
            "chars": 0,
            "clean_text": "",
        }

    if not raw_str.strip():
        return False, "Candidate response contains only whitespace. Please enter valid text.", {
            "words": 0,
            "chars": chars,
            "clean_text": "",
        }

    return True, None, {
        "words": words,
        "chars": chars,
        "clean_text": raw_str.strip(),
    }


def build_staged_payload(
    doc_mode: str,
    doc_meta: Dict[str, Any],
    prompt: Optional[str],
    response: str,
    regime: str,
    workflow_mode: str = "Manual Verification",
    generation_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Constructs the canonical staged payload ready for Step 10.3 pipeline execution."""
    regime_key = "retrieval" if regime.lower().startswith("retrieval") else "direct"
    clean_prompt = prompt.strip() if (prompt and prompt.strip()) else None

    return {
        "doc_mode": doc_mode,
        "doc_name": doc_meta.get("name", "Unknown Document"),
        "doc_type": doc_meta.get("type", "Unknown Type"),
        "doc_bytes": doc_meta.get("bytes"),
        "doc_text": doc_meta.get("text"),
        "doc_size_display": doc_meta.get("size_display", "0 B"),
        "prompt": clean_prompt,
        "response": response.strip(),
        "regime": regime_key,
        "workflow_mode": workflow_mode,
        "generation_metadata": generation_metadata,
        "is_validated": True,
        "staged_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def _init_session_state() -> None:
    """Initializes persistent session state variables for input controls."""
    if "doc_mode" not in st.session_state:
        st.session_state["doc_mode"] = "File Upload (PDF / TXT)"
    if "raw_doc_text" not in st.session_state:
        st.session_state["raw_doc_text"] = ""
    if "question_text" not in st.session_state:
        st.session_state["question_text"] = ""
    if "response_text" not in st.session_state:
        st.session_state["response_text"] = ""
    if "regime_choice" not in st.session_state:
        st.session_state["regime_choice"] = "Retrieval (Recommended)"
    if "uploader_version" not in st.session_state:
        st.session_state["uploader_version"] = 0
    if "prepared_inputs" not in st.session_state:
        st.session_state["prepared_inputs"] = None
    if "verification_result" not in st.session_state:
        st.session_state["verification_result"] = None
    if "last_verification_time" not in st.session_state:
        st.session_state["last_verification_time"] = None

    # Live Gemini workflow state
    if "workflow_mode" not in st.session_state:
        st.session_state["workflow_mode"] = "Manual Verification"
    if "generated_response_text" not in st.session_state:
        st.session_state["generated_response_text"] = ""
    if "generation_metadata" not in st.session_state:
        st.session_state["generation_metadata"] = None
    if "generation_error" not in st.session_state:
        st.session_state["generation_error"] = None
    if "generation_error_code" not in st.session_state:
        st.session_state["generation_error_code"] = None
    if "generated_for_question" not in st.session_state:
        st.session_state["generated_for_question"] = None
    if "generated_for_doc_name" not in st.session_state:
        st.session_state["generated_for_doc_name"] = None


def render_verify_view(on_navigate_results: Optional[Callable[[], None]] = None) -> None:
    """Renders the functional Verify Response input interface supporting Manual and Live Gemini workflows."""
    _init_session_state()

    render_page_header(
        title="Verify LLM Response",
        subtitle=(
            "Upload a reference document or supply text, provide or generate the candidate LLM response, "
            "and execute document-grounded factual verification using calibrated DeBERTa-v3."
        ),
        badge="Verification Input Setup",
    )

    # Active session notification if prior result exists
    active_res = st.session_state.get("verification_result")
    if active_res is not None:
        res_doc = active_res.document_name
        res_time = st.session_state.get("last_verification_time", "Recent")
        col_banner_txt, col_banner_btn = st.columns([4, 1], gap="medium")
        with col_banner_txt:
            st.markdown(
                f"""
                <div class="summary-box" style="margin-top: 0; margin-bottom: 16px; border-color: #2563EB;">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div>
                            <span style="font-weight: 600; color: #60A5FA;">Active Verification In Session</span>
                            <span style="color: #9CA3AF; font-size: 0.85rem; margin-left: 8px;">• Document: <strong>{res_doc}</strong> ({res_time})</span>
                        </div>
                        <span class="status-pill"><span class="status-dot-active"></span>Result Ready</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_banner_btn:
            st.markdown("<div style='height: 2px;'></div>", unsafe_allow_html=True)
            if st.button("Open Results →", type="secondary", use_container_width=True, key="btn_banner_open_results"):
                if on_navigate_results:
                    on_navigate_results()

    # -------------------------------------------------------------------------
    # Two-Column Input Layout
    # -------------------------------------------------------------------------
    col_doc, col_resp = st.columns([1, 1], gap="large")

    # Document State Holders
    doc_is_valid = False
    doc_error_msg: Optional[str] = None
    doc_meta: Dict[str, Any] = {
        "name": "None",
        "type": "None",
        "size_display": "0 B",
        "bytes": None,
        "text": None,
    }

    # =========================================================================
    # Left Column: Reference Grounding Document & Regime Selection
    # =========================================================================
    with col_doc:
        st.markdown(
            """
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                <h3 style="margin: 0; font-size: 1.1rem; color: #FFFFFF;">1. Reference Grounding Document</h3>
                <span class="status-pill" style="font-size: 0.72rem;">Evidence Ground</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        doc_mode = st.radio(
            "Document Input Mode",
            options=["File Upload (PDF / TXT)", "Direct Text Entry"],
            index=0 if st.session_state["doc_mode"].startswith("File") else 1,
            horizontal=True,
            label_visibility="collapsed",
            key="doc_mode_radio",
        )
        st.session_state["doc_mode"] = doc_mode

        if doc_mode == "File Upload (PDF / TXT)":
            uploader_key = f"doc_file_uploader_{st.session_state['uploader_version']}"
            uploaded_file = st.file_uploader(
                "Upload Reference File",
                type=["pdf", "txt"],
                accept_multiple_files=False,
                help="Upload a PDF or plain text document to serve as the ground-truth reference context.",
                key=uploader_key,
            )

            if uploaded_file is not None:
                file_bytes = uploaded_file.getvalue()
                is_valid, err_msg, meta = validate_document_file(uploaded_file.name, file_bytes)
                doc_is_valid = is_valid
                doc_error_msg = err_msg
                doc_meta = meta

                if not is_valid and err_msg:
                    st.error(err_msg)
                else:
                    format_detail = meta["type"]
                    if meta.get("words"):
                        extra_detail = f"• {meta['words']} words"
                    else:
                        extra_detail = ""

                    ext_label = "PDF" if meta["type"].startswith("PDF") else "TXT"
                    st.markdown(
                        f"""
                        <div class="summary-item" style="margin-top: 10px;">
                            <div style="display: flex; align-items: center; justify-content: space-between;">
                                <div style="font-weight: 600; color: #F3F4F6;">{meta['name']}</div>
                                <span class="status-pill"><span class="status-dot-active"></span>Ready (Valid {ext_label})</span>
                            </div>
                            <div style="font-size: 0.8rem; color: #9CA3AF; margin-top: 4px;">
                                Format: <strong>{format_detail}</strong> {extra_detail} • Size: <strong>{meta['size_display']}</strong>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        else:
            # Direct text entry mode
            raw_text = st.text_area(
                "Paste Reference Content",
                value=st.session_state["raw_doc_text"],
                height=180,
                placeholder="Paste reference document text here (e.g. source article, manual, transcript)...",
                help="Enter ground-truth context directly without uploading a file.",
                key="raw_doc_text_input",
            )
            st.session_state["raw_doc_text"] = raw_text

            is_valid, err_msg, meta = validate_document_text(raw_text)
            doc_is_valid = is_valid
            doc_error_msg = err_msg
            doc_meta = meta

            st.markdown(
                f"<div class='input-counter'>{meta['words']} words • {meta['chars']} characters</div>",
                unsafe_allow_html=True,
            )

            if raw_text and not is_valid:
                st.error("Reference text contains only whitespace. Please enter valid text.")

        # ---------------------------------------------------------------------
        # Regime Selection
        # ---------------------------------------------------------------------
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                <h3 style="margin: 0; font-size: 1.1rem; color: #FFFFFF;">2. Verification Regime</h3>
                <span class="status-pill" style="font-size: 0.72rem;">Decision Policy</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        regime_choice = st.radio(
            "Select Verification Regime",
            options=["Retrieval (Recommended)", "Direct Context"],
            index=0 if st.session_state["regime_choice"].startswith("Retrieval") else 1,
            horizontal=False,
            label_visibility="collapsed",
            key="regime_radio_select",
        )
        st.session_state["regime_choice"] = regime_choice

        if regime_choice.startswith("Retrieval"):
            regime_key = "retrieval"
            st.info(
                "**Retrieval Mode (Recommended):** Indexes document passages into dense vector space "
                "(all-MiniLM-L6-v2) and retrieves candidate evidence per claim. Recommended for multi-page "
                "PDFs and texts >350 words. Enforces the 0.20 similarity floor for safe abstention.",
                icon="ℹ️",
            )
        else:
            regime_key = "direct"
            st.warning(
                "**Direct Context Mode:** Evaluates propositions directly against the entire reference text. "
                "Ideal for short summaries and single-page contexts (<350 words). Note: inputs exceeding "
                "512 tokens will be truncated by the DeBERTa cross-encoder.",
                icon="⚠️",
            )

    # =========================================================================
    # Right Column: Question & Candidate Response Input (Dual Workflows)
    # =========================================================================
    with col_resp:
        st.markdown(
            """
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                <h3 style="margin: 0; font-size: 1.1rem; color: #FFFFFF;">3. Response Provision & Workflow</h3>
                <span class="status-pill" style="font-size: 0.72rem;">Target Evaluation</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        workflow_mode = st.radio(
            "Verification Workflow",
            options=["Manual Verification", "Live Gemini Verification"],
            index=0 if st.session_state["workflow_mode"] == "Manual Verification" else 1,
            horizontal=True,
            help="Select Manual Verification to paste an existing response, or Live Gemini Verification to generate an answer via Gemini API.",
            key="workflow_mode_select",
        )
        st.session_state["workflow_mode"] = workflow_mode

        is_stale = False
        stale_reasons = []

        if workflow_mode == "Manual Verification":
            # Optional Prompt Field
            prompt_text = st.text_input(
                "User Question / Prompt (Optional)",
                value=st.session_state["question_text"],
                placeholder="e.g. What were the key achievements of the Apollo 11 mission?",
                help="Provides conversational context for interpreting candidate output. Not mandatory.",
                key="prompt_text_input_manual",
            )
            st.session_state["question_text"] = prompt_text
            st.caption("Optional context. Factual verification is strictly grounded against the reference document.")

            st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)

            # Candidate Response Text Area
            response_text = st.text_area(
                "Candidate LLM Response (Required)",
                value=st.session_state["response_text"],
                height=200,
                placeholder=(
                    "Paste the LLM-generated answer here...\n\nExample:\n"
                    "The Apollo 11 mission was launched on a Saturn V rocket on July 16, 1969. "
                    "Neil Armstrong and Buzz Aldrin landed the lunar module Eagle on July 20, 1969. "
                    "Michael Collins walked on the Moon with Armstrong while Buzz Aldrin flew the Command Module."
                ),
                help="The multi-sentence text whose atomic claims will be decomposed, verified, and scored.",
                key="candidate_response_input_manual",
            )
            st.session_state["response_text"] = response_text

            resp_is_valid, resp_err_msg, resp_meta = validate_candidate_response(response_text)
            resp_words = resp_meta["words"]
            resp_chars = resp_meta["chars"]

            st.markdown(
                f"<div class='input-counter'>{resp_words} words • {resp_chars} characters</div>",
                unsafe_allow_html=True,
            )

            if response_text and not resp_is_valid and resp_err_msg:
                st.error(resp_err_msg)

        else:
            # Live Gemini Verification Workflow
            is_gemini_avail, active_model_id, status_msg = check_gemini_availability()

            # Privacy and Data Sovereignty Guarantee
            st.markdown(
                """
                <div style="background-color: #0E1420; border: 1px solid #1E3A8A; border-radius: 6px; padding: 10px 14px; margin-bottom: 12px; font-size: 0.82rem; color: #93C5FD; line-height: 1.4;">
                    <strong style="color: #60A5FA;">🔒 Local Privacy Invariant:</strong>
                    The reference document is <strong>never</strong> transmitted to Google Gemini. Only your prompt is sent to generate candidate text. The candidate text is verified locally on CPU by the fine-tuned DeBERTa model.
                </div>
                """,
                unsafe_allow_html=True,
            )

            if not is_gemini_avail:
                st.warning(
                    f"Gemini API is not configured ({status_msg}). "
                    "Please configure GEMINI_API_KEY in your environment or .env file to enable live response generation. "
                    "You can switch to 'Manual Verification' to verify any response without an API key.",
                    icon="⚠️",
                )

            # Provider Status and Model Display
            col_prov_st, col_prov_mdl = st.columns([1, 1], gap="small")
            with col_prov_st:
                dot_class = "status-dot-active" if is_gemini_avail else "status-dot-neutral"
                prov_text = "Connected" if is_gemini_avail else "Not Configured"
                st.markdown(
                    f"""
                    <div style="font-size: 0.82rem; color: #9CA3AF; margin-bottom: 6px;">
                        Provider Status: <span class="status-pill"><span class="{dot_class}"></span>{prov_text}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_prov_mdl:
                st.markdown(
                    f"""
                    <div style="font-size: 0.82rem; color: #9CA3AF; margin-bottom: 6px; text-align: right;">
                        Active Model: <code style="color: #60A5FA;">{active_model_id}</code>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Mandatory Question / Prompt for Live Generation
            prompt_text = st.text_input(
                "User Question / Prompt (Required for Generation)",
                value=st.session_state["question_text"],
                placeholder="e.g. Summarize the landing and crew of Apollo 11 according to the document.",
                help="Prompt sent to Gemini to generate the candidate response.",
                key="prompt_text_input_live",
            )
            st.session_state["question_text"] = prompt_text

            # Generation Action Trigger
            has_generated_text = bool(st.session_state.get("generated_response_text"))
            gen_btn_label = "Regenerate Candidate with Gemini" if has_generated_text else "Generate Candidate with Gemini"

            if st.button(
                gen_btn_label,
                type="secondary",
                use_container_width=True,
                disabled=not is_gemini_avail,
                key="btn_generate_gemini",
            ):
                clean_q = prompt_text.strip()
                if not clean_q:
                    st.error("Please enter a question or prompt before generating a candidate response.")
                else:
                    with st.spinner(f"Requesting candidate response from {active_model_id}..."):
                        gen_result = generate_candidate_response(question=clean_q, model=active_model_id)

                        if gen_result.success:
                            st.session_state["generated_response_text"] = gen_result.response_text
                            st.session_state["response_text"] = gen_result.response_text
                            st.session_state["generation_metadata"] = {
                                "model_id": gen_result.model_id,
                                "execution_time_seconds": gen_result.execution_time_seconds,
                                "prompt_tokens": gen_result.prompt_tokens,
                                "candidates_tokens": gen_result.candidates_tokens,
                                "total_tokens": gen_result.total_tokens,
                                "finish_reason": gen_result.finish_reason,
                                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                            }
                            st.session_state["generated_for_question"] = clean_q
                            st.session_state["generated_for_doc_name"] = doc_meta.get("name", "Unknown Document")
                            st.session_state["generation_error"] = None
                            st.session_state["generation_error_code"] = None
                            st.rerun()
                        else:
                            err_msg = gen_result.error.message if gen_result.error else "Unknown generation error"
                            st.session_state["generation_error"] = err_msg
                            st.session_state["generation_error_code"] = gen_result.error.error_code if gen_result.error else None

            if st.session_state.get("generation_error"):
                err_code = st.session_state.get("generation_error_code")
                if err_code == "SERVICE_UNAVAILABLE":
                    st.warning(
                        f"**Gemini Model Temporarily Overloaded (HTTP 503):** {st.session_state['generation_error']}\n\n"
                        "The upstream Gemini model is currently experiencing high demand. Spikes in demand are usually temporary. "
                        "You can wait a moment and click **Regenerate Candidate with Gemini**, or switch to **Manual Verification** "
                        "to paste your response and verify without delay.",
                        icon="⏳",
                    )
                else:
                    st.error(f"Generation Error: {st.session_state['generation_error']}")

            # Stale-Response Guard Evaluation
            if has_generated_text:
                gen_q = st.session_state.get("generated_for_question") or ""
                gen_doc = st.session_state.get("generated_for_doc_name") or ""
                curr_q = prompt_text.strip()
                curr_doc = doc_meta.get("name", "Unknown Document")

                if gen_q != curr_q:
                    is_stale = True
                    stale_reasons.append("Prompt was edited after generation")
                if gen_doc != curr_doc:
                    is_stale = True
                    stale_reasons.append(f"Document changed from '{gen_doc}' to '{curr_doc}'")

            if is_stale:
                reasons_txt = ", ".join(stale_reasons)
                st.warning(
                    f"**Out-of-Sync Response Detected ({reasons_txt}):** "
                    "The current response below was generated for prior inputs. Verification is locked to prevent "
                    "verifying mismatched content. Click **Regenerate Candidate with Gemini** or switch to "
                    "**Manual Verification** to proceed.",
                    icon="⚠️",
                )
            else:
                gen_meta = st.session_state.get("generation_metadata")
                if gen_meta:
                    lat = gen_meta.get("execution_time_seconds", 0.0)
                    mod = gen_meta.get("model_id", "Gemini")
                    tot_tok = gen_meta.get("total_tokens")
                    tok_info = f" • {tot_tok} tokens" if tot_tok else ""
                    gen_at = gen_meta.get("generated_at", "")
                    st.markdown(
                        f"""
                        <div style="background-color: #0E1420; border: 1px solid #1F2937; border-radius: 6px; padding: 8px 12px; margin-top: 8px; margin-bottom: 8px; font-size: 0.8rem; color: #9CA3AF;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span>Generated by <strong style="color: #60A5FA;">{mod}</strong> ({lat:.2f}s{tok_info})</span>
                                <span class="status-pill"><span class="status-dot-active"></span>Ready to Review</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Review & Edit Candidate Response Text Area
            st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
            response_text = st.text_area(
                "Candidate LLM Response (Review & Edit Before Verification)",
                value=st.session_state["response_text"],
                height=200,
                placeholder="Generated response will appear here. You can inspect and edit it before verification...",
                help="Review and edit the generated response. The exact text here will be decomposed and verified.",
                key="candidate_response_input_live",
            )
            st.session_state["response_text"] = response_text

            resp_is_valid, resp_err_msg, resp_meta = validate_candidate_response(response_text)
            resp_words = resp_meta["words"]
            resp_chars = resp_meta["chars"]

            st.markdown(
                f"<div class='input-counter'>{resp_words} words • {resp_chars} characters</div>",
                unsafe_allow_html=True,
            )
            st.caption("✏️ **Review View:** You may freely edit the response above. The exact text in this box will be verified by the local pipeline.")

            if response_text and not resp_is_valid and resp_err_msg:
                st.error(resp_err_msg)

    # =========================================================================
    # Full Width: Compact Input Summary & Actions
    # =========================================================================
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Determine overall readiness
    inputs_ready = doc_is_valid and resp_is_valid and not is_stale
    if workflow_mode == "Live Gemini Verification" and not prompt_text.strip():
        inputs_ready = False

    st.markdown(
        f"""
        <div class="summary-box">
            <div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #1F2937; padding-bottom: 10px; margin-bottom: 12px;">
                <span style="font-weight: 600; color: #FFFFFF; font-size: 0.95rem;">Input Staging Summary</span>
                <span class="status-pill">
                    <span class="{'status-dot-active' if inputs_ready else 'status-dot-neutral'}"></span>
                    <span>{'Ready for Verification' if inputs_ready else 'Awaiting Required Inputs'}</span>
                </span>
            </div>
            <div class="summary-grid">
                <div class="summary-item">
                    <div class="summary-item-label">Reference Document</div>
                    <div class="summary-item-value">{doc_meta['name'] if doc_is_valid else 'Not Provided'}</div>
                    <div style="font-size: 0.78rem; color: #9CA3AF; margin-top: 2px;">
                        {'Type: ' + doc_meta['type'] + ' (' + doc_meta['size_display'] + ')' if doc_is_valid else 'Requires valid PDF or text'}
                    </div>
                </div>
                <div class="summary-item">
                    <div class="summary-item-label">Candidate Response</div>
                    <div class="summary-item-value">{f"{resp_words} words ({resp_chars} chars)" if resp_is_valid else 'Not Provided'}</div>
                    <div style="font-size: 0.78rem; color: #9CA3AF; margin-top: 2px;">
                        {'⚠️ Out-of-sync' if is_stale else (f"Workflow: {workflow_mode}" if resp_is_valid else 'Requires non-empty response text')}
                    </div>
                </div>
                <div class="summary-item">
                    <div class="summary-item-label">Verification Regime</div>
                    <div class="summary-item-value">{'Retrieval Augmented' if regime_key == 'retrieval' else 'Direct Context'}</div>
                    <div style="font-size: 0.78rem; color: #9CA3AF; margin-top: 2px;">
                        {'&tau;floor = 0.20' if regime_key == 'retrieval' else 'Direct 512-token limit'}
                    </div>
                </div>
                <div class="summary-item">
                    <div class="summary-item-label">Prompt Context</div>
                    <div class="summary-item-value">{f"'{prompt_text[:28]}...'" if prompt_text.strip() else 'None (Optional)'}</div>
                    <div style="font-size: 0.78rem; color: #9CA3AF; margin-top: 2px;">
                        {'Prompt attached' if prompt_text.strip() else ('Required for Live Gemini' if workflow_mode == 'Live Gemini Verification' else 'Optional for Manual')}
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # Verification Action Buttons
    # -------------------------------------------------------------------------
    col_act_run, col_act_clear = st.columns([3, 1], gap="medium")

    with col_act_run:
        if st.button("Verify Response", type="primary", use_container_width=True):
            # Strict Validation
            if not doc_is_valid:
                st.error("Validation Error: A valid reference document (PDF or text entry) is required.")
            elif not resp_is_valid:
                st.error("Validation Error: Candidate LLM response cannot be empty or whitespace.")
            elif workflow_mode == "Live Gemini Verification" and not prompt_text.strip():
                st.error("Validation Error: A user question or prompt is required for the Live Gemini Verification workflow.")
            elif workflow_mode == "Live Gemini Verification" and is_stale:
                st.error(
                    "Validation Error: Cannot verify a stale response. The prompt or reference document was modified after generation. "
                    "Please click 'Regenerate Candidate with Gemini' or switch to 'Manual Verification'."
                )
            else:
                # Stage inputs in session state
                staged_payload = build_staged_payload(
                    doc_mode=doc_mode,
                    doc_meta=doc_meta,
                    prompt=prompt_text,
                    response=response_text,
                    regime=regime_key,
                    workflow_mode=workflow_mode,
                    generation_metadata=st.session_state.get("generation_metadata") if workflow_mode == "Live Gemini Verification" else None,
                )
                st.session_state["prepared_inputs"] = staged_payload

                # Execute real VeriLLMPipeline
                with st.spinner("Executing calibrated factual verification with fine-tuned DeBERTa-v3..."):
                    try:
                        result = execute_verification(staged_payload)
                        st.session_state["verification_result"] = result
                        st.session_state["last_verification_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    except Exception as exc:
                        st.error(f"Verification Pipeline Error: {exc}")
                        return

                gfi = result.score.indices.grounded_factuality_index
                gfi_display = f"{gfi:.1f}%" if gfi is not None else "N/A"
                supp_cnt = result.score.counts.supported_count
                fact_cnt = result.score.counts.factual_claims
                contra_cnt = result.score.counts.contradicted_count

                st.success(
                    f"Factual verification completed in {result.execution_time_seconds:.2f}s! "
                    f"Grounded Factuality Index: {gfi_display} ({supp_cnt}/{fact_cnt} claims supported, {contra_cnt} contradicted)."
                )

                # Show operational warnings if any
                for w in result.warnings:
                    st.warning(w, icon="⚠️")

                col_res_btn, col_res_exp = st.columns([1, 1], gap="medium")
                with col_res_btn:
                    if st.button("View Full Results & Analytics →", type="primary", use_container_width=True, key="btn_view_full_results"):
                        if on_navigate_results:
                            on_navigate_results()

                with col_res_exp:
                    with st.expander("Inspect Staged Input & Execution Metadata", expanded=False):
                        st.json({
                            "document_name": result.document_name,
                            "regime": result.regime,
                            "workflow_mode": staged_payload.get("workflow_mode"),
                            "execution_time_seconds": result.execution_time_seconds,
                            "total_claims": result.score.counts.total_claims,
                            "factual_claims": result.score.counts.factual_claims,
                            "supported": result.score.counts.supported_count,
                            "contradicted": result.score.counts.contradicted_count,
                            "unsupported": result.score.counts.unsupported_count,
                            "unverifiable": result.score.counts.unverifiable_count,
                            "grounded_factuality_index": result.score.indices.grounded_factuality_index,
                            "contradiction_penalized_factuality": result.score.indices.contradiction_penalized_factuality,
                            "warnings_count": len(result.warnings),
                            "timestamp": st.session_state["last_verification_time"],
                        })

    with col_act_clear:
        if st.button("Clear Inputs", type="secondary", use_container_width=True):
            st.session_state["raw_doc_text"] = ""
            st.session_state["question_text"] = ""
            st.session_state["response_text"] = ""
            st.session_state["generated_response_text"] = ""
            st.session_state["generation_metadata"] = None
            st.session_state["generation_error"] = None
            st.session_state["generated_for_question"] = None
            st.session_state["generated_for_doc_name"] = None
            st.session_state["prepared_inputs"] = None
            st.session_state["verification_result"] = None
            st.session_state["last_verification_time"] = None
            st.session_state["uploader_version"] = st.session_state.get("uploader_version", 0) + 1
            st.rerun()
