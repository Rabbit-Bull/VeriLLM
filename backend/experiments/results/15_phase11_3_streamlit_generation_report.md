# VeriLLM Phase 11 — Step 11.3: Streamlit Live Gemini Generation Interface & Review View

**Date:** November 14, 2026  
**Status:** Completed & Validated  
**Module:** `frontend/views/verify.py`, `frontend/views/results.py`, `frontend/utils/generation_service.py`, `backend/tests/test_generation_service.py`  
**Test Suite:** 94 passing tests (14 generation service, 20 verify UI, 23 results UI, 12 workflow integration, 14 gemini client, 11 pipeline service)

---

## 1. Executive Summary & Overview of Implemented Features

Phase 11.3 introduces live LLM response generation into the VeriLLM Streamlit application while strictly maintaining the architectural boundaries established in Phase 11.1 and the Gemini provider adapter built in Phase 11.2. 

The primary enhancements include:
1. **Dual Verification Workflows:** Cleanly partitioned into **Manual Verification** (upload document + paste response) and **Live Gemini Verification** (upload document + enter prompt + generate answer with Gemini + review/edit + verify).
2. **Data Sovereignty & Local Privacy Invariant:** The grounding reference document is strictly retained on local CPU memory. Only the user prompt is transmitted to the external Google Gemini API.
3. **Editable Review View:** Following live generation, the candidate text is presented in an interactive review textarea where users can inspect, modify, or refine propositions prior to factual verification.
4. **Stale-Response Guard:** If a user edits the prompt or swaps reference documents after generating a response, the system locks verification and surfaces an actionable warning to prevent verifying mismatched content.
5. **Full Pipeline Convergence:** Both workflows stage identical normalized data structures and converge on the exact same calibrated `VeriLLMPipeline.verify()` engine and visual analytics dashboard.
6. **Graceful Degradation:** If `GEMINI_API_KEY` is absent or unconfigured, the application alerts the user cleanly while keeping the Manual Verification workflow 100% operational.

---

## 2. User Workflow Comparison: Manual vs. Live Gemini

| Dimension | Manual Verification Workflow | Live Gemini Verification Workflow |
| :--- | :--- | :--- |
| **User Objective** | Validate existing AI or human-authored text against source material. | Ask a question, generate an answer live with Gemini, and verify its accuracy. |
| **Reference Document** | Upload PDF/TXT or paste text into direct entry area. | Upload PDF/TXT or paste text into direct entry area (**never sent to Gemini**). |
| **User Prompt** | Optional context for verification interpretation. | **Mandatory** query string sent to Google Gemini for response generation. |
| **Response Provision** | User manually types or pastes candidate response. | Gemini API generates candidate text; placed in editable review area. |
| **Review & Edit** | Standard text area editing. | User can freely inspect, review, and edit the generated response before verification. |
| **Stale Detection** | Not applicable (response is manually staged). | Active guard: detects prompt or document changes post-generation and locks verification. |
| **Verification Execution** | Local `VeriLLMPipeline.verify()` via DeBERTa-v3 cross-encoder. | Local `VeriLLMPipeline.verify()` via DeBERTa-v3 cross-encoder. |
| **API Key Requirement** | None (100% local CPU). | Requires `GEMINI_API_KEY` in environment or `.env` file. |

---

## 3. UI Component Design and Review View Implementation

### 3.1 Consistent Visual Design System
The new UI components adhere to the established VeriLLM design system:
- **Palette:** Dark Navy (`#0B0F17`), Surface Charcoal (`#111827`, `#0E1420`), Card Borders (`#1F2937`), Slate Blue Accent (`#3B82F6`), Success Emerald (`#10B981`), Warning Amber (`#F59E0B`), Danger Red (`#EF4444`).
- **Typography:** System sans-serif with negative letter spacing on titles and tabular font metrics for numeric indicators.

### 3.2 Dual Workflow Selector
Located at the top of Section 3 in the Right Column:
```python
workflow_mode = st.radio(
    "Verification Workflow",
    options=["Manual Verification", "Live Gemini Verification"],
    index=0 if st.session_state["workflow_mode"] == "Manual Verification" else 1,
    horizontal=True,
    help="Select Manual Verification to paste an existing response, or Live Gemini Verification to generate an answer via Gemini API.",
    key="workflow_mode_select",
)
```

### 3.3 Provider Status & Privacy Banner (Live Mode)
When Live Gemini mode is active:
- **Privacy Notice:** Informs the user that the grounding document remains local on CPU and is never transmitted to Google Gemini.
- **Provider Status Indicator:** Shows `Connected` (green dot) or `Not Configured` (neutral dot).
- **Active Model Badge:** Displays the active model identifier (e.g. `gemini-3.8-flash` or `gemini-2.5-flash`).

### 3.4 Generation Trigger & Editable Review View
- **Action Button:** "Generate Candidate with Gemini" (or "Regenerate Candidate with Gemini" once text is present).
- **Spinner:** Displays `Generating candidate response with <model_id>...` during generation.
- **Metadata Pill:** Displays generation latency (seconds), token counts (prompt, candidate, total), and finish reason upon completion.
- **Editable Text Area:** Renders the generated text in a textarea with character and word counters, allowing the user to make real-time corrections.
- **Review Caption:** Reminds the user that the exact text currently inside the review textarea will be evaluated by DeBERTa-v3.

### 3.5 Provenance Banner in Results View
The Results view (`frontend/views/results.py`) displays the full execution lineage:
- Reference document name and verification regime (`Retrieval Augmented` or `Direct Context`).
- Workflow mode (`Manual Verification` or `Live Gemini Verification`).
- Model provenance and generation duration when applicable (e.g. `Generated by gemini-3.8-flash (0.85s)`).

---

## 4. API Key Safety and Privacy Boundary Enforcement

### 4.1 Secret Value Invariants
- **Zero Exposure:** `GEMINI_API_KEY` is loaded strictly through `os.environ` or loaded from `.env` via `python-dotenv`.
- **No Leaks:** The secret value is never printed in logs, rendered in HTML attributes, exposed in Streamlit session state, or committed to Git (`.env` is verified in `.gitignore`).
- **Availability Masking:** The `check_gemini_availability()` utility inspects `adapter.is_available` and returns only a boolean, model ID, and high-level status message (`Ready • Model: ...`).

### 4.2 Architectural Data Sovereignty
- **Document Boundary:** `generate_candidate_response(question, model, adapter)` accepts **only** the user prompt. Its signature and execution path completely omit reference document bytes, text, and metadata.
- **Adjudication Independence:** Gemini acts purely as a generative candidate producer. Factual correctness, claim extraction, evidence passage matching, and hallucination classification are exclusively performed by the local fine-tuned DeBERTa-v3 model with the frozen C2 calibration bundle.

---

## 5. Verification Handoff and Stale-Response Guard Mechanism

### 5.1 Staged Payload Normalization
When the user clicks **Verify Response**, `build_staged_payload()` constructs the canonical staging dictionary:
```python
{
    "doc_mode": doc_mode,
    "doc_name": doc_meta.get("name"),
    "doc_type": doc_meta.get("type"),
    "doc_bytes": doc_meta.get("bytes"),
    "doc_text": doc_meta.get("text"),
    "doc_size_display": doc_meta.get("size_display"),
    "prompt": clean_prompt,
    "response": response.strip(),
    "regime": regime_key,
    "workflow_mode": workflow_mode,
    "generation_metadata": generation_metadata,
    "is_validated": True,
    "staged_at": "YYYY-MM-DD HH:MM:SS",
}
```
This dictionary maps through `map_staged_inputs_to_pipeline_args()` into the exact arguments expected by `VeriLLMPipeline.verify()`.

### 5.2 Stale-Response Detection Logic
To prevent accidental verification of out-of-sync responses:
```python
is_stale = False
if has_generated_text:
    gen_q = st.session_state.get("generated_for_question") or ""
    gen_doc = st.session_state.get("generated_for_doc_name") or ""
    curr_q = prompt_text.strip()
    curr_doc = doc_meta.get("name", "Unknown Document")

    if gen_q != curr_q:
        is_stale = True
    if gen_doc != curr_doc:
        is_stale = True
```
When `is_stale` is `True`:
1. An amber warning alert informs the user that prompt or document inputs have changed.
2. The Input Staging Summary marks the candidate response as `⚠️ Out-of-sync`.
3. Clicking **Verify Response** is blocked with an explicit validation error directing the user to click **Regenerate Candidate with Gemini** or switch to **Manual Verification**.

---

## 6. Test Coverage and Validation Results

### 6.1 Automated Test Execution Summary
All tests were executed using the dedicated virtual environment `backend/.venv`:

```
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.2.0, pluggy-1.7.0

backend/tests/test_generation_service.py ........ (14 passed)
backend/tests/test_frontend_verify.py .......... (20 passed)
backend/tests/test_frontend_results.py ......... (23 passed)
backend/tests/test_frontend_workflows.py ....... (12 passed)
backend/tests/test_gemini_client.py ............ (14 passed)
backend/tests/test_pipeline_service.py ......... (11 passed)

======================= 94 passed, 2 warnings in 143.43s =======================
```

### 6.2 Key Assertions Verified
- **Provider Availability:** Correctly detects configured vs. unconfigured keys without exposing secret values.
- **Offline Safety:** All tests mock Gemini API interactions; zero live network calls occur during testing.
- **Empty Query Guard:** Rejects empty or whitespace questions locally before any API attempt.
- **Data Sovereignty:** Asserts signature of `generate_candidate_response` contains no document parameters.
- **Review View Editing:** Verifies that edited responses are preserved in the staged payload.
- **Stale Detection:** Verifies prompt and document change detection flags.
- **Workflow Convergence:** Verifies that both Manual and Live payloads map to identical arguments for `pipeline.verify()`.
- **Zero Regressions:** 100% pass rate across existing frontend results, verify, workflow, and pipeline suites.

---

## 7. Instructions for Running the Streamlit Application Locally

To run the complete VeriLLM Streamlit application with dual verification workflows:

### Step 1: Verify Environment & API Key
Ensure `.env` exists in the project root containing your Gemini API key (or set it in your environment):
```powershell
# Optional: verify in PowerShell that GEMINI_API_KEY is present
$env:GEMINI_API_KEY
```

### Step 2: Launch Streamlit
Execute Streamlit using the dedicated virtual environment:
```powershell
& 'backend\.venv\Scripts\streamlit.exe' run frontend/app.py
```

### Step 3: Test Both Workflows
1. **Manual Verification:**
   - In sidebar, navigate to **Verify Response**.
   - Upload a PDF or TXT file (or paste plain text in Direct Text Entry).
   - Keep workflow set to **Manual Verification**.
   - Paste a multi-sentence candidate response.
   - Click **Verify Response** to execute local calibrated verification.
2. **Live Gemini Verification:**
   - In Section 3, switch the workflow radio to **Live Gemini Verification**.
   - Note the connected status badge and local privacy invariant banner.
   - Enter a query in the **User Question / Prompt** field.
   - Click **Generate Candidate with Gemini**.
   - Review and optionally edit the generated response in the review area.
   - Click **Verify Response** to run factual verification against your reference document.
   - Click **View Full Results & Analytics →** to inspect the KPI dashboard, proposition composition, calibrated probability chart, and claim cards with evidence citations.

