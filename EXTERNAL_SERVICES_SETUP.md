# VeriLLM: External Services & API Configuration Guide

**Version:** 1.0.0 (Production Release)  
**Date:** November 2026  
**Status:** Authoritative Configuration Reference

> [!CAUTION]
> **CRITICAL SECURITY REQUIREMENT:** Never commit, write, or paste real API keys, tokens, or credentials into version-controlled files, commit messages, or public issue trackers. All credentials must reside strictly in local `.env` files that are ignored by Git.

---

## 1. Architectural Role of External Services

VeriLLM is engineered as a **local-first, document-grounded verification system**. 

- **Core Verification Engine:** Executes **100% offline on your local CPU**. Document ingestion, text parsing, sliding-window chunking, dense semantic retrieval, proposition extraction, DeBERTa cross-encoder neural inference, probability calibration, and factuality scoring require **zero external API keys, zero cloud subscriptions, and zero internet connectivity**.
- **Optional Candidate Generation:** The application provides an optional convenience adapter connecting to Google's Gemini API (`gemini-3.8-flash`). This allows users to generate candidate AI responses directly within the Streamlit interface (in Tab 2 of the *Verify Response* page) before submitting them to the local verification pipeline.

```text
+-------------------------------------------------------------------------------+
| VERILLM LOCAL WORKSTATION                                                     |
|                                                                               |
|  [ Manual Verification Mode (Tab 1) ]  ---> 100% OFFLINE (No API Key Needed)  |
|                                                                               |
|  [ Live Gemini Mode (Tab 2) ]          ---> Requires GEMINI_API_KEY           |
|                                             (Sends prompt question only;     |
|                                              Reference doc NEVER leaves PC)   |
+-------------------------------------------------------------------------------+
```

---

## 2. Services Configuration Matrix

| Service | Architectural Role | Required for Verification? | Required for Candidate Generation? | Environment Variable | Status |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **Local ML Engine** | Document parsing, retrieval, calibrated DeBERTa inference, scoring | **YES** | N/A | None (Runs locally on CPU) | Built-in |
| **Google Gemini API** | Optional candidate response generation (Tab 2) | **NO** | **YES** | `GEMINI_API_KEY` | Optional |
| **Hugging Face Hub** | Public model checkpoint distribution (`Rabbit-Bull/verillm-deberta`) | **First Run Only** (if not locally cached) | **NO** | None required (Public repo; `HF_HUB_OFFLINE` for offline gating) | Automatic |

*(Note: Earlier preliminary plans considered MongoDB Atlas for remote multi-user persistence and OpenAI API for generation. These have been superseded by Streamlit session state and Google Gen AI SDK integration; no MongoDB or OpenAI accounts are required).*

---

## 3. Configuring the Optional Google Gemini API

If you wish to use the **Live Gemini Verification** workflow in Tab 2, follow these steps to configure your API key securely.

### Step 1: Obtain a Gemini API Key
1. Visit Google AI Studio at: [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey).
2. Sign in with your Google account.
3. Click **Create API key** and copy the generated key.

### Step 2: Create Your Local Environment File
In the `backend/` directory (or the project root), copy the provided example environment template:

```powershell
# From the project root (D:\drive e\VeriLLM)
cp backend/.env.example backend/.env
```

### Step 3: Add Your API Key to `.env`
Open `backend/.env` in your text editor and add your key:

```bash
# Application Settings
ENVIRONMENT=development
LOG_LEVEL=INFO
PORT=8501
HOST=127.0.0.1

# Optional Google Gemini API Key for Live Candidate Generation (Tab 2)
GEMINI_API_KEY="your-actual-gemini-api-key-here"
```

Save the file. VeriLLM automatically loads this variable using `python-dotenv`.

> [!IMPORTANT]
> The `.gitignore` file in the repository root explicitly ignores all `.env` files (`.env`, `backend/.env`, `*.env`). Verify that `git status` does not show your `.env` file before making any Git commits.

---

## 4. Known Upstream API Limitations & Resilience Design

During Phase 11 integration diagnostics, Google's preview endpoint for `gemini-3.8-flash` demonstrated intermittent capacity constraints. VeriLLM implements defensive engineering patterns to handle these upstream conditions gracefully:

### 1. HTTP 503 UNAVAILABLE (High Demand Spikes)
- **Upstream Diagnostic:** Google's API front-end (`scaffolding on HTTPServer2`) occasionally returns HTTP 503 with the diagnostic message:
  ```text
  This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.
  ```
- **Built-in Bounded Retries:** The `GeminiClient` in `backend/ml_engine/llm_generator/client.py` automatically retries the request up to **2 times** with exponential backoff (`INITIAL_RETRY_BACKOFF_SECONDS = 1.0`, `BACKOFF_MULTIPLIER = 2.0`).
- **UI Advisory Banner:** If the service remains unavailable after retries, the Streamlit UI catches `GeminiServiceUnavailableError` and displays an amber advisory banner explaining that Google's preview servers are experiencing high load, advising the user to proceed with **Manual Verification**.

### 2. HTTP 504 Gateway Timeouts
- Configured with a generous client-side timeout of **90.0 seconds** (`DEFAULT_TIMEOUT_SECONDS = 90.0`) to accommodate peak network latency.
- If a timeout occurs, `GeminiTimeoutError` is raised, preventing the Streamlit interface from hanging or freezing.

### 3. Strict Data Sovereignty Guarantee
- Under all circumstances, the prompt payload sent to Google's API contains **only the user's question string**.
- The reference document uploaded in Step 1 is completely excluded from the API request payload, ensuring zero document leakage to cloud servers.

---

## 5. Hugging Face Model Hub Integration

VeriLLM utilizes the fine-tuned DeBERTa-v3 cross-encoder hosted on the Hugging Face Model Hub for document-grounded proposition verification.

### Repository Details
- **Repository URL:** [https://huggingface.co/Rabbit-Bull/verillm-deberta](https://huggingface.co/Rabbit-Bull/verillm-deberta)
- **Pinned Git Revision:** `71259e48c794e89f92075ddf848813c93327461c`
- **Total Download Size:** Approximately 283 MB
- **Local Cache Directory:** `backend/models/verillm_deberta/`
- **Required Artifacts:**
  1. `config.json`
  2. `model.safetensors`
  3. `tokenizer.json`
  4. `tokenizer_config.json`

### Public Access & Authentication
- The model repository is completely **public**.
- **No Hugging Face token or account is required** for the default download.
- VeriLLM never requires `HF_TOKEN` for standard operation.

### First-Run Internet Requirement vs. Subsequent Offline Operation
- **First Run:** If the four required model files are not present in `backend/models/verillm_deberta/`, VeriLLM automatically connects to Hugging Face Hub on first startup or verification to retrieve the pinned checkpoint files.
- **Subsequent Runs:** Once downloaded into `backend/models/verillm_deberta/`, all model loading occurs locally from disk. Subsequent verification sessions run **100% offline** on local CPU hardware with zero network traffic.

### Disconnected & Offline Environments
For strictly disconnected environments where network access is disabled:
1. Ensure the four required files are pre-placed in `backend/models/verillm_deberta/`.
2. You may explicitly set standard offline environment variables in your environment or `.env`:
   ```bash
   HF_HUB_OFFLINE=1
   TRANSFORMERS_OFFLINE=1
   ```
> [!IMPORTANT]
> The model files must be fully downloaded and present in `backend/models/verillm_deberta/` before setting `HF_HUB_OFFLINE=1`. If offline mode is enabled while files are missing, VeriLLM will safely abort with a `RuntimeError` rather than attempting a network connection.

### Independence from Gemini Configuration
Hugging Face model checkpoint retrieval is completely independent of Google Gemini API configuration:
- Neither Hugging Face credentials nor Gemini API keys are needed for ordinary document-grounded verification.
- A Gemini API key is strictly optional, used only if candidate AI text is generated inside Tab 2.

---

## 6. Verification Commands

### Check Gemini API Key Configuration
To verify that your `GEMINI_API_KEY` is recognized by the local virtual environment without launching the UI:

```powershell
cd "D:\drive e\VeriLLM"
.\.venv\Scripts\python.exe -c "
import os, dotenv
dotenv.load_dotenv('backend/.env')
key = os.getenv('GEMINI_API_KEY')
print('GEMINI_API_KEY configured:', bool(key and len(key) > 10))
"
```

If configured correctly, this command will print:
`GEMINI_API_KEY configured: True`
*(without revealing your actual key)*.

### Check Local Model Integrity
To verify that the fine-tuned DeBERTa model files are present locally and intact:

```powershell
cd "D:\drive e\VeriLLM"
.\.venv\Scripts\python.exe -c "
from pathlib import Path
model_dir = Path('backend/models/verillm_deberta')
required = ['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json']
missing = [f for f in required if not (model_dir / f).is_file()]
print('Local model complete:', len(missing) == 0, 'Missing:', missing if missing else 'None')
"
```
