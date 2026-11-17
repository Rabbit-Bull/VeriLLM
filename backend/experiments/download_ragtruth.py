"""
VeriLLM Phase 7: RAGTruth Dataset Acquisition Script.
Downloads the official ParticleMedia/RAGTruth repository archive and extracts
the dataset directory (`source_info.jsonl` and `response.jsonl`) to `backend/data/ragtruth/`.
"""

import sys
import os
import zipfile
import urllib.request
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data" / "ragtruth"
ZIP_PATH = BACKEND_DIR / "data" / "ragtruth_main.zip"
REPO_ZIP_URL = "https://github.com/ParticleMedia/RAGTruth/archive/refs/heads/main.zip"


def download_and_extract():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    source_file = DATA_DIR / "source_info.jsonl"
    response_file = DATA_DIR / "response.jsonl"

    if source_file.exists() and response_file.exists():
        print(f"[OK] RAGTruth dataset already present at: {DATA_DIR}")
        print(f"  - source_info.jsonl: {source_file.stat().st_size / (1024*1024):.2f} MB")
        print(f"  - response.jsonl: {response_file.stat().st_size / (1024*1024):.2f} MB")
        return

    print(f"Downloading RAGTruth archive from {REPO_ZIP_URL}...")
    headers = {"User-Agent": "VeriLLM-Research/1.0"}
    req = urllib.request.Request(REPO_ZIP_URL, headers=headers)
    
    with urllib.request.urlopen(req, timeout=60) as resp, open(ZIP_PATH, "wb") as out_f:
        total_bytes = 0
        while True:
            chunk = resp.read(1024 * 64)
            if not chunk:
                break
            out_f.write(chunk)
            total_bytes += len(chunk)
            print(f"\rDownloaded {total_bytes / (1024*1024):.2f} MB...", end="", flush=True)

    print(f"\nDownload complete. Extracting dataset files to {DATA_DIR}...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        for member in z.namelist():
            # Extract files under dataset/ or data/
            if "dataset/" in member and not member.endswith("/"):
                filename = Path(member).name
                target_path = DATA_DIR / filename
                with z.open(member) as src, open(target_path, "wb") as dst:
                    dst.write(src.read())
                print(f"  -> Extracted: {filename} ({target_path.stat().st_size / (1024*1024):.2f} MB)")

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
        print(f"Cleaned up temporary archive {ZIP_PATH.name}.")

    print("[Success] RAGTruth dataset acquired successfully.")


if __name__ == "__main__":
    download_and_extract()

