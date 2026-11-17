"""
VeriLLM Phase 1: Environment & Dependency Import Verification Script.
Tests all foundational machine learning, backend, and utility packages.
"""

import sys
import importlib

def check_module(module_name: str, display_name: str = None) -> tuple[bool, str]:
    if display_name is None:
        display_name = module_name
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", "installed")
        return True, f"{display_name} ({version})"
    except Exception as e:
        return False, f"{display_name} FAILED: {str(e)}"

def run_all_checks():
    print("=" * 60)
    print("VeriLLM Phase 1: Dependency Verification")
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version: {sys.version.split()[0]}")
    print("=" * 60)

    modules_to_test = [
        ("torch", "PyTorch"),
        ("transformers", "Hugging Face Transformers"),
        ("datasets", "Hugging Face Datasets"),
        ("sentence_transformers", "Sentence Transformers"),
        ("sklearn", "scikit-learn"),
        ("numpy", "NumPy"),
        ("pandas", "pandas"),
        ("spacy", "spaCy"),
        ("fastapi", "FastAPI"),
        ("pydantic", "Pydantic"),
        ("uvicorn", "Uvicorn"),
        ("dotenv", "python-dotenv"),
        ("pymongo", "PyMongo"),
        ("pypdf", "pypdf"),
        ("multipart", "python-multipart"),
        ("minicheck", "MiniCheck (UT Austin Research Repo)"),
    ]

    all_passed = True
    results = []

    for mod_name, disp_name in modules_to_test:
        success, msg = check_module(mod_name, disp_name)
        if success:
            results.append((True, f"[PASS] {msg}"))
        else:
            all_passed = False
            results.append((False, f"[FAIL] {msg}"))

    # Specific MiniCheck class check
    try:
        from minicheck.minicheck import MiniCheck
        results.append((True, f"[PASS] MiniCheck Class (minicheck.minicheck.MiniCheck)"))
    except Exception as e:
        all_passed = False
        results.append((False, f"[FAIL] MiniCheck Class Import: {e}"))

    for _, res in results:
        print(res)

    print("=" * 60)

    # Detailed torch hardware check
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        print(f"PyTorch Device Status: CUDA Available = {cuda_avail}")
        if cuda_avail:
            print(f"  Device Name: {torch.cuda.get_device_name(0)}")
        else:
            print("  Running in CPU mode (suitable for local testing & small checkpoints)")
    except Exception as e:
        print(f"PyTorch hardware check failed: {e}")

    print("=" * 60)

    if all_passed:
        print("ALL CRITICAL IMPORTS PASSED SUCCESSFULLY!")
        return 0
    else:
        print("SOME IMPORTS FAILED. Please review the errors above.")
        return 1

def test_all_imports():
    assert run_all_checks() == 0

if __name__ == "__main__":
    sys.exit(run_all_checks())

