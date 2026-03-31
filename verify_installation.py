#!/usr/bin/env python
"""
Quick installation verification script
Run this to verify all dependencies are properly installed
"""

import sys

print("=" * 60)
print("FastAPI RAG - Installation Verification")
print("=" * 60)
print()

# Check Python version
print(f"Python Version: {sys.version}")
print()

# Check installed packages
packages_to_check = [
    "fastapi",
    "uvicorn",
    "pydantic",
    "pypdf",
    "numpy",
    "tiktoken",
    "openai",
    "huggingface_hub",
    "dotenv",
    "python_multipart",
    "faiss",
    "langchain_community",
    "langchain_core",
    "spacy",
    "pydantic_settings",
]

all_ok = True
print("Checking installed packages:")
print("-" * 60)

for package in packages_to_check:
    try:
        __import__(package)
        print(f"[OK]   {package:<30} OK")
    except ImportError as e:
        print(f"[FAIL] {package:<30} FAILED - {str(e)}")
        all_ok = False

print()
print("=" * 60)

if all_ok:
    print("All dependencies are installed!")
    print()
    print("Next steps:")
    print("  1. Start the server: python main.py")
    print("  2. Visit: http://localhost:8000/docs")
    print("  3. Test endpoints in the Swagger UI")
    sys.exit(0)
else:
    print("Some dependencies are missing!")
    print()
    print("To fix this, run:")
    print("  pip install -r requirements.txt")
    sys.exit(1)
