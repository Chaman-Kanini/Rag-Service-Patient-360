# Python 3.11 Virtual Environment Setup

This project uses a Python 3.11 virtual environment (`venv311`) to ensure compatibility with PyPDFLoader and its dependencies (spacy, langchain-community).

## Why Python 3.11?

Python 3.14 has compatibility issues with spacy's pydantic v1 dependencies. The Python 3.11 virtual environment provides a stable environment where PyPDFLoader works correctly.

## Starting the Server

### Option 1: Using the batch script (Windows)
```bash
start_server.bat
```

### Option 2: Using PowerShell
```powershell
.\start_server.ps1
```

### Option 3: Manual start
```bash
.\venv311\Scripts\python.exe -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Virtual Environment Details

- **Location**: `./venv311/`
- **Python Version**: 3.11.4
- **Key Dependencies**:
  - langchain-community (with PyPDFLoader)
  - spacy 3.8.11
  - pydantic 2.12.5
  - fastapi, uvicorn, openai, tiktoken, etc.

## Reinstalling Dependencies

If you need to reinstall dependencies:

```bash
.\venv311\Scripts\python.exe -m pip install -r requirements.txt
```

## Testing PyPDFLoader

To verify PyPDFLoader is working:

```bash
.\venv311\Scripts\python.exe -c "from app.utils.text_processing import extract_text_from_pdf; print('PyPDFLoader test passed')"
```
