Write-Host "Starting RAG API server with Python 3.11 virtual environment..." -ForegroundColor Green
.\venv311\Scripts\python.exe -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
