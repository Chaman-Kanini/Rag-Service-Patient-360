# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spacy model if needed
RUN python -m spacy download en_core_web_sm || true

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p app/rag_data/pdfs app/rag_data/output app/rag_data/qna_logs app/rag_data/chromadb && \
    mkdir -p rag_data/icd10_faiss rag_data/cpt_faiss

# Expose port (Railway will set PORT env variable)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Run the application with uvicorn
# Railway provides PORT env variable, defaulting to 8000 if not set
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
