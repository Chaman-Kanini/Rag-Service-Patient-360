# RAG API - FastAPI Server

This is a FastAPI implementation of the RAG (Retrieval-Augmented Generation) pipeline for clinical document processing.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables (optional):
Create a `.env` file in this directory:
```
GEMINI_API_KEY=your_api_key_here
```

## Running the Server

From this directory:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

### Interactive API Documentation
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### Health
- `GET /health` - Health check

### Batch Processing
- `POST /api/batch/process` - Process a batch of PDFs
- `GET /api/batch/status/{batch_id}` - Get batch status

### Q&A
- `POST /api/qa/ask` - Ask a question about indexed documents

## Example Usage

### 1. Process a batch of PDFs
```bash
curl -X POST "http://localhost:8000/api/batch/process" \
  -H "Content-Type: application/json" \
  -d '{"batch_id": "my_batch"}'
```

### 2. Ask a question
```bash
curl -X POST "http://localhost:8000/api/qa/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What diagnoses are mentioned?", "batch_id": "my_batch"}'
```

### 3. Check batch status
```bash
curl -X GET "http://localhost:8000/api/batch/status/my_batch"
```

## Data Organization

All data is organized in the `rag_data` directory:
- `rag_data/pdfs/` - Input PDF files
- `rag_data/chunks/` - Chunked text from PDFs
- `rag_data/embeddings/` - Generated embeddings (numpy arrays)
- `rag_data/output/` - Consolidated clinical JSON output
- `rag_data/qna_logs/` - Question and answer logs
- `rag_data/similarity/` - Similarity scores (for future use)

## Architecture

- **FastAPI**: Web framework
- **Pydantic**: Data validation
- **Gemini API**: Embeddings and LLM
- **pdfplumber**: PDF text extraction
- **tiktoken**: Token counting for proper chunking
