import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent
RAG_DATA_DIR = BASE_DIR / "rag_data"

# RAG Data subdirectories
PDF_DIR = RAG_DATA_DIR / "pdfs"
OUTPUT_DIR = RAG_DATA_DIR / "output"
QNA_DIR = RAG_DATA_DIR / "qna_logs"

# Vector Store Configuration (using FAISS)
# Note: Variable name kept as CHROMA_PERSIST_DIR for backward compatibility
CHROMA_PERSIST_DIR = RAG_DATA_DIR / "chromadb"

# Ensure directories exist
for d in [PDF_DIR, OUTPUT_DIR, QNA_DIR, CHROMA_PERSIST_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# RAG Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 16

# Azure OpenAI Configuration (for LLM)
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "xxx")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "xxx")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
CHATGPT_MODEL = os.getenv("CHATGPT_MODEL", "gpt-5-mini")

# Azure OpenAI Configuration (for Embeddings)
AZURE_EMBEDDING_ENDPOINT = os.getenv("AZURE_EMBEDDING_ENDPOINT", "https://health-rag-proj.cognitiveservices.azure.com/")
AZURE_EMBEDDING_KEY = os.getenv("AZURE_EMBEDDING_KEY", "")
AZURE_EMBEDDING_API_VERSION = os.getenv("AZURE_EMBEDDING_API_VERSION", "2024-02-01")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

# ICD-10 Vector Index Configuration
ICD10_FAISS_DIR = Path(os.getenv("ICD10_FAISS_DIR", str(BASE_DIR.parent / "rag_data" / "icd10_faiss")))
ICD10_BATCH_ID = os.getenv("ICD10_BATCH_ID", "icd10_fy2026")
ICD10_TOP_K = int(os.getenv("ICD10_TOP_K", "10"))
ICD10_MAX_DIAGNOSES = int(os.getenv("ICD10_MAX_DIAGNOSES", "10"))

# CPT Vector Index Configuration
CPT_FAISS_DIR = Path(os.getenv("CPT_FAISS_DIR", str(BASE_DIR.parent / "rag_data" / "cpt_faiss")))
CPT_BATCH_ID = os.getenv("CPT_BATCH_ID", "cpt_2026")
CPT_TOP_K = int(os.getenv("CPT_TOP_K", "10"))
CPT_MAX_PROCEDURES = int(os.getenv("CPT_MAX_PROCEDURES", "10"))

# API Settings
API_TITLE = "RAG Pipeline API"
API_DESCRIPTION = "Clinical Document RAG Pipeline with Entity Extraction and Q&A"
API_VERSION = "1.0.0"
