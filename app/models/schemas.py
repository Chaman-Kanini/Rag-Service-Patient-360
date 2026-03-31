from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime


class BatchUploadResponse(BaseModel):
    batch_id: str
    status: str
    message: str
    output_file: Optional[str] = None
    timestamp: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
 

class ProcessBatchRequest(BaseModel):
    batch_id: str


class ProcessBatchResponse(BaseModel):
    message: str
    batch_id: str
    output_file: str
    timestamp: str


class QuestionRequest(BaseModel):
    question: str
    batch_id: Optional[str] = None
    top_k: Optional[int] = 10


class QuestionResponse(BaseModel):
    question: str
    answer: str
    log_file: str
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    message: str
    version: str


class BatchStatusResponse(BaseModel):
    batch_id: str
    pdf_count: int
    chunk_count: int
    embedding_count: int
    has_output: bool
    has_qna_logs: bool
