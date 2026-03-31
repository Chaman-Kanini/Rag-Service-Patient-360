from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from app.services.rag_pipeline import RagPipelineService

router = APIRouter(prefix="/api/patient/chatbot", tags=["chatbot"])


class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: str


class ChatbotRequest(BaseModel):
    question: str
    batchId: Optional[str] = None


class SourceDocument(BaseModel):
    name: str
    fileName: str
    chunks_used: int


class ChatbotResponse(BaseModel):
    success: bool
    answer: str
    timestamp: str
    sourceDocuments: Optional[List[SourceDocument]] = []
    conversationId: Optional[str] = None


class ChatHistoryResponse(BaseModel):
    messages: List[ChatMessage]
    batchId: Optional[str] = None


@router.post("/ask", response_model=ChatbotResponse)
async def ask_chatbot(request: ChatbotRequest):
    try:
        if not request.question.strip():
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )

        embeddings = RagPipelineService.load_embeddings(request.batchId)
        if not embeddings:
            raise HTTPException(
                status_code=404,
                detail=f"No indexed documents found for batch {request.batchId or 'default'}"
            )

        answer, log_file, source_docs = RagPipelineService.answer_question(
            request.question,
            request.batchId,
            top_k=10
        )

        if not answer:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate answer"
            )

        return ChatbotResponse(
            success=True,
            answer=answer,
            timestamp=datetime.now().isoformat(),
            sourceDocuments=source_docs
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chatbot request: {str(e)}"
        )


@router.get("/history", response_model=ChatHistoryResponse)
async def get_chat_history(batchId: Optional[str] = None):
    try:
        return ChatHistoryResponse(
            messages=[],
            batchId=batchId
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching chat history: {str(e)}"
        )
