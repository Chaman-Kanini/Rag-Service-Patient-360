from fastapi import APIRouter, HTTPException
from app.models.schemas import QuestionRequest, QuestionResponse
from app.services.rag_pipeline import RagPipelineService
from datetime import datetime

router = APIRouter(prefix="/api/qa", tags=["qa"])


@router.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question and get an answer based on indexed documents.
    """
    try:
        if not request.question.strip():
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )

        # Check if embeddings exist
        embeddings = RagPipelineService.load_embeddings(request.batch_id)
        if not embeddings:
            raise HTTPException(
                status_code=404,
                detail=f"No indexed documents found for batch {request.batch_id}"
            )

        # Answer the question
        top_k = request.top_k if request.top_k else 10
        answer, log_file = RagPipelineService.answer_question(
            request.question,
            request.batch_id,
            top_k
        )

        if not answer:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate answer"
            )

        return QuestionResponse(
            question=request.question,
            answer=answer,
            log_file=log_file,
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error answering question: {str(e)}"
        )
