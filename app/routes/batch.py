from fastapi import APIRouter, HTTPException, UploadFile, File
from app.models.schemas import (
    ProcessBatchRequest, ProcessBatchResponse, BatchStatusResponse, BatchUploadResponse
)
from app.services.rag_pipeline import RagPipelineService
from datetime import datetime
import json
from pathlib import Path
import time
import os
 
router = APIRouter(prefix="/api/batch", tags=["batch"])
 
 
@router.post("/upload", response_model=BatchUploadResponse)
async def upload_batch(files: list[UploadFile] = File(...)):
    """
    Upload PDF files, create a new batch, and automatically process them.
    Returns the batch_id, status, and consolidated output file path.
    """
    try:
        if not files or len(files) == 0:
            raise HTTPException(
                status_code=400,
                detail="No files provided"
            )
 
        # Create batch from uploaded files and auto-process
        batch_id, saved_count, output_file = RagPipelineService.create_batch_from_uploads(files)
 
        if saved_count == 0:
            raise HTTPException(
                status_code=400,
                detail="No valid document files were uploaded. Supported formats: .pdf, .doc, .docx"
            )
 
        message = f"Uploaded and processed {saved_count} file(s)"
       
        status = "Processed" if output_file else "Uploaded"
       
        # Read consolidated output file if it exists
        data = None
        if output_file:
            # Normalize the path for better cross-platform compatibility
            normalized_path = os.path.normpath(output_file)
            output_path = Path(normalized_path)
           
            print(f"DEBUG: Output file path: {output_file}")
            print(f"DEBUG: Normalized path: {normalized_path}")
            print(f"DEBUG: Path exists: {output_path.exists()}")
           
            # Wait a brief moment to ensure file is written
            if not output_path.exists():
                time.sleep(0.5)
           
            print(f"DEBUG: Path exists after wait: {output_path.exists()}")
           
            if output_path.exists() and output_path.is_file():
                try:
                    # Read using standard open() with the normalized path
                    with open(normalized_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        cleaned = content.replace("```json", "").replace("```", "").strip()
                        print(f"DEBUG: File size: {len(content)} bytes")
                        data = json.loads(cleaned)
                    print(f"DEBUG: Successfully loaded JSON data with {len(str(data))} characters")
                except json.JSONDecodeError as e:
                    print(f"WARNING: Invalid JSON in {output_file}: {str(e)}")
                except Exception as e:
                    print(f"WARNING: Could not read output file {output_file}: {type(e).__name__}: {str(e)}")
            else:
                print(f"WARNING: Output file path is not a valid file: {output_path}")
 
        return BatchUploadResponse(
            batch_id=batch_id,
            status=status,
            message=message,
            output_file=output_file,
            timestamp=datetime.now().isoformat(),
            data=data
        )
 
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading and processing files: {str(e)}"
        )
 
 
@router.post("/process", response_model=ProcessBatchResponse)
async def process_batch(request: ProcessBatchRequest):
    """
    Process a batch of PDFs: ingest, chunk, embed, extract, and consolidate.
    """
    try:
        # Ingest PDFs and create chunks/embeddings
        all_chunks, all_embeddings, chunk_count = RagPipelineService.ingest_pdfs(
            request.batch_id
        )
 
        if chunk_count == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No PDFs found for batch {request.batch_id}"
            )
 
        # Extract and consolidate
        output_file = RagPipelineService.extract_and_consolidate(
            request.batch_id, all_chunks, all_embeddings
        )
 
        if not output_file:
            raise HTTPException(
                status_code=500,
                detail="Failed to extract and consolidate entities"
            )
 
        return ProcessBatchResponse(
            message="Batch processed successfully",
            batch_id=request.batch_id,
            output_file=output_file,
            timestamp=datetime.now().isoformat()
        )
 
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing batch: {str(e)}"
        )
 
 
@router.get("/status/{batch_id}", response_model=BatchStatusResponse)
async def get_batch_status(batch_id: str):
    """
    Get the status of a batch.
    """
    try:
        status = RagPipelineService.get_batch_status(batch_id)
        return BatchStatusResponse(**status)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting batch status: {str(e)}"
        )


@router.post("/finalize-codes/{batch_id}")
async def finalize_codes(batch_id: str, codes: dict):
    """
    Save finalized ICD-10 and CPT codes for a batch.
    Creates a new folder structure: rag_data/finalized/{batch_id}/finalized_codes.json
    """
    try:
        # Normalize batch_id
        normalized_batch_id = batch_id if batch_id.startswith("batch_") else f"batch_{batch_id}"
        
        # Get the finalized directory path
        base_path = Path(__file__).parent.parent / "rag_data" / "finalized" / normalized_batch_id
        base_path.mkdir(parents=True, exist_ok=True)
        
        finalized_file = base_path / "finalized_codes.json"
        
        # Structure the finalized codes data
        finalized_data = {
            "batch_id": normalized_batch_id,
            "finalized_at": datetime.now().isoformat(),
            "icd10_codes": codes.get("icd10", []),
            "cpt_codes": codes.get("cpt", []),
            "summary": {
                "total_icd10": len(codes.get("icd10", [])),
                "total_cpt": len(codes.get("cpt", [])),
                "ai_suggested_accepted": len([c for c in codes.get("icd10", []) + codes.get("cpt", []) if c.get("isAISuggested") and c.get("isAccepted")]),
                "manually_added": len([c for c in codes.get("icd10", []) + codes.get("cpt", []) if not c.get("isAISuggested")])
            }
        }
        
        # Write to file
        with open(finalized_file, 'w', encoding='utf-8') as f:
            json.dump(finalized_data, f, indent=2, ensure_ascii=False)
        
        print(f"Finalized codes saved to: {finalized_file}")
        
        return {
            "success": True,
            "message": "Codes finalized successfully",
            "batch_id": normalized_batch_id,
            "file_path": str(finalized_file),
            "summary": finalized_data["summary"]
        }
        
    except Exception as e:
        print(f"Error finalizing codes: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error finalizing codes: {str(e)}"
        )


@router.get("/finalized-codes/{batch_id}")
async def get_finalized_codes(batch_id: str):
    """
    Get finalized codes for a batch if they exist.
    """
    try:
        # Normalize batch_id
        normalized_batch_id = batch_id if batch_id.startswith("batch_") else f"batch_{batch_id}"
        
        finalized_file = Path(__file__).parent.parent / "rag_data" / "finalized" / normalized_batch_id / "finalized_codes.json"
        
        if not finalized_file.exists():
            return {
                "success": False,
                "exists": False,
                "message": "No finalized codes found for this batch"
            }
        
        with open(finalized_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return {
            "success": True,
            "exists": True,
            "data": data
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving finalized codes: {str(e)}"
        )