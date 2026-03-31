from fastapi import APIRouter, HTTPException
from pathlib import Path
import json
import os
from datetime import datetime
from app.config import OUTPUT_DIR, PDF_DIR

router = APIRouter(prefix="/api/patient", tags=["patient"])



@router.get("/rag-batches")
async def get_rag_batches():
    """
    Get all RAG batches with their metadata.
    Returns a list of all processed batches from the output directory.
    """
    try:
        batches = []
        output_path = Path(OUTPUT_DIR)
        
        if not output_path.exists():
            return {"batches": []}
        
        # Iterate through all batch directories
        for batch_dir in output_path.iterdir():
            if batch_dir.is_dir() and batch_dir.name.startswith("batch_"):
                batch_id = batch_dir.name.replace("batch_", "")
                
                # Look for the consolidated output file
                output_file = batch_dir / "clinical_consolidated_output.json"
                
                if output_file.exists():
                    try:
                        # Get file metadata
                        stat = output_file.stat()
                        
                        batch_info = {
                            "id": batch_id,
                            "name": batch_dir.name,
                            "createdAt": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                            "lastModified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                            "size": stat.st_size,
                            "filePath": str(output_file)
                        }
                        
                        batches.append(batch_info)
                    except Exception as e:
                        print(f"Error processing batch {batch_id}: {str(e)}")
                        continue
        
        # Sort by last modified date (newest first)
        batches.sort(key=lambda x: x["lastModified"], reverse=True)
        
        return {"batches": batches}
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching RAG batches: {str(e)}"
        )


@router.get("/rag-batches/{batch_id}")
async def get_rag_batch_data(batch_id: str):
    """
    Get the consolidated data for a specific batch.
    """
    try:
        output_path = Path(OUTPUT_DIR) / f"batch_{batch_id}" / "clinical_consolidated_output.json"
        
        if not output_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Batch {batch_id} not found"
            )
        
        # Read the JSON file
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Clean up any markdown formatting
            cleaned = content.replace("```json", "").replace("```", "").strip()
            data = json.loads(cleaned)
        
        # Count source PDFs for this batch
        pdf_batch_path = Path(PDF_DIR) / f"batch_{batch_id}"
        pdf_count = 0
        if pdf_batch_path.exists():
            for f in pdf_batch_path.iterdir():
                if f.is_file() and f.suffix.lower() in ['.pdf', '.doc', '.docx']:
                    pdf_count += 1
        
        return {
            "success": True,
            "batch_id": batch_id,
            "data": data,
            "pdf_count": pdf_count
        }
        
    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Invalid JSON in batch file: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching batch data: {str(e)}"
        )


@router.get("/rag-batches/{batch_id}/pdfs")
async def get_rag_batch_pdfs(batch_id: str):
    """
    Get list of PDF files for a specific batch.
    """
    try:
        pdf_path = Path(PDF_DIR) / f"batch_{batch_id}"
        
        if not pdf_path.exists():
            return {
                "success": True,
                "batch_id": batch_id,
                "pdfs": []
            }
        
        pdfs = []
        for pdf_file in pdf_path.iterdir():
            if pdf_file.is_file() and pdf_file.suffix.lower() in ['.pdf', '.doc', '.docx']:
                stat = pdf_file.stat()
                pdfs.append({
                    "name": pdf_file.name,
                    "size": stat.st_size,
                    "uploadedAt": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "path": str(pdf_file)
                })
        
        # Sort by name
        pdfs.sort(key=lambda x: x["name"])
        
        return {
            "success": True,
            "batch_id": batch_id,
            "pdfs": pdfs
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching PDF files: {str(e)}"
        )


@router.get("/rag-batches/{batch_id}/pdfs/{file_name}")
async def get_pdf_file(batch_id: str, file_name: str, download: bool = False):
    """
    Get a specific PDF file from a batch.
    """
    try:
        from fastapi.responses import FileResponse
        
        pdf_path = Path(PDF_DIR) / f"batch_{batch_id}" / file_name
        
        if not pdf_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"PDF file {file_name} not found in batch {batch_id}"
            )
        
        media_type = "application/pdf"
        if pdf_path.suffix.lower() == '.doc':
            media_type = "application/msword"
        elif pdf_path.suffix.lower() == '.docx':
            media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        
        if download:
            return FileResponse(
                path=str(pdf_path),
                media_type=media_type,
                filename=file_name,
                headers={"Content-Disposition": f"attachment; filename=\"{file_name}\""}
            )
        else:
            return FileResponse(
                path=str(pdf_path),
                media_type=media_type,
                headers={"Content-Disposition": f"inline; filename=\"{file_name}\""}
            )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching PDF file: {str(e)}"
        )


