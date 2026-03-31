from fastapi import APIRouter, HTTPException, Query
from app.services.icd10_service import ICD10LookupService
from app.services.cpt_service import CPTLookupService

router = APIRouter(prefix="/api/codes", tags=["codes"])


@router.get("/search")
async def search_codes(
    query: str = Query(..., min_length=1, description="Diagnosis or procedure description to search"),
    type: str = Query(..., pattern="^(icd10|cpt)$", description="Code type: 'icd10' or 'cpt'"),
    top_k: int = Query(15, ge=1, le=30, description="Number of results to return"),
):
    """
    Search ICD-10 or CPT codes by description using vector similarity.
    Returns top_k candidate codes without LLM re-ranking.
    """
    try:
        trimmed = query.strip()
        if not trimmed:
            raise HTTPException(status_code=400, detail="Query must not be empty")

        results = []

        if type == "icd10":
            candidates = ICD10LookupService.retrieve_candidates(trimmed, top_k=top_k)
            for c in candidates:
                results.append({
                    "code": c.get("code", ""),
                    "description": c.get("long_description") or c.get("short_description") or "",
                    "short_description": c.get("short_description", ""),
                    "distance": c.get("distance"),
                })
        elif type == "cpt":
            candidates = CPTLookupService.retrieve_candidates(trimmed, top_k=top_k)
            for c in candidates:
                results.append({
                    "code": c.get("code", ""),
                    "description": c.get("description") or "",
                    "distance": c.get("distance"),
                })

        return {
            "success": True,
            "data": results,
            "query": trimmed,
            "type": type,
            "count": len(results),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error searching codes: {str(e)}",
        )
