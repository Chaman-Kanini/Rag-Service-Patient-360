import json
import os
from pathlib import Path
from typing import Any, Optional

from app.config import ICD10_BATCH_ID, ICD10_FAISS_DIR, ICD10_MAX_DIAGNOSES, ICD10_TOP_K
from app.services.chroma_service import ChromaVectorStore
from app.utils.embeddings import get_embedding
from app.utils.llm import call_llm


class ICD10LookupService:
    _store: Optional[ChromaVectorStore] = None

    @classmethod
    def _get_store(cls) -> Optional[ChromaVectorStore]:
        if cls._store is not None:
            return cls._store

        if not ICD10_FAISS_DIR.exists():
            return None

        cls._store = ChromaVectorStore(persist_directory=str(ICD10_FAISS_DIR))
        return cls._store

    @staticmethod
    def _clean_json(text: str) -> str:
        return text.replace("```json", "").replace("```", "").strip()

    @classmethod
    def retrieve_candidates(cls, diagnosis: str, top_k: int = ICD10_TOP_K) -> list[dict[str, Any]]:
        store = cls._get_store()
        if store is None:
            return []

        query_embedding = get_embedding(diagnosis)
        chunk_ids, chunk_texts, metadatas, distances = store.similarity_search(
            batch_id=ICD10_BATCH_ID,
            query_embedding=query_embedding,
            top_k=top_k,
        )

        candidates: list[dict[str, Any]] = []
        for chunk_id, text, meta, dist in zip(chunk_ids, chunk_texts, metadatas, distances):
            candidates.append(
                {
                    "chunk_id": chunk_id,
                    "code": meta.get("code"),
                    "short_description": meta.get("short_description"),
                    "long_description": meta.get("long_description"),
                    "distance": dist,
                }
            )

        return candidates

    @classmethod
    def select_best_code(cls, diagnosis: str, candidates: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if not candidates:
            return None

        candidates_json = json.dumps(
            [
                {
                    "code": c.get("code"),
                    "short_description": c.get("short_description"),
                    "long_description": c.get("long_description"),
                }
                for c in candidates
                if c.get("code")
            ],
            ensure_ascii=False,
        )

        prompt = (
            "You are a medical coding assistant. Given a diagnosis, pick the best matching ICD-10 code from the provided candidates. "
            "Do not invent codes. Only return a code present in the candidate list. "
            "Return VALID JSON only.\n\n"
            f"DIAGNOSIS:\n{diagnosis}\n\n"
            f"CANDIDATES (JSON):\n{candidates_json}\n\n"
            "OUTPUT JSON SCHEMA:\n"
            "{\n"
            '  "icd10_code": "<code from candidates>",\n'
            '  "description": "<best matching description>",\n'
            '  "confidence": 0.0\n'
            "}"
        )

        raw = call_llm(prompt)
        cleaned = cls._clean_json(raw)
        try:
            data = json.loads(cleaned)
        except Exception:
            return None

        code = data.get("icd10_code")
        if not code:
            return None

        allowed = {c.get("code") for c in candidates}
        if code not in allowed:
            return None

        return {
            "diagnosis": diagnosis,
            "icd10_code": code,
            "description": data.get("description"),
            "confidence": data.get("confidence"),
            "candidates": candidates,
        }

    @classmethod
    def assign_icd10_codes(cls, diagnoses: list[str]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []

        for diagnosis in diagnoses[:ICD10_MAX_DIAGNOSES]:
            d = (diagnosis or "").strip()
            if not d:
                continue

            candidates = cls.retrieve_candidates(d)
            best = cls.select_best_code(d, candidates)
            if best is not None:
                results.append(best)

        return results
