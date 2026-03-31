import os
import re
import uuid
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from app.config import (
    PDF_DIR, OUTPUT_DIR, QNA_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
)
from app.utils.text_processing import extract_text_from_document, tokenize_and_chunk
from app.utils.embeddings import get_embedding, cosine_similarity
from app.utils.llm import call_llm
from app.utils.prompts import (
    CLINICAL_MASTER_PROMPT, QNA_PROMPT_TEMPLATE
)
from app.services.chroma_service import ChromaVectorStore
from app.services.icd10_service import ICD10LookupService
from app.services.cpt_service import CPTLookupService


class RagPipelineService:
    """Service for managing RAG pipeline operations."""
    
    # Initialize ChromaDB vector store
    _vector_store = None
    
    @classmethod
    def get_vector_store(cls) -> ChromaVectorStore:
        """Get or create the ChromaDB vector store instance."""
        if cls._vector_store is None:
            cls._vector_store = ChromaVectorStore()
        return cls._vector_store

    @staticmethod
    def create_batch_from_uploads(files: list) -> tuple:
        """
        Create a new batch from uploaded files and automatically process them.
        Returns: (batch_id, saved_count, output_file_path)
        """
        # Generate unique batch ID
        batch_id = str(uuid.uuid4())[:8]
        
        pdf_dir = RagPipelineService.get_batch_dir(batch_id, 'pdf')
        
        saved_count = 0
        failed_files = []
        
        print(f"Creating batch: {batch_id}")
        print(f"Saving {len(files)} files to {pdf_dir}")
        
        for file in files:
            try:
                if file.filename.lower().endswith(('.pdf', '.doc', '.docx')):
                    # Save file to batch directory
                    file_path = Path(pdf_dir) / file.filename
                    
                    # Read file content and save
                    content = file.file.read()
                    with open(file_path, 'wb') as f:
                        f.write(content)
                    
                    saved_count += 1
                    print(f"Saved: {file.filename}")
                else:
                    failed_files.append(file.filename)
                    print(f"Skipped (unsupported file type): {file.filename}")
            except Exception as e:
                failed_files.append(file.filename)
                print(f"Error saving {file.filename}: {str(e)}")
        
        if saved_count == 0:
            return batch_id, saved_count, None
        
        # Automatically process the batch
        print(f"\n{'='*60}")
        print(f"AUTO-PROCESSING BATCH {batch_id}")
        print(f"{'='*60}")
        
        try:
            # Step 1: Ingest PDFs and create chunks/embeddings
            all_chunks, all_embeddings, chunk_count = RagPipelineService.ingest_pdfs(batch_id)
            
            # Step 2: Extract and consolidate
            output_file = RagPipelineService.extract_and_consolidate(batch_id, all_chunks, all_embeddings)
            
            print(f"{'='*60}")
            print(f"BATCH {batch_id} PROCESSING COMPLETE")
            print(f"Output: {output_file}")
            print(f"{'='*60}\n")
            
            return batch_id, saved_count, output_file
        except Exception as e:
            print(f"Error auto-processing batch {batch_id}: {str(e)}")
            return batch_id, saved_count, None

    @staticmethod
    def get_batch_dir(batch_id: str, dir_type: str = 'pdf'):
        """Get the batch-specific directory for a given type."""
        base_dirs = {
            'pdf': PDF_DIR,
            'output': OUTPUT_DIR,
            'qna': QNA_DIR
        }
        base_dir = base_dirs.get(dir_type, PDF_DIR)
        
        if batch_id:
            batch_dir = base_dir / f"batch_{batch_id}"
        else:
            batch_dir = base_dir
        
        batch_dir.mkdir(parents=True, exist_ok=True)
        return batch_dir

    @staticmethod
    def ingest_pdfs(batch_id: str = None) -> tuple:
        """
        Ingest all PDFs from the PDF directory and create chunks and embeddings.
        Stores chunks and embeddings in ChromaDB.
        Returns: (all_chunks, all_embeddings, chunk_count)
        """
        pdf_dir = RagPipelineService.get_batch_dir(batch_id, 'pdf')
        vector_store = RagPipelineService.get_vector_store()

        all_chunks = []
        all_embeddings = []
        chunk_count = 0

        print(f"Ingesting PDFs from {pdf_dir}...")

        supported_extensions = ['*.pdf', '*.doc', '*.docx']
        doc_files = []
        for ext in supported_extensions:
            doc_files.extend(Path(pdf_dir).glob(ext))

        for doc_file in doc_files:
            print(f"Processing {doc_file.name}")
            try:
                text = extract_text_from_document(str(doc_file))
                chunks = tokenize_and_chunk(text, CHUNK_SIZE, CHUNK_OVERLAP)

                for idx, chunk in enumerate(chunks):
                    chunk_id = f"{doc_file.stem}_chunk_{idx}"

                    # Generate embedding
                    try:
                        embedding = get_embedding(chunk)
                        
                        # Prepare metadata
                        metadata = {
                            "document_name": doc_file.name,
                            "document_stem": doc_file.stem,
                            "chunk_index": idx,
                            "batch_id": batch_id or "default",
                            "chunk_size": len(chunk),
                            "total_chunks": len(chunks)
                        }
                        
                        # Store in ChromaDB
                        vector_store.add_single_chunk(
                            batch_id=batch_id or "default",
                            chunk_id=chunk_id,
                            chunk_text=chunk,
                            embedding=embedding,
                            metadata=metadata
                        )

                        all_chunks.append((chunk_id, chunk))
                        all_embeddings.append((chunk_id, embedding))
                        chunk_count += 1
                    except Exception as e:
                        print(f"Error embedding chunk {chunk_id}: {str(e)}")

            except Exception as e:
                print(f"Error processing document {doc_file.name}: {str(e)}")

        print(f"Total chunks indexed: {chunk_count}")
        return all_chunks, all_embeddings, chunk_count

    @staticmethod
    def load_embeddings(batch_id: str = None) -> list:
        """Check if batch exists in ChromaDB by getting chunk count."""
        vector_store = RagPipelineService.get_vector_store()
        try:
            chunk_count = vector_store.get_chunk_count(batch_id or "default")
            # Return a non-empty list if chunks exist (for backward compatibility with qa.py check)
            return [True] if chunk_count > 0 else []
        except Exception as e:
            print(f"Error checking embeddings for batch {batch_id}: {str(e)}")
            return []

    @staticmethod
    def load_chunks(batch_id: str = None) -> list:
        """Load all chunks from ChromaDB."""
        vector_store = RagPipelineService.get_vector_store()
        chunk_ids, chunk_texts, metadatas = vector_store.get_all_chunks(batch_id or "default")
        
        # Return in the same format as before: list of tuples (chunk_id, chunk_text)
        all_chunks = list(zip(chunk_ids, chunk_texts))
        return all_chunks

    @staticmethod
    def extract_and_consolidate(batch_id: str = None, all_chunks: list = None, all_embeddings: list = None) -> str:
        """
        Extract and consolidate clinical information using ALL chunks + single LLM call.
        Uses all chunks ordered by document and chunk index for comprehensive extraction.
        Returns: output file path
        """
        vector_store = RagPipelineService.get_vector_store()
        output_dir = RagPipelineService.get_batch_dir(batch_id, 'output')

        print(f"Running entity extraction + consolidation (similarity search, TOP_K={TOP_K})...")

        # Count source documents from metadata
        source_doc_names = set()
        try:
            _, _, all_metadatas = vector_store.get_all_chunks(batch_id or "default")
            for m in (all_metadatas or []):
                doc_name = m.get('document_name', '')
                if doc_name:
                    source_doc_names.add(doc_name)
        except Exception:
            pass
        source_document_count = max(len(source_doc_names), 1)

        # Use cosine similarity search to retrieve the most relevant chunks
        try:
            # Create a clinical extraction query embedding for similarity search
            clinical_query = (
                "patient demographics diagnoses medications allergies procedures "
                "lab results vital signs clinical history treatment plan "
                "medical conditions symptoms assessment"
            )
            query_embedding = get_embedding(clinical_query)

            # Perform similarity search to get top-k most relevant chunks
            chunk_ids, chunk_texts, metadatas, distances = vector_store.similarity_search(
                batch_id=batch_id or "default",
                query_embedding=query_embedding,
                top_k=TOP_K
            )

            if not chunk_texts:
                print(f"No chunks found for batch {batch_id}")
                return None

            print(f"Retrieved top {len(chunk_texts)} chunks (of TOP_K={TOP_K}) via cosine similarity search")
        except Exception as e:
            print(f"Error retrieving chunks via similarity search: {str(e)}")
            return None

        # Build context from retrieved chunks, ordered by document and chunk index
        # Sort chunks by document name and chunk index for coherent reading order
        indexed_chunks = []
        for i, (cid, text) in enumerate(zip(chunk_ids, chunk_texts)):
            meta = metadatas[i] if i < len(metadatas) else {}
            doc_name = meta.get('document_name', '')
            chunk_idx = meta.get('chunk_index', i)
            indexed_chunks.append((doc_name, chunk_idx, text))
        indexed_chunks.sort(key=lambda x: (x[0], x[1]))
        context = "\n\n".join(c[2] for c in indexed_chunks)

        try:
            # SINGLE STEP: Extract + consolidate in one LLM call
            print("Calling LLM for extraction + consolidation (single prompt)...")
            master_prompt = f"""
{CLINICAL_MASTER_PROMPT}

====================
SOURCE DOCUMENTS: {', '.join(source_doc_names) if source_doc_names else 'Unknown'}
DOCUMENT COUNT: {source_document_count}
====================

====================
DOCUMENT TEXT
====================
{context}
"""
            result = call_llm(master_prompt, timeout=900)
            cleaned = result.replace("```json", "").replace("```", "").strip()

            try:
                consolidated = json.loads(cleaned)
                print(f"Extraction + consolidation successful: {len(str(consolidated))} chars")
            except Exception as e:
                print(f"WARNING: Could not parse JSON: {str(e)}")
                print(f"Raw result (first 500 chars): {cleaned[:500]}")
                json_match = re.search(r'\{[\s\S]*\}', cleaned)
                if json_match:
                    try:
                        consolidated = json.loads(json_match.group())
                        print("Recovered JSON from response")
                    except Exception:
                        print("ERROR: Could not recover JSON")
                        return None
                else:
                    print("ERROR: No JSON found in response")
                    return None

            # Add source document metadata
            consolidated["_source_document_count"] = source_document_count
            consolidated["_source_document_names"] = list(source_doc_names)

            # Enrich with ICD-10 and CPT codes
            diagnoses = []
            for d in consolidated.get("diagnoses", []) if isinstance(consolidated, dict) else []:
                if isinstance(d, dict):
                    dx = (d.get("diagnosis") or "").strip()
                    if dx:
                        diagnoses.append(dx)

            procedures = []
            for p in consolidated.get("procedures", []) if isinstance(consolidated, dict) else []:
                if isinstance(p, dict):
                    proc = (p.get("procedure") or "").strip()
                    if proc:
                        procedures.append(proc)

            if diagnoses:
                print("Assigning ICD-10 codes...")
                icd10_results = ICD10LookupService.assign_icd10_codes(diagnoses)
                if isinstance(consolidated, dict):
                    consolidated["icd10_codes"] = icd10_results

            if procedures:
                print("Assigning CPT codes...")
                cpt_results = CPTLookupService.assign_cpt_codes(procedures)
                if isinstance(consolidated, dict):
                    consolidated["cpt_codes"] = cpt_results

            # Write final enriched JSON
            enriched_json_text = json.dumps(consolidated, ensure_ascii=False, indent=2)
            out_path = output_dir / "clinical_consolidated_output.json"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(enriched_json_text)

            print(f"Stored consolidated JSON -> {out_path}")
            return str(out_path)
        except Exception as e:
            print(f"Error calling LLM: {str(e)}")
            return None

    @staticmethod
    def answer_question(question: str, batch_id: str = None, top_k: int = TOP_K) -> tuple:
        """
        Answer a question based on the indexed documents.
        Uses ChromaDB for similarity search.
        Returns: (answer, log_file_path, source_documents)
        """
        vector_store = RagPipelineService.get_vector_store()
        qna_dir = RagPipelineService.get_batch_dir(batch_id, 'qna')

        try:
            question_embedding = get_embedding(question)
        except Exception as e:
            print(f"Error embedding question: {str(e)}")
            return None, None, []

        # Use ChromaDB for similarity search
        try:
            chunk_ids, chunk_texts, metadatas, distances = vector_store.similarity_search(
                batch_id=batch_id or "default",
                query_embedding=question_embedding,
                top_k=top_k
            )
        except Exception as e:
            print(f"Error performing similarity search: {str(e)}")
            return None, None, []

        # Build context from top chunks
        context = "\n\n".join(chunk_texts)

        # Extract unique source documents from metadata
        source_docs = []
        seen_docs = set()
        for metadata in metadatas:
            doc_name = metadata.get('document_name', 'Unknown')
            if doc_name not in seen_docs:
                seen_docs.add(doc_name)
                source_docs.append({
                    'name': doc_name,
                    'fileName': doc_name,
                    'chunks_used': sum(1 for m in metadatas if m.get('document_name') == doc_name)
                })

        qna_prompt = QNA_PROMPT_TEMPLATE.format(question=question, context=context)

        try:
            answer = call_llm(qna_prompt)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = qna_dir / f"qna_{ts}.txt"

            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"QUESTION:\n{question}\n\nANSWER:\n{answer}")

            print(f"Q&A stored -> {log_path}")
            return answer, str(log_path), source_docs
        except Exception as e:
            print(f"Error answering question: {str(e)}")
            return None, None, []

    @staticmethod
    def get_batch_status(batch_id: str = None) -> dict:
        """Get the status of a batch."""
        vector_store = RagPipelineService.get_vector_store()
        pdf_dir = RagPipelineService.get_batch_dir(batch_id, 'pdf')
        output_dir = RagPipelineService.get_batch_dir(batch_id, 'output')
        qna_dir = RagPipelineService.get_batch_dir(batch_id, 'qna')

        supported_extensions = ['*.pdf', '*.doc', '*.docx']
        pdf_count = sum(len(list(Path(pdf_dir).glob(ext))) for ext in supported_extensions)
        chunk_count = vector_store.get_chunk_count(batch_id or "default")
        has_output = (output_dir / "clinical_consolidated_output.json").exists()
        qna_logs = len(list(Path(qna_dir).glob("qna_*.txt")))

        return {
            "batch_id": batch_id or "default",
            "pdf_count": pdf_count,
            "chunk_count": chunk_count,
            "embedding_count": chunk_count,
            "has_output": has_output,
            "has_qna_logs": qna_logs > 0,
            "qna_log_count": qna_logs
        }
