import faiss
import pickle
import json
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from pathlib import Path
from app.config import CHROMA_PERSIST_DIR


class ChromaVectorStore:
    """Service for managing vector embeddings using FAISS."""
    
    def __init__(self, persist_directory: str = None):
        """
        Initialize FAISS vector store with persistent storage.
        
        Args:
            persist_directory: Path to persist FAISS data. Defaults to CHROMA_PERSIST_DIR.
        """
        if persist_directory is None:
            persist_directory = str(CHROMA_PERSIST_DIR)
        
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Dictionary to store FAISS indices and metadata per batch
        self.indices = {}  # batch_id -> faiss.Index
        self.metadata_store = {}  # batch_id -> {chunk_id: {"text": str, "metadata": dict}}
        self.id_mapping = {}  # batch_id -> {chunk_id: index_position}
        
        # Load existing indices
        self._load_all_indices()
        
    def _normalize_batch_id(self, batch_id: str) -> str:
        """
        Normalize batch_id to always have 'batch_' prefix.
        This ensures consistency between storage and lookup.
        """
        if not batch_id:
            return "default_batch"
        if batch_id.startswith("batch_"):
            return batch_id.replace("-", "_").lower()
        return f"batch_{batch_id}".replace("-", "_").lower()
    
    def _get_batch_files(self, batch_id: str) -> Tuple[Path, Path, Path]:
        """
        Get file paths for a batch's FAISS index, metadata, and ID mapping.
        
        Args:
            batch_id: Unique identifier for the batch
            
        Returns:
            Tuple of (index_path, metadata_path, id_mapping_path)
        """
        batch_name = self._normalize_batch_id(batch_id)
        
        index_path = self.persist_directory / f"{batch_name}.index"
        metadata_path = self.persist_directory / f"{batch_name}_metadata.pkl"
        id_mapping_path = self.persist_directory / f"{batch_name}_ids.pkl"
        
        return index_path, metadata_path, id_mapping_path
    
    def _load_all_indices(self):
        """Load all existing FAISS indices from disk."""
        for index_file in self.persist_directory.glob("*.index"):
            batch_name = index_file.stem
            # Keep the full batch name (with batch_ prefix) as the batch_id for consistency
            batch_id = batch_name
            
            try:
                self._load_index(batch_id)
            except Exception as e:
                print(f"Error loading index for batch {batch_id}: {str(e)}")
    
    def _load_index(self, batch_id: str):
        """Load a FAISS index and its metadata from disk."""
        index_path, metadata_path, id_mapping_path = self._get_batch_files(batch_id)
        
        if index_path.exists():
            # Load FAISS index
            self.indices[batch_id] = faiss.read_index(str(index_path))
            
            # Load metadata
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    self.metadata_store[batch_id] = pickle.load(f)
            else:
                self.metadata_store[batch_id] = {}
            
            # Load ID mapping
            if id_mapping_path.exists():
                with open(id_mapping_path, 'rb') as f:
                    self.id_mapping[batch_id] = pickle.load(f)
            else:
                self.id_mapping[batch_id] = {}
    
    def _save_index(self, batch_id: str):
        """Save a FAISS index and its metadata to disk."""
        index_path, metadata_path, id_mapping_path = self._get_batch_files(batch_id)
        
        # Save FAISS index
        if batch_id in self.indices:
            faiss.write_index(self.indices[batch_id], str(index_path))
        
        # Save metadata
        if batch_id in self.metadata_store:
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.metadata_store[batch_id], f)
        
        # Save ID mapping
        if batch_id in self.id_mapping:
            with open(id_mapping_path, 'wb') as f:
                pickle.dump(self.id_mapping[batch_id], f)
    
    def _get_or_create_index(self, batch_id: str, dimension: int = 768) -> faiss.Index:
        """
        Get or create a FAISS index for a batch.
        
        Args:
            batch_id: Unique identifier for the batch
            dimension: Dimension of the embedding vectors
            
        Returns:
            FAISS index object
        """
        batch_id = self._normalize_batch_id(batch_id)
        if batch_id not in self.indices:
            # Create new FAISS index (using L2 distance)
            self.indices[batch_id] = faiss.IndexFlatL2(dimension)
            self.metadata_store[batch_id] = {}
            self.id_mapping[batch_id] = {}
        
        return self.indices[batch_id]
    
    def add_chunks(
        self,
        batch_id: str,
        chunk_ids: List[str],
        chunks: List[str],
        embeddings: List[np.ndarray],
        metadatas: List[Dict[str, Any]]
    ) -> None:
        """
        Add chunks with their embeddings to FAISS index.
        
        Args:
            batch_id: Batch identifier
            chunk_ids: List of unique chunk identifiers
            chunks: List of chunk texts
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries for each chunk
        """
        batch_id = self._normalize_batch_id(batch_id)
        if not embeddings:
            return
        
        # Convert embeddings to numpy array
        embeddings_array = np.array([emb if isinstance(emb, np.ndarray) else np.array(emb) 
                                     for emb in embeddings]).astype('float32')
        
        dimension = embeddings_array.shape[1]
        index = self._get_or_create_index(batch_id, dimension)
        
        # Handle upsert: remove existing chunks if they exist
        for chunk_id in chunk_ids:
            if chunk_id in self.id_mapping.get(batch_id, {}):
                # Remove old entry from metadata
                if chunk_id in self.metadata_store[batch_id]:
                    del self.metadata_store[batch_id][chunk_id]
        
        # Add new embeddings to index
        start_idx = index.ntotal
        index.add(embeddings_array)
        
        # Store metadata and ID mapping
        for i, (chunk_id, chunk_text, metadata) in enumerate(zip(chunk_ids, chunks, metadatas)):
            idx = start_idx + i
            self.id_mapping[batch_id][chunk_id] = idx
            self.metadata_store[batch_id][chunk_id] = {
                "text": chunk_text,
                "metadata": metadata
            }
        
        # Persist to disk
        self._save_index(batch_id)
    
    def add_single_chunk(
        self,
        batch_id: str,
        chunk_id: str,
        chunk_text: str,
        embedding: np.ndarray,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Add a single chunk with its embedding to ChromaDB.
        
        Args:
            batch_id: Batch identifier
            chunk_id: Unique chunk identifier
            chunk_text: Chunk text content
            embedding: Embedding vector
            metadata: Metadata dictionary
        """
        self.add_chunks(
            batch_id=batch_id,
            chunk_ids=[chunk_id],
            chunks=[chunk_text],
            embeddings=[embedding],
            metadatas=[metadata]
        )
    
    def similarity_search(
        self,
        batch_id: str,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]], List[float]]:
        """
        Perform similarity search using FAISS.
        
        Args:
            batch_id: Batch identifier
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            
        Returns:
            Tuple of (chunk_ids, chunk_texts, metadatas, distances)
        """
        batch_id = self._normalize_batch_id(batch_id)
        if batch_id not in self.indices or self.indices[batch_id].ntotal == 0:
            return [], [], [], []
        
        index = self.indices[batch_id]
        
        # Convert query embedding to correct format
        query_vector = query_embedding if isinstance(query_embedding, np.ndarray) else np.array(query_embedding)
        query_vector = query_vector.astype('float32').reshape(1, -1)
        
        # Perform search
        k = min(top_k, index.ntotal)
        distances, indices = index.search(query_vector, k)
        
        # Get results
        chunk_ids = []
        chunk_texts = []
        metadatas = []
        result_distances = []
        
        # Reverse mapping: index position -> chunk_id
        idx_to_chunk_id = {v: k for k, v in self.id_mapping[batch_id].items()}
        
        for i, idx in enumerate(indices[0]):
            if idx == -1:  # FAISS returns -1 for invalid results
                continue
            
            chunk_id = idx_to_chunk_id.get(int(idx))
            if chunk_id and chunk_id in self.metadata_store[batch_id]:
                data = self.metadata_store[batch_id][chunk_id]
                chunk_ids.append(chunk_id)
                chunk_texts.append(data["text"])
                metadatas.append(data["metadata"])
                result_distances.append(float(distances[0][i]))
        
        return chunk_ids, chunk_texts, metadatas, result_distances
    
    def get_all_chunks(
        self,
        batch_id: str
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        """
        Get all chunks from a batch.
        
        Args:
            batch_id: Batch identifier
            
        Returns:
            Tuple of (chunk_ids, chunk_texts, metadatas)
        """
        batch_id = self._normalize_batch_id(batch_id)
        if batch_id not in self.metadata_store:
            return [], [], []
        
        chunk_ids = []
        chunk_texts = []
        metadatas = []
        
        for chunk_id, data in self.metadata_store[batch_id].items():
            chunk_ids.append(chunk_id)
            chunk_texts.append(data["text"])
            metadatas.append(data["metadata"])
        
        return chunk_ids, chunk_texts, metadatas
    
    def get_chunk_count(self, batch_id: str) -> int:
        """
        Get the number of chunks in a batch.
        
        Args:
            batch_id: Batch identifier
            
        Returns:
            Number of chunks in the batch
        """
        batch_id = self._normalize_batch_id(batch_id)
        if batch_id in self.indices:
            return self.indices[batch_id].ntotal
        return 0
    
    def delete_collection(self, batch_id: str) -> None:
        """
        Delete a batch and its files.
        
        Args:
            batch_id: Batch identifier
        """
        batch_id = self._normalize_batch_id(batch_id)
        # Remove from memory
        if batch_id in self.indices:
            del self.indices[batch_id]
        if batch_id in self.metadata_store:
            del self.metadata_store[batch_id]
        if batch_id in self.id_mapping:
            del self.id_mapping[batch_id]
        
        # Delete files
        index_path, metadata_path, id_mapping_path = self._get_batch_files(batch_id)
        
        try:
            if index_path.exists():
                index_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()
            if id_mapping_path.exists():
                id_mapping_path.unlink()
        except Exception as e:
            print(f"Error deleting batch files for {batch_id}: {str(e)}")
    
    def list_collections(self) -> List[str]:
        """
        List all batches.
        
        Returns:
            List of batch IDs
        """
        return list(self.indices.keys())
    
    def collection_exists(self, batch_id: str) -> bool:
        """
        Check if a batch exists.
        
        Args:
            batch_id: Batch identifier
            
        Returns:
            True if batch exists, False otherwise
        """
        batch_id = self._normalize_batch_id(batch_id)
        return batch_id in self.indices
