import numpy as np
from huggingface_hub import InferenceClient
from app.config import EMBEDDING_MODEL, EMBEDDING_PROVIDER, HUGGINGFACE_API_TOKEN

_hf_client = None
_local_model = None


def _get_hf_client() -> InferenceClient:
    global _hf_client
    if _hf_client is None:
        _hf_client = InferenceClient(token=HUGGINGFACE_API_TOKEN)
    return _hf_client


def _get_local_model():
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer

        _local_model = SentenceTransformer(EMBEDDING_MODEL)
    return _local_model


def get_embedding(text: str) -> np.ndarray:
    """Generate embedding for a text using HuggingFace Inference API."""
    try:
        provider = (EMBEDDING_PROVIDER or "hf_api").strip().lower()
        if provider == "local":
            model = _get_local_model()
            embedding = model.encode(text)
            return np.asarray(embedding, dtype=np.float32).reshape(-1)

        client = _get_hf_client()
        embedding = client.feature_extraction(text=text, model=EMBEDDING_MODEL)
        embedding_array = np.asarray(embedding, dtype=np.float32)
        if len(embedding_array.shape) > 1:
            embedding_array = embedding_array[0]
        return embedding_array.reshape(-1)
    except Exception as e:
        raise Exception(f"Error generating embedding: {str(e)}")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    try:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    except Exception as e:
        raise Exception(f"Error calculating similarity: {str(e)}")
