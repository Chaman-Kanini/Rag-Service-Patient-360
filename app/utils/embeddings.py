import numpy as np
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from app.config import (
    AZURE_EMBEDDING_ENDPOINT,
    AZURE_EMBEDDING_KEY,
    AZURE_EMBEDDING_API_VERSION,
    EMBEDDING_MODEL,
    EMBEDDING_DEPLOYMENT
)

_azure_client = None


def _get_azure_client() -> AzureOpenAI:
    """Get or create Azure OpenAI client for embeddings."""
    global _azure_client
    if _azure_client is None:
        _azure_client = AzureOpenAI(
            api_version=AZURE_EMBEDDING_API_VERSION,
            azure_endpoint=AZURE_EMBEDDING_ENDPOINT,
            api_key=AZURE_EMBEDDING_KEY
        )
    return _azure_client


def get_embedding(text: str) -> np.ndarray:
    """Generate embedding for a text using Azure OpenAI."""
    try:
        client = _get_azure_client()
        response = client.embeddings.create(
            model=EMBEDDING_DEPLOYMENT,
            input=text
        )
        embedding = response.data[0].embedding
        return np.asarray(embedding, dtype=np.float32).reshape(-1)
    except Exception as e:
        raise Exception(f"Error generating embedding: {str(e)}")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    try:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    except Exception as e:
        raise Exception(f"Error calculating similarity: {str(e)}")
