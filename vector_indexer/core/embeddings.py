"""Embedding model configuration and setup."""

from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import EMBEDDING_MODEL_NAME


def get_embedding_model():
    """
    Initialize and return the embedding model.

    Returns:
        HuggingFaceEmbeddings instance
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
