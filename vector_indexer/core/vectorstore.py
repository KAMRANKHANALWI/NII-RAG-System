"""Vector store operations using Chroma."""

import os
import logging
from typing import List, Set
from langchain_chroma import Chroma
from langchain_core.documents import Document
from core.embeddings import get_embedding_model
from config.settings import VECTOR_DB_DIR


class VectorStoreManager:
    """Manages Chroma vector store operations."""

    def __init__(self, collection_name: str):
        """
        Initialize vector store manager.

        Args:
            collection_name: Name of the collection
        """
        self.collection_name = collection_name
        self.embedding_model = get_embedding_model()
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding_model,
            persist_directory=os.path.join(VECTOR_DB_DIR, collection_name),
        )

    def get_existing_hashes(self) -> Set[str]:
        """
        Retrieve existing document hashes from the vector store.

        Returns:
            Set of existing document hashes
        """
        existing_hashes = set()
        try:
            existing = self.vectorstore.get(include=["metadatas"])
            for metadata in existing["metadatas"]:
                if isinstance(metadata, dict) and "doc_hash" in metadata:
                    existing_hashes.add(metadata["doc_hash"])
        except Exception as e:
            logging.warning(f"Could not load existing metadata from Chroma: {e}")
        return existing_hashes

    def add_new_documents(self, documents: List[Document]) -> int:
        """
        Add new documents to the vector store, skipping duplicates.

        Args:
            documents: List of documents to add

        Returns:
            Number of documents actually added
        """
        existing_hashes = self.get_existing_hashes()
        new_docs = []

        for doc in documents:
            doc_hash = doc.metadata.get("doc_hash")
            if doc_hash and doc_hash not in existing_hashes:
                new_docs.append(doc)
            else:
                logging.debug(f"Skipped duplicate: {doc_hash}")

        if new_docs:
            self.vectorstore.add_documents(new_docs)
            logging.info(
                f"✅ Indexed {len(new_docs)} new documents into {self.collection_name}"
            )
        else:
            logging.info("⏩ All documents already exist. Skipping indexing.")

        return len(new_docs)
