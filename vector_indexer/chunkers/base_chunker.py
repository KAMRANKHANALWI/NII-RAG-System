"""Base abstract class for document chunkers."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from langchain_core.documents import Document


class BaseChunker(ABC):
    """Abstract base class for document chunkers."""
    
    @abstractmethod
    def chunk_data(self, data: Dict[str, Any], source_file: str) -> List[Document]:
        """
        Chunk data into documents.
        
        Args:
            data: Raw data to chunk
            source_file: Source file name
            
        Returns:
            List of chunked documents
        """
        pass
    
    def create_base_metadata(self, source_file: str, category: str, **kwargs) -> Dict[str, Any]:
        """
        Create base metadata common to all documents.
        
        Args:
            source_file: Source file name
            category: Document category
            **kwargs: Additional metadata fields
            
        Returns:
            Base metadata dictionary
        """
        metadata = {
            "source_file": source_file,
            "category": category,
        }
        metadata.update(kwargs)
        return metadata