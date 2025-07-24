# ===== chunkers/generic_chunker.py =====
"""Generic chunker for unspecialized data types."""

from typing import List, Dict, Any
from langchain_core.documents import Document
from chunkers.base_chunker import BaseChunker
from utils.content_utils import flatten_content
from utils.metadata_utils import flatten_metadata
from utils.hash_utils import content_hash


class GenericChunker(BaseChunker):
    """Handles chunking of generic data types that don't have specialized chunkers."""

    def chunk_data(self, data: Any, source_file: str) -> List[Document]:
        """
        Chunk generic data into documents.

        Args:
            data: Generic data to chunk
            source_file: Source file name

        Returns:
            List of generic documents
        """
        documents = []

        if isinstance(data, list):
            for item in data:
                doc = self._create_generic_document(item, source_file)
                if doc:
                    documents.append(doc)
        elif isinstance(data, dict):
            doc = self._create_generic_document(data, source_file)
            if doc:
                documents.append(doc)

        return documents

    def _create_generic_document(
        self, item: Dict[str, Any], source_file: str
    ) -> Document:
        """Create a generic document from data item."""
        content = flatten_content(item.get("content", item))
        metadata = flatten_metadata(item.get("metadata", {}))

        # Add base metadata
        base_metadata = self.create_base_metadata(
            source_file=source_file, category="generic"
        )
        metadata.update(base_metadata)

        # Enhanced metadata for specific content types
        content_lower = content.lower()
        title_lower = metadata.get("title", "").lower()

        if "alumni" in title_lower or "nii alumni:" in content_lower:
            metadata["doc_type"] = "alumni_list"
            metadata["is_alumni"] = True
            metadata["chunk_type"] = "alumni_list"
        elif (
            "all faculty lists" in title_lower
            or "current faculty members" in content_lower
        ):
            metadata["doc_type"] = "faculty_list"
            metadata["is_alumni"] = False
            metadata["chunk_type"] = "faculty_list"
        else:
            metadata["chunk_type"] = "generic_content"

        metadata["doc_hash"] = content_hash(content)

        return Document(page_content=content, metadata=metadata)
