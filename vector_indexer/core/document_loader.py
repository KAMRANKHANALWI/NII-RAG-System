"""Document loading and processing orchestration."""

import os
import json
import logging
from typing import List, Dict, Any
from langchain_core.documents import Document
from chunkers.faculty_chunker import FacultyChunker
from chunkers.publications_chunker import PublicationsChunker
from utils.content_utils import flatten_content
from utils.metadata_utils import flatten_metadata
from utils.hash_utils import content_hash


class DocumentLoader:
    """Orchestrates document loading and chunking based on collection type."""

    def __init__(self):
        self.chunkers = {
            "faculty_info": FacultyChunker(),
            "publications": PublicationsChunker(),
            # Add other chunkers as needed
        }

    def load_documents_from_folder(
        self, folder_path: str, collection_name: str
    ) -> List[Document]:
        """
        Load and process all JSON documents from a folder.

        Args:
            folder_path: Path to folder containing JSON files
            collection_name: Name of the collection for chunker selection

        Returns:
            List of processed documents
        """
        documents = []

        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".json"):
                    docs = self._process_json_file(
                        os.path.join(root, file), file, collection_name, root
                    )
                    documents.extend(docs)
                    logging.info(f"📄 Processed {len(docs)} documents from {file}")

        return documents

    def _process_json_file(
        self, file_path: str, filename: str, collection_name: str, root_dir: str
    ) -> List[Document]:
        """Process a single JSON file."""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            # Use specific chunker if available
            if collection_name in self.chunkers:
                return self.chunkers[collection_name].chunk_data(data, filename)

            # Handle specific collection types by directory name
            if "Publications" in root_dir:
                return PublicationsChunker().chunk_data(data, filename)
            elif "FacultyInfo" in root_dir or "faculty" in filename.lower():
                return FacultyChunker().chunk_data(data, filename)

            # Default processing for other types
            return self._default_processing(data, filename)

        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
            return []

    def _default_processing(self, data: Any, filename: str) -> List[Document]:
        """Default processing for unspecialized data types."""
        documents = []

        if isinstance(data, list):
            for item in data:
                doc = self._create_generic_document(item, filename)
                if doc:
                    documents.append(doc)
        elif isinstance(data, dict):
            doc = self._create_generic_document(data, filename)
            if doc:
                documents.append(doc)

        return documents

    def _create_generic_document(self, item: Dict[str, Any], filename: str) -> Document:
        """Create a generic document from data item."""
        content = flatten_content(item.get("content", item))
        metadata = flatten_metadata(item.get("metadata", {}))
        metadata["doc_hash"] = content_hash(content)
        metadata["source_file"] = filename

        # Enhanced metadata for specific content types
        content_lower = content.lower()
        title_lower = metadata.get("title", "").lower()

        if "alumni" in title_lower or "nii alumni:" in content_lower:
            metadata["doc_type"] = "alumni_list"
            metadata["is_alumni"] = True
        elif (
            "all faculty lists" in title_lower
            or "current faculty members" in content_lower
        ):
            metadata["doc_type"] = "faculty_list"
            metadata["is_alumni"] = False

        return Document(page_content=content, metadata=metadata)
