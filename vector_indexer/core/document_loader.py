"""Document loading and processing orchestration - COMPLETE & BUG-FREE VERSION."""

import os
import json
import logging
from typing import List, Dict, Any
from langchain_core.documents import Document

# Import ALL your existing chunkers
from chunkers.faculty_chunker import FacultyChunker
from chunkers.publications_chunker import PublicationsChunker
from chunkers.research_chunker import ResearchChunker
from chunkers.lab_chunker import LabChunker
from chunkers.staff_chunker import MarkdownStaffChunker
from chunkers.generic_chunker import GenericChunker

from utils.content_utils import flatten_content
from utils.metadata_utils import flatten_metadata
from utils.hash_utils import content_hash


class DocumentLoader:
    """Orchestrates document loading and chunking based on collection type."""

    def __init__(self):
        # REGISTER ALL YOUR CHUNKERS
        self.chunkers = {
            "faculty_info": FacultyChunker(),
            "publications": PublicationsChunker(),
            "research": ResearchChunker(),
            "labs": LabChunker(),
            "staff": MarkdownStaffChunker(),
            # Add other collections as needed:
            # "infrastructure": GenericChunker(),
            # "alumni": GenericChunker(), etc
        }

        # FALLBACK CHUNKER for collections without specific chunkers
        self.generic_chunker = GenericChunker()

    def load_documents_from_folder(
        self, folder_path: str, collection_name: str
    ) -> List[Document]:
        """
        Load and process all JSON and MD documents from a folder.

        Args:
            folder_path: Path to folder containing JSON/MD files
            collection_name: Name of the collection for chunker selection

        Returns:
            List of processed documents
        """
        documents = []

        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)

                # ✅ HANDLE BOTH JSON AND MARKDOWN FILES
                if file.endswith(".json"):
                    docs = self._process_json_file(
                        file_path, file, collection_name, root
                    )
                    documents.extend(docs)
                    logging.info(f"📄 Processed {len(docs)} JSON documents from {file}")

                elif file.endswith(".md"):  # Add markdown support
                    docs = self._process_markdown_file(
                        file_path, file, collection_name, root
                    )
                    documents.extend(docs)
                    logging.info(f"📄 Processed {len(docs)} MD documents from {file}")

        return documents

    def _process_json_file(
        self, file_path: str, filename: str, collection_name: str, root_dir: str
    ) -> List[Document]:
        """Process a single JSON file with proper chunker selection."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # ✅ PRIORITY 1: Use registered chunker for collection
            if collection_name in self.chunkers:
                logging.info(f"🎯 Using {collection_name} chunker for {filename}")
                return self.chunkers[collection_name].chunk_data(data, filename)

            # ✅ PRIORITY 2: Smart detection by directory/filename
            chunker_to_use = self._detect_chunker_by_path(root_dir, filename)
            if chunker_to_use:
                logging.info(
                    f"🔍 Auto-detected {chunker_to_use.__class__.__name__} for {filename}"
                )
                return chunker_to_use.chunk_data(data, filename)

            # ✅ PRIORITY 3: Use generic chunker (ALWAYS WORKS)
            logging.info(f"🔧 Using generic chunker for {filename}")
            return self.generic_chunker.chunk_data(data, filename)

        except Exception as e:
            logging.error(f"❌ Error processing {file_path}: {e}")
            return []

    def _process_markdown_file(
        self, file_path: str, filename: str, collection_name: str, root_dir: str
    ) -> List[Document]:
        """Process a single Markdown file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = f.read()

            # PRIORITY 1: Use registered chunker for collection
            if collection_name in self.chunkers:
                logging.info(f"🎯 Using {collection_name} chunker for {filename}")
                return self.chunkers[collection_name].chunk_data(data, filename)

            # PRIORITY 2: Auto-detect staff files
            if self._is_staff_file(filename, data):
                logging.info(f"👥 Auto-detected staff file: {filename}")
                return MarkdownStaffChunker().chunk_data(data, filename)

            # PRIORITY 3: Generic markdown processing
            logging.info(f"📝 Using generic processing for markdown: {filename}")
            return self._create_generic_markdown_document(data, filename)

        except Exception as e:
            logging.error(f"❌ Error processing {file_path}: {e}")
            return []

    def _detect_chunker_by_path(self, root_dir: str, filename: str):
        """Smart chunker detection based on path and filename."""
        root_lower = root_dir.lower()
        file_lower = filename.lower()

        # Publications detection
        if "publications" in root_lower or "publication" in file_lower:
            return PublicationsChunker()

        # Faculty detection
        elif "facultyinfo" in root_lower or "faculty" in file_lower:
            return FacultyChunker()

        # Research detection
        elif "research" in root_lower or "research" in file_lower:
            return ResearchChunker()

        # Labs detection
        elif "labs" in root_lower or "lab" in file_lower:
            return LabChunker()

        # Staff detection (for JSON staff files if any)
        elif "staff" in root_lower or "staff" in file_lower:
            # Note: This would be for JSON staff files, MD files use different logic
            return self.generic_chunker  # Use generic for JSON staff files

        return None  # No specific chunker detected

    def _is_staff_file(self, filename: str, content: str) -> bool:
        """Determine if a markdown file contains staff data."""
        filename_lower = filename.lower()
        content_lower = content.lower()

        # Check filename patterns
        staff_filename_patterns = ["staff", "who", "contact", "directory"]
        if any(pattern in filename_lower for pattern in staff_filename_patterns):
            return True

        # Check content patterns
        staff_content_patterns = [
            "| name | designation | email",
            "extension no.",
            "faculty",
            "administration",
            "technical staff",
        ]
        if any(pattern in content_lower for pattern in staff_content_patterns):
            return True

        return False

    def _create_generic_markdown_document(
        self, content: str, filename: str
    ) -> List[Document]:
        """Create a generic document from markdown content."""
        metadata = {
            "source_file": filename,
            "category": "generic",
            "doc_hash": content_hash(content),
        }

        return [Document(page_content=content, metadata=metadata)]

    def _default_processing(self, data: Any, filename: str) -> List[Document]:
        """Default processing using generic chunker."""
        logging.info(f"🔧 Using default processing for {filename}")
        return self.generic_chunker.chunk_data(data, filename)

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
