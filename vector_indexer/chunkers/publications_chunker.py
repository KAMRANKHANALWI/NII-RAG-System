"""Publications data chunker implementation."""

from typing import List, Dict, Any
from langchain_core.documents import Document
from chunkers.base_chunker import BaseChunker
from utils.hash_utils import content_hash


class PublicationsChunker(BaseChunker):
    """Handles chunking of publication data."""

    def chunk_data(self, data: Dict[str, Any], source_file: str) -> List[Document]:
        """
        Chunk publications data - each publication becomes a separate document.

        Args:
            data: Publications data
            source_file: Source file name

        Returns:
            List of publication documents
        """
        documents = []

        if isinstance(data, dict):
            documents.extend(self._process_unified_structure(data, source_file))
        elif isinstance(data, list):
            documents.extend(self._process_legacy_structure(data, source_file))

        return documents

    def _process_unified_structure(
        self, data: Dict[str, Any], source_file: str
    ) -> List[Document]:
        """Process new unified publication structure."""
        documents = []
        faculty_profile = data.get("faculty_profile", {})
        publications = data.get("publications", [])

        base_metadata = self.create_base_metadata(
            source_file=source_file,
            category="publications",
            faculty_name=faculty_profile.get("name", "Unknown"),
            email=faculty_profile.get("email", ""),
            source=faculty_profile.get("source_url", ""),
            research_domain=faculty_profile.get("research_domain", ""),
            faculty_status=faculty_profile.get("faculty_status", "current"),
            total_publications=len(publications),
        )

        for i, pub in enumerate(publications):
            doc = self._create_publication_document(
                pub, i, faculty_profile, base_metadata
            )
            if doc:
                documents.append(doc)

        return documents

    def _process_legacy_structure(
        self, data: List[Dict[str, Any]], source_file: str
    ) -> List[Document]:
        """Process old publication list structure."""
        documents = []

        for i, item in enumerate(data):
            if isinstance(item, dict) and item.get("content"):
                from utils.content_utils import flatten_content
                from utils.metadata_utils import flatten_metadata

                content = flatten_content(item.get("content", item))
                metadata = flatten_metadata(item.get("metadata", {}))
                metadata.update(
                    self.create_base_metadata(
                        source_file=source_file,
                        category="publications",
                        chunk_type="publication",
                        publication_number=i + 1,
                    )
                )
                metadata["doc_hash"] = content_hash(content)

                documents.append(Document(page_content=content, metadata=metadata))

        return documents

    def _create_publication_document(
        self,
        pub: Dict[str, Any],
        index: int,
        faculty_profile: Dict[str, Any],
        base_metadata: Dict[str, Any],
    ) -> Document:
        """Create a single publication document."""
        pub_content = []

        # Publication details
        if pub.get("title"):
            pub_content.append(f"**Title**: {pub['title']}")

        if pub.get("authors"):
            authors_str = (
                ", ".join(pub["authors"])
                if isinstance(pub["authors"], list)
                else str(pub["authors"])
            )
            pub_content.append(f"**Authors**: {authors_str}")

        for field, label in [
            ("year", "Year"),
            ("journal", "Journal"),
            ("doi", "DOI"),
            ("abstract", "Abstract"),
            ("doi_link", "DOI Link"),
        ]:
            if pub.get(field):
                pub_content.append(f"**{label}**: {pub[field]}")

        # Create publication-specific metadata
        pub_metadata = base_metadata.copy()
        pub_metadata.update(
            {
                "publication_id": pub.get("publication_id", f"pub_{index+1}"),
                "title": pub.get("title", ""),
                "year": pub.get("year", ""),
                "journal": pub.get("journal", ""),
                "doi": pub.get("doi", ""),
                "chunk_type": "publication",
                "publication_number": index + 1,
            }
        )

        content = (
            f"Publication by {faculty_profile.get('name', 'Unknown Faculty')}\n\n"
            + "\n\n".join(pub_content)
        )
        pub_metadata["doc_hash"] = content_hash(content)

        return Document(page_content=content, metadata=pub_metadata)
