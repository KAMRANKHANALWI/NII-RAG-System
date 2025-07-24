"""Faculty data chunker implementation."""

from typing import List, Dict, Any
from langchain_core.documents import Document
from chunkers.base_chunker import BaseChunker
from utils.hash_utils import content_hash


class FacultyChunker(BaseChunker):
    """Handles chunking of faculty profile data."""

    def chunk_data(self, data: Dict[str, Any], source_file: str) -> List[Document]:
        """
        Chunk faculty data into individual profile documents.

        Args:
            data: Faculty data dictionary
            source_file: Source file name

        Returns:
            List of faculty profile documents
        """
        documents = []

        if not isinstance(data, dict):
            return documents

        metadata_info = data.get("metadata", {})
        base_metadata = self.create_base_metadata(
            source_file=source_file,
            category="faculty_info",
            data_collection_date=metadata_info.get("data_collection_date", ""),
            last_updated=metadata_info.get("last_updated", ""),
        )

        faculty_list = self._extract_faculty_list(data)

        for i, faculty in enumerate(faculty_list):
            doc = self._create_faculty_document(faculty, i, base_metadata)
            if doc:
                documents.append(doc)

        return documents

    def _extract_faculty_list(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract faculty list from different data structures."""
        faculty_list = []

        if "current_faculty" in data and "former_faculty" in data:
            faculty_list.extend(data.get("current_faculty", []))
            faculty_list.extend(data.get("former_faculty", []))
        elif "faculty_profiles" in data:
            faculty_list = data.get("faculty_profiles", [])

        return faculty_list

    def _create_faculty_document(
        self, faculty: Dict[str, Any], index: int, base_metadata: Dict[str, Any]
    ) -> Document:
        """Create a single faculty profile document."""
        faculty_content = []
        faculty_name = faculty.get("name", "Unknown Faculty")

        # Basic information
        faculty_content.append(f"**Faculty Name**: {faculty_name}")

        if faculty.get("email"):
            faculty_content.append(f"**Email**: {faculty['email']}")

        if faculty.get("department"):
            faculty_content.append(f"**Department**: {faculty['department']}")

        if faculty.get("faculty_status"):
            faculty_content.append(f"**Faculty Status**: {faculty['faculty_status']}")

        # Research information
        self._add_research_info(faculty, faculty_content)

        # Qualifications
        self._add_qualifications(faculty, faculty_content)

        # Awards and other info
        self._add_additional_info(faculty, faculty_content)

        # Create metadata
        faculty_metadata = base_metadata.copy()
        faculty_metadata.update(
            {
                "faculty_id": faculty.get("faculty_id", f"faculty_{index+1}"),
                "faculty_name": faculty_name,
                "email": faculty.get("email", ""),
                "department": faculty.get("department", ""),
                "faculty_status": faculty.get("faculty_status", "current"),
                "research_areas": ", ".join(faculty.get("research_areas", [])),
                "research_keywords": ", ".join(faculty.get("research_keywords", [])),
                "career_level": faculty.get("career_level", ""),
                "chunk_type": "faculty_profile",
                "source": faculty.get("source_url", ""),
            }
        )

        content = f"Faculty Profile for {faculty_name}\n\n" + "\n\n".join(
            faculty_content
        )
        faculty_metadata["doc_hash"] = content_hash(content)

        return Document(page_content=content, metadata=faculty_metadata)

    def _add_research_info(self, faculty: Dict[str, Any], content_list: List[str]):
        """Add research-related information to content."""
        if faculty.get("research_interest"):
            content_list.append(
                f"**Research Interest**: {faculty['research_interest']}"
            )

        if faculty.get("research_areas"):
            content_list.append(
                f"**Research Areas**: {', '.join(faculty['research_areas'])}"
            )

        if faculty.get("research_keywords"):
            content_list.append(
                f"**Research Keywords**: {', '.join(faculty['research_keywords'])}"
            )

        if faculty.get("techniques_methods"):
            content_list.append(
                f"**Techniques & Methods**: {', '.join(faculty['techniques_methods'])}"
            )

    def _add_qualifications(self, faculty: Dict[str, Any], content_list: List[str]):
        """Add qualification information to content."""
        qualifications = faculty.get("qualifications", {})
        if qualifications:
            if qualifications.get("degrees"):
                content_list.append(
                    f"**Degrees**: {', '.join(qualifications['degrees'])}"
                )
            if qualifications.get("institutions"):
                content_list.append(
                    f"**Institutions**: {', '.join(qualifications['institutions'])}"
                )
            if qualifications.get("postdoc_experience"):
                content_list.append(
                    f"**Postdoc Experience**: {', '.join(qualifications['postdoc_experience'])}"
                )

    def _add_additional_info(self, faculty: Dict[str, Any], content_list: List[str]):
        """Add additional faculty information to content."""
        # Awards
        awards = faculty.get("awards", [])
        if awards:
            award_texts = [
                award.get("award", "") for award in awards if award.get("award")
            ]
            if award_texts:
                content_list.append(f"**Awards**: {', '.join(award_texts)}")

        # Other fields
        for field, label in [
            ("group_members", "Group Members"),
            ("collaboration_potential", "Collaboration Areas"),
            ("career_level", "Career Level"),
            ("profile_summary", "Profile Summary"),
            ("source_url", "Source"),
        ]:
            if faculty.get(field):
                if isinstance(faculty[field], list):
                    content_list.append(f"**{label}**: {', '.join(faculty[field])}")
                else:
                    content_list.append(f"**{label}**: {faculty[field]}")
