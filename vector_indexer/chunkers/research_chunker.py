"""Research data chunker implementation."""

from typing import List, Dict, Any
from langchain_core.documents import Document
from chunkers.base_chunker import BaseChunker
from utils.hash_utils import content_hash


class ResearchChunker(BaseChunker):
    """Handles chunking of research data."""

    def chunk_data(self, data: Dict[str, Any], source_file: str) -> List[Document]:
        """
        Chunk research data - each faculty's research becomes a separate document.

        Args:
            data: Research data dictionary
            source_file: Source file name

        Returns:
            List of research profile documents
        """
        documents = []

        if not isinstance(data, dict):
            return documents

        metadata_info = data.get("metadata", {})
        base_metadata = self.create_base_metadata(
            source_file=source_file,
            category="research",
            data_collection_date=metadata_info.get("data_collection_date", ""),
            last_updated=metadata_info.get("last_updated", ""),
        )

        faculty_list = self._extract_faculty_research_list(data)

        for i, faculty in enumerate(faculty_list):
            doc = self._create_research_document(faculty, i, base_metadata)
            if doc:
                documents.append(doc)

        # Also create statistics document if available
        stats_doc = self._create_statistics_document(data, source_file)
        if stats_doc:
            documents.append(stats_doc)

        return documents

    def _extract_faculty_research_list(
        self, data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract faculty research list from different data structures."""
        faculty_list = []

        if "current_faculty" in data and "former_faculty" in data:
            faculty_list.extend(data.get("current_faculty", []))
            faculty_list.extend(data.get("former_faculty", []))
        elif "faculty_research" in data:
            faculty_list = data.get("faculty_research", [])

        return faculty_list

    def _create_research_document(
        self, faculty: Dict[str, Any], index: int, base_metadata: Dict[str, Any]
    ) -> Document:
        """Create a single faculty research document."""
        research_content = []
        faculty_name = faculty.get("name", "Unknown Faculty")

        # Basic information
        research_content.append(f"**Faculty Name**: {faculty_name}")

        if faculty.get("department"):
            research_content.append(f"**Department**: {faculty['department']}")

        if faculty.get("faculty_status"):
            research_content.append(f"**Faculty Status**: {faculty['faculty_status']}")

        # Research details
        research_fields = [
            ("research_interest", "Research Interest"),
            ("research_summary", "Research Summary"),
            ("lab_affiliation", "Lab Affiliation"),
        ]

        for field, label in research_fields:
            if faculty.get(field):
                research_content.append(f"**{label}**: {faculty[field]}")

        # Research arrays
        array_fields = [
            ("research_domains", "Research Domains"),
            ("research_keywords", "Research Keywords"),
            ("research_focus_areas", "Research Focus Areas"),
            ("collaboration_potential", "Collaboration Areas"),
        ]

        for field, label in array_fields:
            if faculty.get(field):
                research_content.append(f"**{label}**: {', '.join(faculty[field])}")

        if faculty.get("source_url"):
            research_content.append(f"**Source**: {faculty['source_url']}")

        # Create metadata
        faculty_metadata = base_metadata.copy()
        faculty_metadata.update(
            {
                "faculty_id": faculty.get("faculty_id", f"faculty_{index+1}"),
                "faculty_name": faculty_name,
                "department": faculty.get("department", ""),
                "faculty_status": faculty.get("faculty_status", "current"),
                "research_domains": ", ".join(faculty.get("research_domains", [])),
                "research_keywords": ", ".join(faculty.get("research_keywords", [])),
                "chunk_type": "faculty_research",
                "source": faculty.get("source_url", ""),
            }
        )

        content = f"Research Profile for {faculty_name}\n\n" + "\n\n".join(
            research_content
        )
        faculty_metadata["doc_hash"] = content_hash(content)

        return Document(page_content=content, metadata=faculty_metadata)

    def _create_statistics_document(
        self, data: Dict[str, Any], source_file: str
    ) -> Document:
        """Create research statistics document."""
        metadata_info = data.get("metadata", {})
        statistics = data.get("statistics", {})

        if not statistics and not metadata_info:
            return None

        stats_content = ["**Research Statistics and Overview**"]

        # Basic stats from metadata
        for field, label in [
            ("total_faculty", "Total Faculty"),
            ("total_current_faculty", "Current Faculty"),
            ("total_former_faculty", "Former Faculty"),
        ]:
            if metadata_info.get(field):
                stats_content.append(f"**{label}**: {metadata_info[field]}")

        if metadata_info.get("research_domains"):
            stats_content.append(
                f"**Research Domains**: {', '.join(metadata_info['research_domains'])}"
            )

        # Detailed statistics
        if statistics:
            stats_content.append("**Detailed Statistics**")

            # Distribution data
            for dist_field, label in [
                ("department_distribution", "Department Distribution"),
                ("research_domain_distribution", "Research Domain Distribution"),
            ]:
                if statistics.get(dist_field):
                    items = [f"{k}: {v}" for k, v in statistics[dist_field].items()]
                    stats_content.append(f"**{label}**: {', '.join(items)}")

            # Top keywords
            if statistics.get("top_keywords"):
                top_keywords = sorted(
                    statistics["top_keywords"].items(), key=lambda x: x[1], reverse=True
                )[:10]
                keyword_info = [f"{k}: {v}" for k, v in top_keywords]
                stats_content.append(
                    f"**Top Research Keywords**: {', '.join(keyword_info)}"
                )

        # Create metadata
        stats_metadata = {
            "source_file": source_file,
            "category": "research",
            "chunk_type": "research_statistics",
            "faculty_status": metadata_info.get("faculty_status", "all"),
            "total_faculty": metadata_info.get("total_faculty", 0),
            "data_collection_date": metadata_info.get("data_collection_date", ""),
            "last_updated": metadata_info.get("last_updated", ""),
        }

        content = "\n\n".join(stats_content)
        stats_metadata["doc_hash"] = content_hash(content)

        return Document(page_content=content, metadata=stats_metadata)
