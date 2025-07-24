"""Staff data chunker implementation."""

from typing import List, Dict, Any
from langchain_core.documents import Document
from chunkers.base_chunker import BaseChunker
from utils.hash_utils import content_hash


class StaffChunker(BaseChunker):
    """Handles chunking of staff directory data."""

    def chunk_data(self, data: Dict[str, Any], source_file: str) -> List[Document]:
        """
        Chunk staff data - each staff member becomes a separate document.

        Args:
            data: Staff data dictionary
            source_file: Source file name

        Returns:
            List of staff profile documents
        """
        documents = []

        if not isinstance(data, dict):
            return documents

        metadata_info = data.get("metadata", {})
        base_metadata = self.create_base_metadata(
            source_file=source_file,
            category="staff",
            data_collection_date=metadata_info.get("data_collection_date", ""),
            last_updated=metadata_info.get("last_updated", ""),
        )

        staff_list = self._extract_staff_list(data)

        for i, staff in enumerate(staff_list):
            doc = self._create_staff_document(staff, i, base_metadata)
            if doc:
                documents.append(doc)

        # Create statistics document for main database files
        if self._should_create_stats(source_file):
            stats_doc = self._create_statistics_document(data, source_file)
            if stats_doc:
                documents.append(stats_doc)

        return documents

    def _extract_staff_list(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract staff list from different data structures."""
        staff_list = []

        # Check different possible structures
        if "staff_directory" in data:
            staff_list = data.get("staff_directory", [])
        elif "staff" in data:
            staff_list = data.get("staff", [])
        elif "faculty" in data:
            staff_list = data.get("faculty", [])

        return staff_list

    def _create_staff_document(
        self, staff: Dict[str, Any], index: int, base_metadata: Dict[str, Any]
    ) -> Document:
        """Create a single staff profile document."""
        staff_content = []
        staff_name = staff.get("name", "Unknown Staff")

        # Basic information
        staff_content.append(f"**Staff Name**: {staff_name}")

        # Staff details
        staff_fields = [
            ("designation", "Designation"),
            ("category", "Category"),
            ("email", "Email"),
            ("extension", "Extension"),
            ("research_area", "Research Area"),
        ]

        for field, label in staff_fields:
            if staff.get(field):
                staff_content.append(f"**{label}**: {staff[field]}")

        # Boolean fields
        if staff.get("is_faculty") is not None:
            staff_content.append(
                f"**Is Faculty**: {'Yes' if staff['is_faculty'] else 'No'}"
            )

        if staff.get("has_contact_info") is not None:
            staff_content.append(
                f"**Has Contact Info**: {'Yes' if staff['has_contact_info'] else 'No'}"
            )

        # Create metadata
        staff_metadata = base_metadata.copy()
        staff_metadata.update(
            {
                "staff_id": staff.get("staff_id", f"staff_{index+1}"),
                "staff_name": staff_name,
                "designation": staff.get("designation", ""),
                "staff_category": staff.get("category", ""),
                "email": staff.get("email", ""),
                "extension": staff.get("extension", ""),
                "is_faculty": staff.get("is_faculty", False),
                "research_area": staff.get("research_area", ""),
                "chunk_type": (
                    "faculty_contact" if staff.get("is_faculty") else "staff_contact"
                ),
            }
        )

        content = f"Staff Profile for {staff_name}\n\n" + "\n\n".join(staff_content)
        staff_metadata["doc_hash"] = content_hash(content)

        return Document(page_content=content, metadata=staff_metadata)

    def _should_create_stats(self, source_file: str) -> bool:
        """Determine if statistics document should be created."""
        return "unified_staff_database" in source_file or "staff_summary" in source_file

    def _create_statistics_document(
        self, data: Dict[str, Any], source_file: str
    ) -> Document:
        """Create staff statistics document."""
        metadata_info = data.get("metadata", {})
        statistics = data.get("statistics", {})
        quick_stats = data.get("quick_stats", {})

        stats_content = ["**Staff Statistics and Overview**"]

        # Basic statistics
        for field, label in [
            ("total_staff", "Total Staff"),
        ]:
            if metadata_info.get(field):
                stats_content.append(f"**{label}**: {metadata_info[field]}")

        if metadata_info.get("staff_categories"):
            stats_content.append(
                f"**Staff Categories**: {', '.join(metadata_info['staff_categories'])}"
            )

        # Detailed statistics
        if statistics:
            stats_content.append("**Detailed Statistics**")

            for field, label in [
                ("total_staff", "Total Staff Count"),
                ("with_contact_info", "Staff with Contact Info"),
                ("email_available", "Staff with Email"),
                ("extension_available", "Staff with Extension"),
                ("faculty_count", "Faculty Count"),
            ]:
                if statistics.get(field):
                    stats_content.append(f"**{label}**: {statistics[field]}")

            # Category distribution
            if statistics.get("by_category"):
                category_info = [
                    f"{k}: {v}" for k, v in statistics["by_category"].items()
                ]
                stats_content.append(
                    f"**Staff by Category**: {', '.join(category_info)}"
                )

        # Quick stats
        if quick_stats and quick_stats.get("contact_coverage"):
            stats_content.append("**Contact Coverage**")
            for coverage_type, count in quick_stats["contact_coverage"].items():
                label = coverage_type.replace("_", " ").title()
                stats_content.append(f"**{label}**: {count}")

        # Create metadata
        stats_metadata = {
            "source_file": source_file,
            "category": "staff",
            "chunk_type": "staff_statistics",
            "total_staff": metadata_info.get("total_staff", 0),
            "data_collection_date": metadata_info.get("data_collection_date", ""),
            "last_updated": metadata_info.get("last_updated", ""),
        }

        content = "\n\n".join(stats_content)
        stats_metadata["doc_hash"] = content_hash(content)

        return Document(page_content=content, metadata=stats_metadata)
