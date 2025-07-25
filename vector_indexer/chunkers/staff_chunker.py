"""Markdown staff data chunker implementation."""

import re
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from chunkers.base_chunker import BaseChunker
from utils.hash_utils import content_hash


class MarkdownStaffChunker(BaseChunker):
    """
    Processes staff directory data from markdown format into searchable document chunks.

    This chunker transforms structured markdown tables containing staff contact information
    into individual document chunks optimized for retrieval. Each staff member becomes
    a separate searchable document with comprehensive metadata and search terms.
    """

    def __init__(self):
        super().__init__()
        # Staff categorization mapping for role-based classification
        self.staff_categories = {
            "faculty": ["faculty", "dr.", "professor"],
            "administration": ["administration"],
            "technical_staff": ["technical staff"],
            "supporting_staff": ["supporting staff"],
        }

    def chunk_data(self, data: str, source_file: str) -> List[Document]:
        """
        Transform markdown staff directory into individual staff profile documents.

        Args:
            data: Raw markdown content containing staff directory tables
            source_file: Name of the source markdown file

        Returns:
            List of Document objects, one per staff member plus summary document
        """
        documents = []

        # Extract document metadata from frontmatter section
        metadata_info = self._extract_frontmatter(data)

        # Create base metadata template for all generated documents
        base_metadata = self.create_base_metadata(
            source_file=source_file,
            category="staff",
            data_collection_date=metadata_info.get("date_scraped", ""),
            last_updated=metadata_info.get("date_scraped", ""),
        )

        # Parse individual staff records from markdown tables
        staff_members = self._parse_staff_from_markdown(data)

        # Generate individual document for each staff member
        for staff in staff_members:
            doc = self._create_staff_document(staff, base_metadata)
            if doc:
                documents.append(doc)

        # Create comprehensive summary document with aggregated statistics
        summary_doc = self._create_summary_document(
            staff_members, source_file, base_metadata
        )
        if summary_doc:
            documents.append(summary_doc)

        return documents

    def _extract_frontmatter(self, content: str) -> Dict[str, Any]:
        """
        Extract YAML frontmatter metadata from markdown document header.

        Args:
            content: Full markdown document content

        Returns:
            Dictionary containing parsed frontmatter key-value pairs
        """
        frontmatter_pattern = r"^---\s*\n(.*?)\n---\s*\n"
        match = re.search(frontmatter_pattern, content, re.DOTALL)

        metadata = {}
        if match:
            frontmatter = match.group(1)
            for line in frontmatter.split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    metadata[key.strip().strip('"')] = value.strip().strip('"')

        return metadata

    def _parse_staff_from_markdown(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract staff member records from structured markdown tables.

        Processes paginated staff directory tables by locating table headers,
        skipping separator rows, and parsing individual staff data rows.

        Args:
            content: Full markdown content containing staff tables

        Returns:
            List of staff member dictionaries with contact information
        """
        staff_members = []

        # Split document into page sections for processing
        page_sections = re.split(r"### 📄 Page \d+", content)

        for section in page_sections[1:]:  # Skip document header section
            # Locate table structure within each page section
            lines = section.strip().split("\n")

            # Search for standard table header format
            header_found = False
            data_lines = []

            for i, line in enumerate(lines):
                if "| Name | Designation | Email ID | Extension No. |" in line:
                    header_found = True
                    # Extract data rows following header and separator
                    if i + 2 < len(lines):
                        data_lines = lines[i + 2 :]  # Skip header and separator lines
                    break

            if header_found:
                # Process each data row in the table
                for line in data_lines:
                    line = line.strip()
                    if line and line.startswith("|") and line.endswith("|"):
                        # Parse valid table row format
                        staff_data = self._parse_staff_row(line)
                        if staff_data and staff_data["name"].strip():
                            staff_members.append(staff_data)

        return staff_members

    def _parse_staff_row(self, row: str) -> Optional[Dict[str, Any]]:
        """
        Parse individual staff member data from table row format.

        Extracts and normalizes staff information including name, designation,
        contact details, and determines staff categorization.

        Args:
            row: Single table row string in pipe-delimited format

        Returns:
            Dictionary containing structured staff member data, or None if invalid
        """
        # Split table row by pipe delimiters and normalize whitespace
        parts = [part.strip() for part in row.split("|")]

        if len(parts) < 5:  # Validate minimum required columns
            return None

        # Extract core staff information fields
        name = parts[1].strip()
        designation = parts[2].strip()
        email = parts[3].strip()
        extension = parts[4].strip()

        if not name:  # Skip rows without staff names
            return None

        # Normalize name formatting and whitespace
        name = re.sub(r"\s+", " ", name)

        # Classify staff member by role and designation
        category = self._determine_category(name, designation)

        # Identify faculty status based on title and designation
        is_faculty = "faculty" in designation.lower() or "dr." in name.lower()

        # Normalize email format and handle encoding issues
        if email:
            email = email.replace("{at]", "@").replace("[at]", "@")

        return {
            "name": name,
            "designation": designation,
            "email": email if email else "",
            "extension": extension if extension else "",
            "category": category,
            "is_faculty": is_faculty,
        }

    def _determine_category(self, name: str, designation: str) -> str:
        """
        Classify staff member into appropriate organizational category.

        Uses keyword matching against name and designation to determine
        the most appropriate staff category classification.

        Args:
            name: Staff member's full name
            designation: Staff member's role/position title

        Returns:
            Formatted category string (e.g., "Faculty", "Administration")
        """
        combined_text = f"{name} {designation}".lower()

        for category, keywords in self.staff_categories.items():
            for keyword in keywords:
                if keyword in combined_text:
                    return category.replace("_", " ").title()

        return "Staff"  # Default category for unmatched cases

    def _create_staff_document(
        self, staff: Dict[str, Any], base_metadata: Dict[str, Any]
    ) -> Document:
        """
        Generate individual staff profile document optimized for search retrieval.

        Creates comprehensive staff profile with structured content, contact information,
        and multiple search term variations for flexible query matching.

        Args:
            staff: Dictionary containing staff member information
            base_metadata: Base metadata template for document creation

        Returns:
            Document object containing staff profile and enriched metadata
        """
        staff_name = staff["name"]

        # Construct structured staff profile content
        content_parts = []

        # Core identification and role information
        content_parts.append(f"**Name:** {staff_name}")
        content_parts.append(f"**Designation:** {staff['designation']}")
        content_parts.append(f"**Category:** {staff['category']}")

        # Contact information section
        if staff["email"]:
            content_parts.append(f"**Email:** {staff['email']}")

        if staff["extension"]:
            content_parts.append(f"**Extension/Phone:** {staff['extension']}")

        # Additional classification metadata
        content_parts.append(
            f"**Faculty Status:** {'Yes' if staff['is_faculty'] else 'No'}"
        )

        # Generate comprehensive search term variations for enhanced retrieval
        search_terms = []
        search_terms.append(staff_name.lower())

        # Create name variations for flexible matching capabilities
        name_parts = staff_name.replace("Dr.", "").replace("dr.", "").strip().split()
        if len(name_parts) > 1:
            search_terms.append(" ".join(name_parts))  # Name without title prefix
            search_terms.append(name_parts[-1])  # Last name only
            if len(name_parts) > 2:
                search_terms.append(
                    f"{name_parts[0]} {name_parts[-1]}"
                )  # First and last name combination

        # Add alternative name format variations
        if "dr." in staff_name.lower():
            search_terms.append(
                staff_name.replace("Dr.", "").replace("dr.", "").strip()
            )

        # Assemble final document content with search optimization
        main_content = "\n".join(content_parts)
        search_content = f"\n\n**Search Terms:** {', '.join(set(search_terms))}"
        full_content = (
            f"# Staff Profile: {staff_name}\n\n{main_content}{search_content}"
        )

        # Create comprehensive metadata for vector storage and filtering
        staff_metadata = base_metadata.copy()
        staff_metadata.update(
            {
                "staff_name": staff_name,
                "staff_name_normalized": staff_name.lower().replace("dr.", "").strip(),
                "designation": staff["designation"],
                "staff_category": staff["category"],
                "email": staff["email"],
                "extension": staff["extension"],
                "is_faculty": staff["is_faculty"],
                "search_terms": ", ".join(set(search_terms)),
            }
        )

        staff_metadata["doc_hash"] = content_hash(full_content)

        return Document(page_content=full_content, metadata=staff_metadata)

    def _create_summary_document(
        self,
        staff_members: List[Dict[str, Any]],
        source_file: str,
        base_metadata: Dict[str, Any],
    ) -> Document:
        """
        Generate comprehensive staff directory summary with statistical analysis.

        Creates aggregated overview document containing staff statistics, category
        distributions, and complete staff listings for broad search queries.

        Args:
            staff_members: List of all parsed staff member records
            source_file: Source file name for metadata
            base_metadata: Base metadata template

        Returns:
            Document object containing directory summary and statistics
        """
        # Calculate comprehensive staff statistics
        total_staff = len(staff_members)
        faculty_count = sum(1 for s in staff_members if s["is_faculty"])
        with_email = sum(1 for s in staff_members if s["email"])
        with_extension = sum(1 for s in staff_members if s["extension"])

        # Generate staff category distribution analysis
        categories = {}
        for staff in staff_members:
            cat = staff["category"]
            categories[cat] = categories.get(cat, 0) + 1

        # Construct comprehensive summary content
        content_parts = [
            "# NII Staff Directory Summary",
            "",
            f"**Total Staff Members:** {total_staff}",
            f"**Faculty Members:** {faculty_count}",
            f"**Staff with Email:** {with_email}",
            f"**Staff with Extension:** {with_extension}",
            "",
            "## Staff by Category:",
        ]

        # Add category breakdown with counts
        for category, count in sorted(categories.items()):
            content_parts.append(f"- **{category}:** {count} members")

        # Include complete staff roster for comprehensive search coverage
        if staff_members:
            content_parts.extend(
                [
                    "",
                    "## All Staff Members:",
                    ", ".join([s["name"] for s in staff_members]),
                ]
            )

        content = "\n".join(content_parts)

        # Create summary-specific metadata
        summary_metadata = base_metadata.copy()
        summary_metadata.update(
            {
                "summary_type": "staff_directory_overview",
                "total_staff": total_staff,
                "faculty_count": faculty_count,
                "staff_with_email": with_email,
                "staff_with_extension": with_extension,
                "categories": (
                    ", ".join(sorted(categories.keys())) if categories else ""
                ),
            }
        )

        summary_metadata["doc_hash"] = content_hash(content)

        return Document(page_content=content, metadata=summary_metadata)
