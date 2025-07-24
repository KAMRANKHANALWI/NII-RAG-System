"""Lab data chunker implementation."""

from typing import List, Dict, Any
from langchain_core.documents import Document
from chunkers.base_chunker import BaseChunker
from utils.hash_utils import content_hash


class LabChunker(BaseChunker):
    """Handles chunking of lab data into multiple document types."""

    def chunk_data(self, data: Dict[str, Any], source_file: str) -> List[Document]:
        """
        Chunk lab data - each section becomes a separate document.

        Args:
            data: Lab data dictionary
            source_file: Source file name

        Returns:
            List of lab-related documents
        """
        documents = []

        if not isinstance(data, dict):
            return documents

        lab_profile = data.get("lab_profile", {})
        base_metadata = self.create_base_metadata(
            source_file=source_file,
            category="labs",
            lab_name=lab_profile.get("lab_name", "Unknown Lab"),
            lab_head=lab_profile.get("lab_head", ""),
            lab_category=lab_profile.get("lab_category", ""),
            lab_type=lab_profile.get("lab_type", "wet_lab"),
            faculty_status="current",
        )

        # Create different types of documents
        documents.extend(self._create_overview_document(data, base_metadata))
        documents.extend(self._create_research_program_documents(data, base_metadata))
        documents.extend(self._create_team_member_documents(data, base_metadata))
        documents.extend(self._create_alumni_documents(data, base_metadata))
        documents.extend(self._create_publication_documents(data, base_metadata))
        documents.extend(self._create_resources_document(data, base_metadata))
        documents.extend(self._create_contact_document(data, base_metadata))

        return documents

    def _create_overview_document(
        self, data: Dict[str, Any], base_metadata: Dict[str, Any]
    ) -> List[Document]:
        """Create lab overview document."""
        lab_profile = data.get("lab_profile", {})
        lab_overview = data.get("lab_overview", {})

        if not lab_profile and not lab_overview:
            return []

        overview_content = []

        # Basic lab info
        basic_fields = [
            ("lab_name", "Lab Name"),
            ("lab_head", "Lab Head"),
            ("lab_category", "Lab Category"),
            ("lab_type", "Lab Type"),
            ("institution", "Institution"),
            ("description", "Description"),
        ]

        for field, label in basic_fields:
            if lab_profile.get(field):
                overview_content.append(f"**{label}**: {lab_profile[field]}")

        # Overview details
        overview_fields = [
            ("mission_statement", "Mission Statement"),
        ]

        for field, label in overview_fields:
            if lab_overview.get(field):
                overview_content.append(f"**{label}**: {lab_overview[field]}")

        # Array fields
        array_fields = [
            ("research_focus_areas", "Research Focus Areas"),
            ("key_expertise", "Key Expertise"),
        ]

        for field, label in array_fields:
            if lab_overview.get(field):
                overview_content.append(
                    f"**{label}**: {', '.join(lab_overview[field])}"
                )

        if not overview_content:
            return []

        overview_metadata = base_metadata.copy()
        overview_metadata.update(
            {
                "chunk_type": "lab_overview",
            }
        )

        content = "\n\n".join(overview_content)
        overview_metadata["doc_hash"] = content_hash(content)

        return [Document(page_content=content, metadata=overview_metadata)]

    def _create_research_program_documents(
        self, data: Dict[str, Any], base_metadata: Dict[str, Any]
    ) -> List[Document]:
        """Create documents for each research program."""
        programs = data.get("research_programs", [])
        documents = []

        for i, program in enumerate(programs):
            program_content = [
                f"Research Program in {base_metadata.get('lab_name', 'Unknown Lab')}"
            ]

            # Program details
            program_fields = [
                ("title", "Program Title"),
                ("description", "Description"),
                ("status", "Status"),
            ]

            for field, label in program_fields:
                if program.get(field):
                    program_content.append(f"**{label}**: {program[field]}")

            # Array fields
            array_fields = [
                ("key_findings", "Key Findings"),
                ("techniques_used", "Techniques Used"),
            ]

            for field, label in array_fields:
                if program.get(field):
                    program_content.append(f"**{label}**: {', '.join(program[field])}")

            program_metadata = base_metadata.copy()
            program_metadata.update(
                {
                    "chunk_type": "research_program",
                    "program_id": program.get("program_id", f"program_{i+1}"),
                    "program_title": program.get("title", ""),
                }
            )

            content = "\n\n".join(program_content)
            program_metadata["doc_hash"] = content_hash(content)

            documents.append(Document(page_content=content, metadata=program_metadata))

        return documents

    def _create_team_member_documents(
        self, data: Dict[str, Any], base_metadata: Dict[str, Any]
    ) -> List[Document]:
        """Create documents for current team members."""
        team_members = data.get("current_team", [])
        documents = []

        for i, member in enumerate(team_members):
            member_content = [
                f"Current Team Member in {base_metadata.get('lab_name', 'Unknown Lab')}"
            ]

            # Member details
            member_fields = [
                ("name", "Name"),
                ("position", "Position"),
                ("role", "Role"),
                ("email", "Email"),
                ("bio", "Bio"),
            ]

            for field, label in member_fields:
                if member.get(field):
                    member_content.append(f"**{label}**: {member[field]}")

            # Array fields
            array_fields = [
                ("research_interests", "Research Interests"),
                ("education", "Education"),
            ]

            for field, label in array_fields:
                if member.get(field):
                    member_content.append(f"**{label}**: {', '.join(member[field])}")

            member_metadata = base_metadata.copy()
            member_metadata.update(
                {
                    "chunk_type": "current_team_member",
                    "member_id": member.get("member_id", f"member_{i+1}"),
                    "member_name": member.get("name", ""),
                    "member_position": member.get("position", ""),
                    "member_role": member.get("role", ""),
                    "status": "current",
                }
            )

            content = "\n\n".join(member_content)
            member_metadata["doc_hash"] = content_hash(content)

            documents.append(Document(page_content=content, metadata=member_metadata))

        return documents

    def _create_alumni_documents(
        self, data: Dict[str, Any], base_metadata: Dict[str, Any]
    ) -> List[Document]:
        """Create documents for lab alumni."""
        alumni = data.get("alumni", [])
        documents = []

        for i, alum in enumerate(alumni):
            alumni_content = [
                f"Alumni from {base_metadata.get('lab_name', 'Unknown Lab')}"
            ]

            # Alumni details
            alumni_fields = [
                ("name", "Name"),
                ("role_at_lab", "Role at Lab"),
                ("duration", "Duration"),
                ("graduation_year", "Graduation Year"),
                ("current_position", "Current Position"),
                ("current_institution", "Current Institution"),
                ("current_location", "Current Location"),
                ("career_sector", "Career Sector"),
            ]

            for field, label in alumni_fields:
                if alum.get(field):
                    alumni_content.append(f"**{label}**: {alum[field]}")

            alumni_metadata = base_metadata.copy()
            alumni_metadata.update(
                {
                    "chunk_type": "alumni",
                    "alumni_id": alum.get("alumni_id", f"alumni_{i+1}"),
                    "alumni_name": alum.get("name", ""),
                    "graduation_year": alum.get("graduation_year", ""),
                    "career_sector": alum.get("career_sector", ""),
                    "status": "alumni",
                }
            )

            content = "\n\n".join(alumni_content)
            alumni_metadata["doc_hash"] = content_hash(content)

            documents.append(Document(page_content=content, metadata=alumni_metadata))

        return documents

    def _create_publication_documents(
        self, data: Dict[str, Any], base_metadata: Dict[str, Any]
    ) -> List[Document]:
        """Create documents for lab publications."""
        publications = data.get("publications", [])
        documents = []

        for i, pub in enumerate(publications):
            pub_content = [
                f"Publication from {base_metadata.get('lab_name', 'Unknown Lab')}"
            ]

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

            pub_fields = [
                ("year", "Year"),
                ("journal", "Journal"),
                ("doi", "DOI"),
                ("doi_link", "DOI Link"),
            ]

            for field, label in pub_fields:
                if pub.get(field):
                    pub_content.append(f"**{label}**: {pub[field]}")

            pub_metadata = base_metadata.copy()
            pub_metadata.update(
                {
                    "chunk_type": "lab_publication",
                    "paper_id": pub.get("paper_id", f"pub_{i+1}"),
                    "title": pub.get("title", ""),
                    "year": pub.get("year", ""),
                    "journal": pub.get("journal", ""),
                }
            )

            content = "\n\n".join(pub_content)
            pub_metadata["doc_hash"] = content_hash(content)

            documents.append(Document(page_content=content, metadata=pub_metadata))

        return documents

    def _create_resources_document(
        self, data: Dict[str, Any], base_metadata: Dict[str, Any]
    ) -> List[Document]:
        """Create lab resources and facilities document."""
        resources = data.get("resources_and_facilities", {})

        if not resources:
            return []

        resources_content = [
            f"Resources and Facilities for {base_metadata.get('lab_name', 'Unknown Lab')}"
        ]

        # Resource details
        resource_fields = [
            ("lab_type", "Lab Type"),
        ]

        for field, label in resource_fields:
            if resources.get(field):
                resources_content.append(f"**{label}**: {resources[field]}")

        # Array fields
        array_fields = [
            ("computing_resources", "Computing Resources"),
            ("software_tools", "Software Tools"),
            ("equipment", "Equipment"),
            ("facilities", "Facilities"),
        ]

        for field, label in array_fields:
            if resources.get(field):
                resources_content.append(f"**{label}**: {', '.join(resources[field])}")

        # Databases and servers
        if resources.get("databases_and_servers"):
            db_names = [
                db.get("name", "")
                for db in resources["databases_and_servers"]
                if isinstance(db, dict) and db.get("name")
            ]
            if db_names:
                resources_content.append(
                    f"**Databases and Servers**: {', '.join(db_names)}"
                )

        resources_metadata = base_metadata.copy()
        resources_metadata.update(
            {
                "chunk_type": "lab_resources",
            }
        )

        content = "\n\n".join(resources_content)
        resources_metadata["doc_hash"] = content_hash(content)

        return [Document(page_content=content, metadata=resources_metadata)]

    def _create_contact_document(
        self, data: Dict[str, Any], base_metadata: Dict[str, Any]
    ) -> List[Document]:
        """Create lab contact information document."""
        contact = data.get("contact_information", {})

        if not contact:
            return []

        contact_content = [
            f"Contact Information for {base_metadata.get('lab_name', 'Unknown Lab')}"
        ]

        # Contact details
        contact_fields = [
            ("lab_website", "Lab Website"),
            ("email", "Email"),
            ("phone", "Phone"),
        ]

        for field, label in contact_fields:
            if contact.get(field):
                contact_content.append(f"**{label}**: {contact[field]}")

        # Address
        if contact.get("address") and isinstance(contact["address"], dict):
            address = contact["address"]
            addr_parts = []
            for key in ["street", "city", "postal_code", "country"]:
                if address.get(key):
                    addr_parts.append(address[key])
            if addr_parts:
                contact_content.append(f"**Address**: {', '.join(addr_parts)}")

        contact_metadata = base_metadata.copy()
        contact_metadata.update(
            {
                "chunk_type": "lab_contact",
            }
        )

        content = "\n\n".join(contact_content)
        contact_metadata["doc_hash"] = content_hash(content)

        return [Document(page_content=content, metadata=contact_metadata)]
