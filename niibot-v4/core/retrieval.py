"""Enhanced Faculty Retrieval System with Multi-Strategy Search Architecture.

This module implements a comprehensive 5-tier search strategy for optimal faculty
information retrieval, handling everything from exact names to casual first-name queries.

Key Features:
    - Intelligent faculty name extraction and matching
    - Progressive search strategies with automatic fallback
    - Cross-domain search for comprehensive information coverage
    - Robust error handling with performance logging
    - Smart partial name matching to prevent cross-faculty contamination
"""

import os
from typing import List
from langchain_chroma import Chroma
from langchain_core.documents import Document

from core.faculty_extractor import create_faculty_extractor_with_cache
from utils.document_utils import _deduplicate_documents
from utils.search_utils import get_comprehensive_director_info
from config.settings import VECTOR_DB_DIR, embedding_model


class EnhancedFacultyRetriever:
    """Advanced document retrieval system with 5-tier search strategy.

    Implements progressive search from most precise to most flexible:
        1. Exact metadata filtering
        2. First name direct lookup
        3. Smart partial name matching
        4. Cross-domain search
        5. Semantic fallback
    """

    def __init__(self, vectorstore: Chroma, domain: str):
        """Initialize retriever with domain-specific configuration.

        Args:
            vectorstore (Chroma): Vector database instance for document storage
            domain (str): Domain context for search operations
        """
        self.vectorstore = vectorstore
        self.domain = domain
        self.name_extractor = create_faculty_extractor_with_cache()

    def retrieve_with_faculty_awareness(self, query: str, k: int = 4) -> List[Document]:
        """Execute comprehensive faculty-aware document retrieval.

        Orchestrates 5-tier search strategy to handle all query patterns:
            - Formal: "Dr. Monica Sundd email"
            - Casual: "monica email"
            - Partial: "dr monica contact"
            - Complex: Multi-domain information requests

        Args:
            query (str): Natural language query from user
            k (int, optional): Maximum documents to retrieve. Defaults to 4.

        Returns:
            List[Document]: Ranked list of relevant documents with metadata
        """
        extracted_names = self.name_extractor.extract_names(query)
        query_lower = query.lower()

        print(f"Query: '{query}' | Domain: {self.domain}")
        print(f"Extracted names: {extracted_names}")

        all_documents = []

        # Strategy 1: Exact metadata filtering
        if extracted_names:
            all_documents = self._execute_exact_metadata_search(
                extracted_names, query, k
            )
            if all_documents:
                print(f"EXACT SUCCESS: {len(all_documents)} documents")
                return self._finalize_results(all_documents, k)

        # Strategy 2: First name direct lookup
        if not all_documents:
            all_documents = self._execute_first_name_search(query, query_lower, k)
            if all_documents:
                print(f"FIRST NAME SUCCESS: {len(all_documents)} documents")
                return self._finalize_results(all_documents, k)

        # Strategy 3: Smart partial name matching
        if not all_documents and extracted_names:
            all_documents = self._execute_partial_name_search(extracted_names, query, k)
            if all_documents:
                print(f"PARTIAL SUCCESS: {len(all_documents)} documents")
                return self._finalize_results(all_documents, k)

        # Strategy 4: Cross-domain search
        if not all_documents and extracted_names:
            all_documents = self._execute_cross_domain_search(extracted_names, query, k)
            if all_documents:
                print(f"CROSS-DOMAIN SUCCESS: {len(all_documents)} documents")
                return self._finalize_results(all_documents, k)

        # Strategy 5: Semantic fallback
        all_documents = self._execute_semantic_fallback(query, extracted_names, k)
        print(f"SEMANTIC FALLBACK: {len(all_documents)} documents")
        return self._finalize_results(all_documents, k)

    def _execute_exact_metadata_search(
        self, extracted_names: List[str], query: str, k: int
    ) -> List[Document]:
        """Execute precise metadata-based search for known faculty.

        Provides highest precision by filtering documents using exact faculty
        name metadata with verification to prevent false positives.

        Args:
            extracted_names (List[str]): List of faculty names extracted from query
            query (str): Original user query
            k (int): Number of documents to retrieve

        Returns:
            List[Document]: Verified documents matching exact faculty metadata
        """
        print("EXECUTING: Exact metadata filtering")
        all_docs = []

        for faculty_name in extracted_names:
            try:
                print(f"Searching: {faculty_name}")

                # Apply strict metadata filtering
                filtered_docs = self.vectorstore.similarity_search(
                    query, k=k * 2, filter={"faculty_name": faculty_name}
                )

                if filtered_docs:
                    print(f"Found {len(filtered_docs)} candidates")
                    verified_docs = self._verify_faculty_relevance(
                        filtered_docs, faculty_name
                    )

                    if verified_docs:
                        all_docs.extend(verified_docs)
                        print(f"Verified {len(verified_docs)} documents")
                        break  # Prioritize precision over coverage

            except Exception as e:
                print(f"Search failed for {faculty_name}: {e}")
                continue

        return all_docs

    def _execute_first_name_search(
        self, query: str, query_lower: str, k: int
    ) -> List[Document]:
        """Execute direct first name lookup for casual queries.

        Handles informal patterns like "monica email", "sarika research" by
        using FIRST_NAMES database for immediate faculty identification.
        Fast single-dictionary lookup with comprehensive validation.

        Args:
            query (str): Original user query
            query_lower (str): Lowercase version of query for matching
            k (int): Number of documents to retrieve

        Returns:
            List[Document]: Documents matched through first name lookup
        """
        print("EXECUTING: First name direct lookup")

        from config.faculty_data import FIRST_NAMES

        all_docs = []
        matched_faculty = []

        # Check each first name pattern against query
        for first_name, full_faculty_name in FIRST_NAMES.items():
            query_words = query_lower.split()

            if first_name in query_words or first_name in query_lower:
                if self._validate_first_name_context(query_lower, first_name):
                    matched_faculty.append(full_faculty_name)
                    print(f"Match: '{first_name}' → '{full_faculty_name}'")
                    break  # Use first match to avoid ambiguity

        # Search using resolved faculty name
        if matched_faculty:
            for faculty_name in matched_faculty:
                try:
                    # Try exact metadata filtering first
                    first_name_docs = self.vectorstore.similarity_search(
                        query, k=k * 2, filter={"faculty_name": faculty_name}
                    )

                    if first_name_docs:
                        verified_docs = self._verify_faculty_relevance(
                            first_name_docs, faculty_name
                        )
                        if verified_docs:
                            all_docs.extend(verified_docs)
                            print(f"Verified {len(verified_docs)} docs")
                            break

                    # Fallback: Enhanced semantic search
                    enhanced_query = f"{faculty_name} {query}"
                    semantic_docs = self.vectorstore.similarity_search(
                        enhanced_query, k=k
                    )

                    for doc in semantic_docs:
                        if self._is_document_about_faculty(doc, faculty_name):
                            all_docs.append(doc)

                    if all_docs:
                        print(f"Semantic found {len(all_docs)} docs")
                        break

                except Exception as e:
                    print(f"Failed for {faculty_name}: {e}")
                    continue

        return all_docs

    def _execute_partial_name_search(
        self, extracted_names: List[str], query: str, k: int
    ) -> List[Document]:
        """Execute smart partial name matching with component prioritization.

        Handles partial names by prioritizing unique first names over common surnames
        to prevent cross-faculty contamination.

        Args:
            extracted_names (List[str]): List of faculty names from query
            query (str): Original user query
            k (int): Number of documents to retrieve

        Returns:
            List[Document]: Documents matched through smart partial matching
        """
        print("EXECUTING: Smart partial name matching")
        all_docs = []

        for faculty_name in extracted_names:
            try:
                # Get prioritized name components
                name_components = self._get_prioritized_name_components(faculty_name)
                print(f"Prioritized components for {faculty_name}: {name_components}")

                for component in name_components:
                    if len(component) > 2:
                        enhanced_query = f"{component} {query}"
                        partial_docs = self.vectorstore.similarity_search(
                            enhanced_query, k=k * 3
                        )

                        # Apply strict post-filtering to ensure correct faculty
                        verified_docs = []
                        for doc in partial_docs:
                            if self._validate_partial_match_strict(
                                doc, faculty_name, component
                            ):
                                verified_docs.append(doc)
                                faculty_metadata = doc.metadata.get(
                                    "faculty_name", "Unknown"
                                )
                                print(
                                    f"VERIFIED partial match: {component} → {faculty_metadata}"
                                )

                        if verified_docs:
                            all_docs.extend(verified_docs)
                            break  # Found verified matches for this component

                if all_docs:
                    break  # Found matches for this faculty

            except Exception as e:
                print(f"Partial search failed: {e}")
                continue

        return all_docs

    def _get_prioritized_name_components(self, faculty_name: str) -> List[str]:
        """Get name components in priority order to avoid surname conflicts.

        Strategy:
            1. First names first (more unique: "Sarika", "Monica", "Nimesh")
            2. Middle names/initials (if present)
            3. Surnames last (often shared: "Gupta", "Kumar", "Singh")

        Args:
            faculty_name (str): Full faculty name to decompose

        Returns:
            List[str]: List of components in priority order (most unique first)
        """
        # Remove titles and normalize
        cleaned_name = faculty_name.replace("Dr.", "").replace("Prof.", "").strip()
        components = cleaned_name.split()

        if len(components) <= 1:
            return components

        # Identify common surnames that should be deprioritized
        common_surnames = {
            "gupta",
            "kumar",
            "singh",
            "sharma",
            "yadav",
            "roy",
            "das",
            "pal",
            "suri",
            "rani",
            "ali",
            "anand",
            "basu",
            "dhar",
            "meena",
            "negi",
        }

        prioritized = []
        surnames = []

        # Separate unique names from common surnames
        for component in components:
            if len(component) > 2:
                if component.lower() in common_surnames:
                    surnames.append(component)  # Deprioritize common surnames
                else:
                    prioritized.append(component)  # Prioritize unique names

        # Return unique names first, then surnames
        final_order = prioritized + surnames
        print(f"Component priority: {components} → {final_order}")
        return final_order

    def _validate_partial_match_strict(
        self, document: Document, faculty_name: str, component: str
    ) -> bool:
        """Strict validation for partial matches to prevent cross-faculty contamination.

        Uses multiple verification layers to ensure document is about the correct faculty:
            1. Component presence verification
            2. Faculty name cross-check
            3. Content analysis for name co-occurrence

        Args:
            document (Document): Document to validate
            faculty_name (str): Target faculty name (e.g., "Dr. Sarika Gupta")
            component (str): Component that triggered match (e.g., "Sarika")

        Returns:
            bool: True only if document is verified to be about target faculty
        """
        doc_faculty = document.metadata.get("faculty_name", "").lower()
        doc_content = document.page_content.lower()
        target_lower = faculty_name.lower()
        component_lower = component.lower()

        # Priority 1: Exact faculty metadata match (highest confidence)
        if doc_faculty and target_lower in doc_faculty:
            print(f"Exact metadata match: {doc_faculty}")
            return True

        # Priority 2: Component + target name co-occurrence in content
        if component_lower in doc_content and target_lower in doc_content:
            print("Component + target co-occurrence in content")
            return True

        # Priority 3: Multiple unique components present (strong indicator)
        target_components = self._extract_name_components(faculty_name)
        unique_components = [
            comp
            for comp in target_components
            if comp.lower()
            not in {"gupta", "kumar", "singh", "sharma", "yadav", "roy", "das", "pal"}
        ]

        if len(unique_components) >= 2:
            components_in_content = sum(
                1 for comp in unique_components if comp.lower() in doc_content
            )

            if components_in_content >= 2:
                print(
                    f"Multiple unique components match: {components_in_content}/{len(unique_components)}"
                )
                return True

        # Priority 4: Single unique component with high confidence
        if (
            len(unique_components) == 1
            and unique_components[0].lower() == component_lower
        ):
            if component_lower in doc_content:
                print(f"Unique component match: {component}")
                return True

        # Reject: Component match but wrong faculty detected
        if (
            doc_faculty
            and doc_faculty != target_lower
            and target_lower not in doc_faculty
        ):
            print(
                f"REJECTED: Component '{component}' matches wrong faculty "
                f"'{doc_faculty}', target: '{faculty_name}'"
            )
            return False

        # Reject: Common surname without additional verification
        common_surnames = {
            "gupta",
            "kumar",
            "singh",
            "sharma",
            "yadav",
            "roy",
            "das",
            "pal",
        }
        if component_lower in common_surnames:
            print(
                f"REJECTED: Common surname '{component}' without sufficient verification"
            )
            return False

        # Default: Weak match, likely false positive
        print(f"REJECTED: Insufficient evidence for '{component}' → '{faculty_name}'")
        return False

    def _execute_cross_domain_search(
        self, extracted_names: List[str], query: str, k: int
    ) -> List[Document]:
        """Execute intelligent cross-domain search across related collections.

        When current domain fails, searches related collections based on
        query intent. Contact queries may search faculty_info, research
        queries may search research domain, etc.

        Args:
            extracted_names (List[str]): Faculty names from query
            query (str): Original user query
            k (int): Number of documents to retrieve

        Returns:
            List[Document]: Documents found through cross-domain search
        """
        print(f"EXECUTING: Cross-domain search from {self.domain}")

        target_domains = self._determine_cross_domain_strategy(
            query.lower(), self.domain
        )

        if not target_domains:
            print("No suitable targets identified")
            return []

        all_docs = []

        for target_domain in target_domains:
            try:
                print(f"Trying: {target_domain}")
                domain_docs = self._search_target_domain(
                    extracted_names, query, target_domain, k
                )

                if domain_docs:
                    # Mark with cross-domain metadata
                    for doc in domain_docs:
                        doc.metadata["cross_domain_source"] = target_domain

                    all_docs.extend(domain_docs)
                    print(f"Success: {len(domain_docs)} docs from {target_domain}")
                    break  # Use first successful domain

            except Exception as e:
                print(f"Failed for {target_domain}: {e}")
                continue

        return all_docs

    def _execute_semantic_fallback(
        self, query: str, extracted_names: List[str], k: int
    ) -> List[Document]:
        """Execute semantic similarity search as ultimate fallback.

        Ensures system always returns relevant results by performing
        semantic similarity search, enhanced with extracted names when available.

        Args:
            query (str): Original user query
            extracted_names (List[str]): Faculty names from query
            k (int): Number of documents to retrieve

        Returns:
            List[Document]: Documents found through semantic search
        """
        print("EXECUTING: Semantic fallback")

        try:
            # Enhance query with faculty names if available
            if extracted_names:
                enhanced_query = f"{' '.join(extracted_names)} {query}"
                print(f"Enhanced: '{enhanced_query}'")
            else:
                enhanced_query = query

            semantic_docs = self.vectorstore.similarity_search(enhanced_query, k=k * 2)

            # Filter by faculty if names were extracted
            if extracted_names:
                filtered_docs = []
                for doc in semantic_docs:
                    if any(
                        self._is_document_about_faculty(doc, name)
                        for name in extracted_names
                    ):
                        filtered_docs.append(doc)

                if filtered_docs:
                    print(f"Filtered: {len(filtered_docs)} docs")
                    return filtered_docs

            print(f"Standard: {len(semantic_docs)} docs")
            return semantic_docs

        except Exception as e:
            print(f"Semantic fallback failed: {e}")
            return []

    def _determine_cross_domain_strategy(
        self, query_lower: str, current_domain: str
    ) -> List[str]:
        """Analyze query intent to determine optimal cross-domain targets.

        Uses keyword analysis to prioritize domains most likely to contain
        relevant information for the specific query type.

        Args:
            query_lower (str): Lowercase version of user query
            current_domain (str): Current search domain

        Returns:
            List[str]: Ordered list of target domains to search
        """
        # Analyze query intent
        contact_keywords = ["email", "phone", "contact", "extension", "call", "reach"]
        research_keywords = ["research", "interests", "working on", "projects", "focus"]
        publication_keywords = ["publications", "papers", "articles", "published"]

        is_contact_query = any(word in query_lower for word in contact_keywords)
        is_research_query = any(word in query_lower for word in research_keywords)
        is_publication_query = any(word in query_lower for word in publication_keywords)

        # Determine target domains based on intent
        if is_contact_query:
            domains_to_try = ["staff", "faculty_info"]
        elif is_research_query:
            domains_to_try = ["research", "faculty_info"]
        elif is_publication_query:
            domains_to_try = ["publications", "faculty_info"]
        else:
            domains_to_try = ["faculty_info", "staff", "research"]

        # Remove current domain to avoid redundancy
        domains_to_try = [d for d in domains_to_try if d != current_domain]

        print(f"Strategy: {current_domain} → {domains_to_try}")
        return domains_to_try

    def _search_target_domain(
        self, extracted_names: List[str], query: str, target_domain: str, k: int
    ) -> List[Document]:
        """Execute search within specific target domain.

        Handles vectorstore initialization, search execution, and result
        verification for cross-domain searches.

        Args:
            extracted_names (List[str]): Faculty names from query
            query (str): Original user query
            target_domain (str): Target domain to search
            k (int): Number of documents to retrieve

        Returns:
            List[Document]: Documents found in target domain
        """
        collection_path = os.path.join(VECTOR_DB_DIR, target_domain)
        if not os.path.exists(collection_path):
            print(f"Path not found: {collection_path}")
            return []

        target_vectorstore = Chroma(
            collection_name=target_domain,
            embedding_function=embedding_model,
            persist_directory=collection_path,
        )

        cross_domain_docs = []

        for faculty_name in extracted_names:
            try:
                # Try exact metadata filtering
                exact_docs = target_vectorstore.similarity_search(
                    query, k=k, filter={"faculty_name": faculty_name}
                )

                if exact_docs:
                    verified_docs = self._verify_faculty_relevance(
                        exact_docs, faculty_name
                    )
                    if verified_docs:
                        cross_domain_docs.extend(verified_docs)
                        print(f"Exact: {len(verified_docs)} docs")
                        break

                # Fallback to semantic search
                enhanced_query = f"{faculty_name} {query}"
                semantic_docs = target_vectorstore.similarity_search(
                    enhanced_query, k=k // 2
                )

                verified_semantic = []
                for doc in semantic_docs:
                    if self._is_document_about_faculty(doc, faculty_name):
                        verified_semantic.append(doc)

                if verified_semantic:
                    cross_domain_docs.extend(verified_semantic)
                    print(f"Semantic: {len(verified_semantic)} docs")
                    break

            except Exception as e:
                print(f"Failed for {faculty_name}: {e}")
                continue

        return cross_domain_docs

    def _validate_first_name_context(self, query_lower: str, first_name: str) -> bool:
        """Validate first name appears in appropriate academic context.

        Prevents false positives by checking for academic indicators
        and avoiding non-person query patterns.

        Args:
            query_lower (str): Lowercase version of user query
            first_name (str): First name to validate

        Returns:
            bool: True if context is appropriate for faculty search
        """
        from config.faculty_data import FALSE_POSITIVES

        # Check for false positive patterns
        for false_positive in FALSE_POSITIVES:
            if false_positive in query_lower:
                return False

        # Look for academic context indicators
        academic_indicators = [
            "dr",
            "prof",
            "faculty",
            "email",
            "contact",
            "research",
            "publication",
            "lab",
            "about",
            "tell me",
            "working on",
        ]

        return any(indicator in query_lower for indicator in academic_indicators)

    def _verify_faculty_relevance(
        self, documents: List[Document], target_faculty: str
    ) -> List[Document]:
        """Verify documents are genuinely about target faculty member.

        Implements strict verification to prevent false positives and
        ensure precision in faculty-specific queries.

        Args:
            documents (List[Document]): Documents to verify
            target_faculty (str): Target faculty name

        Returns:
            List[Document]: Verified documents about target faculty
        """
        verified_documents = []

        for doc in documents:
            if self._is_document_about_faculty(doc, target_faculty):
                verified_documents.append(doc)
            else:
                actual_faculty = doc.metadata.get("faculty_name", "Unknown")
                print(f"Rejected: {actual_faculty} ≠ {target_faculty}")

        return verified_documents

    def _is_document_about_faculty(
        self, document: Document, target_faculty: str
    ) -> bool:
        """Determine if document is genuinely about specified faculty member.

        Uses metadata and content analysis for comprehensive verification.

        Args:
            document (Document): Document to check
            target_faculty (str): Target faculty name

        Returns:
            bool: True if document is about target faculty
        """
        doc_faculty = document.metadata.get("faculty_name", "").lower()
        doc_content = document.page_content.lower()
        target_lower = target_faculty.lower()

        # Exact metadata match
        if doc_faculty == target_faculty:
            return True

        # Target name in content with component verification
        if target_lower in doc_content:
            target_components = (
                target_lower.replace("dr.", "").replace("prof.", "").strip().split()
            )
            component_matches = sum(
                1
                for component in target_components
                if len(component) > 3 and component in doc_content
            )

            if component_matches >= 2:
                return True

        # Partial metadata matching
        if target_lower in doc_faculty or doc_faculty in target_lower:
            return True

        return False

    def _extract_name_components(self, faculty_name: str) -> List[str]:
        """Extract searchable components from faculty names.

        Removes titles, normalizes spacing, and filters short components.

        Args:
            faculty_name (str): Full faculty name

        Returns:
            List[str]: List of searchable name components
        """
        cleaned_name = faculty_name.replace("Dr.", "").replace("Prof.", "").strip()
        components = cleaned_name.split()
        return [comp for comp in components if len(comp) > 2]

    def _finalize_results(self, documents: List[Document], k: int) -> List[Document]:
        """Finalize search results with deduplication and ranking.

        Removes duplicates and returns top-k results.

        Args:
            documents (List[Document]): Documents to finalize
            k (int): Maximum number of results to return

        Returns:
            List[Document]: Final deduplicated and ranked results
        """
        unique_documents = _deduplicate_documents(documents)
        final_results = unique_documents[:k] if unique_documents else []

        print(f"FINAL: {len(final_results)} unique documents")
        return final_results
