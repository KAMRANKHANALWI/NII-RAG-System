import os
from typing import List
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Import from other modules
from core.faculty_extractor import create_faculty_extractor_with_cache
from utils.document_utils import _deduplicate_documents
from utils.search_utils import (
    optimize_vectorstore_search,
    get_comprehensive_director_info,
)
from config.settings import VECTOR_DB_DIR, embedding_model


# ==== Enhanced Faculty Retriever Class ====
class EnhancedFacultyRetriever:
    """
    Advanced retrieval system that combines multiple search strategies
    for accurate results and proper cache integration.

    Features:
    - Faculty-aware metadata filtering
    - Semantic similarity search
    - Cross-domain search capabilities
    - Relevance scoring and filtering
    - Deduplication

    Purpose: Ensures that when users ask about specific faculty members,
    we retrieve the most relevant documents about that person, not just
    semantically similar content about other people.
    """

    def __init__(self, vectorstore: Chroma, domain: str):
        """Initialize the enhanced retriever."""
        self.vectorstore = vectorstore
        self.domain = domain
        # Use factory function to get properly configured extractor with caching
        self.name_extractor = create_faculty_extractor_with_cache()

    def retrieve_with_faculty_awareness(self, query: str, k: int = 4) -> List[Document]:
        """FIXED: Main retrieval function with PRECISE faculty filtering"""
        extracted_names = self.name_extractor.extract_names(query)
        query_lower = query.lower()

        print(f"🔍 DEBUG - Query: '{query}'")
        print(f"🔍 DEBUG - Extracted names: {extracted_names}")

        # Check if this is a director query
        is_director_query = any(
            pattern in query_lower
            for pattern in [
                "director",
                "current director",
                "nii director",
                "institute director",
                "head of nii",
                "director's background",
                "director research",
            ]
        )

        # If no names extracted, fall back to basic semantic search
        if not extracted_names:
            return self.vectorstore.similarity_search(query, k=k)

        print(f"🎯 Extracted faculty names: {extracted_names}")

        # Special handling for director queries
        if (
            is_director_query
            and "Dr. Debasisa Mohanty" in extracted_names
            and self.domain == "nii_info"
        ):
            return get_comprehensive_director_info(query)

        # ===== STRATEGY 1: STRICT METADATA FILTERING ONLY =====
        all_docs = []

        for name in extracted_names:
            try:
                print(f"🔍 Searching for EXACT faculty match: {name}")

                # CRITICAL: Use ONLY metadata filtering, NO semantic search
                filtered_docs = self.vectorstore.similarity_search(
                    query,
                    k=k * 2,  # Get more docs to filter from
                    filter={"faculty_name": name},  # EXACT match only
                )

                if filtered_docs:
                    print(
                        f"✅ Metadata filter found {len(filtered_docs)} docs for: {name}"
                    )

                    # ADDITIONAL VERIFICATION: Ensure content is actually about this person
                    verified_docs = []
                    for doc in filtered_docs:
                        doc_faculty = doc.metadata.get("faculty_name", "")
                        doc_content = doc.page_content.lower()

                        # STRICT CHECK: Must match exactly
                        if doc_faculty == name and name.lower() in doc_content:
                            verified_docs.append(doc)
                            print(f"   ✅ VERIFIED doc about {name}")
                        else:
                            print(
                                f"   ❌ REJECTED doc - metadata: {doc_faculty}, target: {name}"
                            )

                    if verified_docs:
                        all_docs.extend(verified_docs)
                        print(
                            f"🎯 Final verified docs for {name}: {len(verified_docs)}"
                        )
                        break  # Found exact matches for this faculty, stop searching
                else:
                    print(f"   ⚠️ No metadata matches for: {name}")

            except Exception as e:
                print(f"   ❌ Metadata search failed for {name}: {e}")

        # ===== STRATEGY 2: FALLBACK WITH STRICT POST-FILTERING =====
        if not all_docs:
            print(
                "🔄 No exact metadata matches, trying semantic search with STRICT filtering..."
            )

            for name in extracted_names:
                try:
                    # Use enhanced query for better semantic matching
                    enhanced_query = f"{name} {query}"
                    semantic_docs = self.vectorstore.similarity_search(
                        enhanced_query, k=k * 2
                    )

                    # STRICT POST-FILTERING
                    filtered_semantic = []
                    for doc in semantic_docs:
                        if self._is_doc_about_faculty(doc, name):
                            filtered_semantic.append(doc)
                            print(f"   ✅ Semantic match VERIFIED for {name}")
                        else:
                            doc_faculty = doc.metadata.get("faculty_name", "Unknown")
                            print(
                                f"   ❌ Semantic match REJECTED - doc about {doc_faculty}, target: {name}"
                            )

                    if filtered_semantic:
                        all_docs.extend(filtered_semantic)
                        print(
                            f"🎯 Semantic search found {len(filtered_semantic)} verified docs for {name}"
                        )
                        break  # Found matches for this faculty

                except Exception as e:
                    print(f"   ❌ Semantic search failed for {name}: {e}")

        # ===== STRATEGY 3: CROSS-DOMAIN SEARCH (if current domain fails) =====
        if not all_docs and self.domain != "faculty_info":
            print("🔄 Trying cross-domain search in faculty_info...")
            all_docs = self._cross_domain_search(extracted_names, query, k)

        # Remove duplicates and return ONLY documents about the requested faculty
        unique_docs = self._deduplicate_docs(all_docs)
        final_docs = unique_docs[:k] if unique_docs else []

        print(
            f"🏁 FINAL RESULT: {len(final_docs)} documents about requested faculty ONLY"
        )

        # If still no specific results, return general search as last resort
        if not final_docs:
            print("⚠️ No faculty-specific results found, falling back to general search")
            return self.vectorstore.similarity_search(query, k=k)

        return final_docs

    def _is_doc_about_faculty(self, doc: Document, target_faculty: str) -> bool:
        """
        STRICT check if document is actually about the target faculty member

        Args:
            doc: Document to check
            target_faculty: Name of faculty we're looking for

        Returns:
            bool: True if document is about target faculty
        """
        doc_faculty = doc.metadata.get("faculty_name", "").lower()
        doc_content = doc.page_content.lower()
        target_lower = target_faculty.lower()

        # PRIORITY 1: Exact metadata match
        if doc_faculty == target_faculty:
            return True

        # PRIORITY 2: Target name appears in content
        if target_lower in doc_content:
            # Additional check: ensure it's not just a passing mention
            target_parts = target_lower.replace("dr.", "").strip().split()
            name_mentions = sum(
                1 for part in target_parts if len(part) > 3 and part in doc_content
            )

            if name_mentions >= 2:  # At least 2 name parts must appear
                return True

        # PRIORITY 3: Partial name matching with strict criteria
        if target_lower in doc_faculty:
            return True

        return False

    def _cross_domain_search(
        self, extracted_names: List[str], query: str, k: int
    ) -> List[Document]:
        """Cross-domain search with strict faculty filtering"""
        try:
            faculty_store = Chroma(
                collection_name="faculty_info",
                embedding_function=embedding_model,
                persist_directory=os.path.join(VECTOR_DB_DIR, "faculty_info"),
            )

            cross_docs = []
            for name in extracted_names:
                try:
                    # Use metadata filtering in cross-domain search too
                    domain_docs = faculty_store.similarity_search(
                        query, k=k // 2, filter={"faculty_name": name}
                    )

                    # Verify these are about the right faculty
                    verified_cross = []
                    for doc in domain_docs:
                        if self._is_doc_about_faculty(doc, name):
                            verified_cross.append(doc)

                    if verified_cross:
                        cross_docs.extend(verified_cross)
                        print(
                            f"✅ Cross-domain search found {len(verified_cross)} verified docs for {name}"
                        )
                        break

                except Exception as e:
                    print(f"   ❌ Cross-domain search failed for {name}: {e}")

            return cross_docs

        except Exception as e:
            print(f"⚠️ Cross-domain search initialization failed: {e}")
            return []

    def _filter_by_faculty_relevance(
        self, docs: List[Document], target_name: str
    ) -> List[Document]:
        """Filter and score documents by relevance to a specific faculty member."""
        relevant_docs = []
        target_lower = target_name.lower()

        for doc in docs:
            if self._is_doc_about_faculty(doc, target_name):
                faculty_name = doc.metadata.get("faculty_name", "").lower()
                content = doc.page_content.lower()

                score = 0
                if target_lower in faculty_name:
                    score += 3
                if target_lower in content:
                    score += 1

                name_parts = target_lower.replace("dr.", "").strip().split()
                for part in name_parts:
                    if len(part) > 2:
                        if part in faculty_name:
                            score += 2
                        if part in content:
                            score += 1

                relevant_docs.append((doc, score))

        relevant_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in relevant_docs]

    def _deduplicate_docs(self, docs: List[Document]) -> List[Document]:
        """Remove duplicate documents using the unified deduplication function."""
        return _deduplicate_documents(docs)
