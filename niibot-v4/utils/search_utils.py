# import os
# from typing import List
# from langchain_chroma import Chroma
# from langchain_core.documents import Document

# from config.settings import VECTOR_DB_DIR, embedding_model
# from utils.document_utils import _deduplicate_documents


# def optimize_vectorstore_search(
#     vectorstore, query: str, k: int, strategies: dict
# ) -> List[Document]:
#     """
#     UNIFIED search function that handles multiple search strategies efficiently.

#     Args:
#         vectorstore: Chroma vectorstore instance
#         query (str): Search query
#         k (int): Number of documents to retrieve
#         strategies (dict): Dictionary of search strategies with their parameters

#     Returns:
#         List[Document]: Combined and deduplicated results from all strategies

#     Purpose: Eliminates redundant search code patterns used throughout the system.
#     """
#     all_docs = []

#     for strategy_name, params in strategies.items():
#         try:
#             if strategy_name == "metadata_filter":
#                 for filter_criteria in params:
#                     docs = vectorstore.similarity_search(
#                         query, k=k, filter=filter_criteria
#                     )
#                     if docs:
#                         print(f"   ✅ {strategy_name} found {len(docs)} docs")
#                         all_docs.extend(docs)

#             elif strategy_name == "content_search":
#                 for search_term in params:
#                     docs = vectorstore.similarity_search(search_term, k=k)
#                     all_docs.extend(docs)

#             elif strategy_name == "semantic_search":
#                 docs = vectorstore.similarity_search(query, k=k)
#                 all_docs.extend(docs)

#         except Exception as e:
#             print(f"   ⚠️ {strategy_name} failed: {e}")

#     return _deduplicate_documents(all_docs)


# def get_comprehensive_director_info(query: str) -> List[Document]:
#     """
#     Gather comprehensive director information from multiple collections/sources.
#     NOW USES UNIFIED SEARCH FUNCTION to reduce code duplication.

#     Args:
#         query (str): User query about the director

#     Returns:
#         List[Document]: Comprehensive set of documents about the director

#     Purpose: Director queries need special handling because information about
#     the director is scattered across multiple collections. Uses optimized
#     search strategies to reduce redundant code.
#     """
#     print("🔍 Gathering comprehensive director information from multiple sources...")

#     all_docs = []
#     director_name = "Dr. Debasisa Mohanty"

#     # Define collections to search
#     collections = ["nii_info", "faculty_info", "research", "publications"]

#     for collection_name in collections:
#         try:
#             collection_path = os.path.join(VECTOR_DB_DIR, collection_name)
#             vectorstore = Chroma(
#                 collection_name=collection_name,
#                 embedding_function=embedding_model,
#                 persist_directory=collection_path,
#             )

#             print(f"📂 Searching {collection_name} collection...")

#             # Define search strategies for this collection
#             if collection_name == "nii_info":
#                 strategies = {
#                     "metadata_filter": [{"faculty_name": director_name}],
#                     "content_search": [
#                         "directors page",
#                         "appointed as director",
#                         "director of the institute",
#                         "Debasisa Mohanty director",
#                     ],
#                 }
#             else:
#                 strategies = {"metadata_filter": [{"faculty_name": director_name}]}

#             # Use unified search function
#             collection_docs = optimize_vectorstore_search(
#                 vectorstore, query, 3, strategies
#             )
#             all_docs.extend(collection_docs)

#         except Exception as e:
#             print(f"❌ Error accessing {collection_name}: {e}")
#             continue

#     # Process and prioritize results
#     unique_docs = _deduplicate_documents(all_docs)
#     prioritized_docs = _prioritize_director_documents(unique_docs)

#     print(
#         f"🎯 Comprehensive search found {len(prioritized_docs)} unique director documents"
#     )
#     return prioritized_docs[:6]


# def handle_multiple_candidates(
#     candidates: List[str], query: str, domain: str
# ) -> List[Document]:
#     """
#     Handle queries that match multiple faculty members by searching for all of them.

#     Args:
#         candidates (List[str]): List of faculty names to search for
#         query (str): Original user query
#         domain (str): Domain/collection to search in

#     Returns:
#         List[Document]: Combined documents from all candidates

#     Purpose: When ambiguous queries match multiple people, gather information
#     about all candidates and let the LLM decide which is most relevant.
#     """
#     print(f"🤔 Found multiple candidates: {candidates}")
#     print("🔍 Searching for all candidates...")

#     all_docs = []

#     try:
#         collection_path = os.path.join(VECTOR_DB_DIR, domain)
#         vectorstore = Chroma(
#             collection_name=domain,
#             embedding_function=embedding_model,
#             persist_directory=collection_path,
#         )

#         # Search for each candidate individually
#         for candidate in candidates:
#             try:
#                 docs = vectorstore.similarity_search(
#                     query, k=2, filter={"faculty_name": candidate}
#                 )
#                 if docs:
#                     print(f"   ✅ Found {len(docs)} docs for {candidate}")
#                     all_docs.extend(docs)
#             except Exception as e:
#                 print(f"   ⚠️ Search failed for {candidate}: {e}")

#     except Exception as e:
#         print(f"❌ Error in multi-candidate search: {e}")

#     return all_docs


# def _prioritize_director_documents(docs: List[Document]) -> List[Document]:
#     """
#     Prioritize documents to ensure the most important director information comes first.

#     Args:
#         docs (List[Document]): List of director-related documents

#     Returns:
#         List[Document]: Prioritized list with most important documents first

#     Priority order:
#     1. Director page documents (official role, appointment info)
#     2. Faculty profile documents (basic info, contact)
#     3. Research documents (current projects)
#     4. Publication documents (research output)
#     5. Other relevant documents
#     """
#     # Categorize documents by type and importance
#     director_page_docs = []
#     faculty_docs = []
#     research_docs = []
#     publication_docs = []
#     other_docs = []

#     for doc in docs:
#         content_lower = doc.page_content.lower()
#         title_lower = doc.metadata.get("title", "").lower()
#         source = doc.metadata.get("source", "").lower()
#         chunk_type = doc.metadata.get("chunk_type", "").lower()

#         # Categorize based on content and metadata
#         if "directors page" in title_lower or (
#             "director" in content_lower
#             and ("appointed" in content_lower or "august 2022" in content_lower)
#         ):
#             director_page_docs.append(doc)
#         elif "faculty" in chunk_type or "faculty" in source or "profile" in chunk_type:
#             faculty_docs.append(doc)
#         elif "research" in chunk_type or "research" in source:
#             research_docs.append(doc)
#         elif "publication" in chunk_type or "publication" in source:
#             publication_docs.append(doc)
#         else:
#             other_docs.append(doc)

#     # Prioritize: Director page first, then faculty, research, publications, others
#     prioritized = []
#     prioritized.extend(director_page_docs[:2])  # Max 2 director page docs
#     prioritized.extend(faculty_docs[:1])  # 1 faculty profile
#     prioritized.extend(research_docs[:1])  # 1 research summary
#     prioritized.extend(publication_docs[:2])  # 2 publication docs
#     prioritized.extend(other_docs[:1])  # 1 other doc

#     return prioritized

import os
from typing import List
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config.settings import VECTOR_DB_DIR, embedding_model
from utils.document_utils import _deduplicate_documents


def optimize_vectorstore_search(
    vectorstore, query: str, k: int, strategies: dict
) -> List[Document]:
    """
    UNIFIED search function that handles multiple search strategies efficiently.

    Args:
        vectorstore: Chroma vectorstore instance
        query (str): Search query
        k (int): Number of documents to retrieve
        strategies (dict): Dictionary of search strategies with their parameters

    Returns:
        List[Document]: Combined and deduplicated results from all strategies

    Purpose: Eliminates redundant search code patterns used throughout the system.
    """
    all_docs = []

    for strategy_name, params in strategies.items():
        try:
            if strategy_name == "metadata_filter":
                for filter_criteria in params:
                    docs = vectorstore.similarity_search(
                        query, k=k, filter=filter_criteria
                    )
                    if docs:
                        print(f"   ✅ {strategy_name} found {len(docs)} docs")
                        all_docs.extend(docs)

            elif strategy_name == "content_search":
                for search_term in params:
                    docs = vectorstore.similarity_search(search_term, k=k)
                    all_docs.extend(docs)

            elif strategy_name == "semantic_search":
                docs = vectorstore.similarity_search(query, k=k)
                all_docs.extend(docs)

        except Exception as e:
            print(f"   ⚠️ {strategy_name} failed: {e}")

    return _deduplicate_documents(all_docs)


def get_multi_domain_faculty_info(query: str, faculty_name: str) -> List[Document]:
    """
    NEW: Search across multiple domains for comprehensive faculty information
    Similar to get_comprehensive_director_info but for any faculty member

    Args:
        query (str): User's query
        faculty_name (str): Name of faculty member to search for

    Returns:
        List[Document]: Combined documents from multiple domains
    """
    print(f"🔍 Multi-domain search for: {faculty_name}")

    all_docs = []
    query_lower = query.lower()

    # Determine which domains to search based on query keywords
    search_domains = {}

    # Always search faculty_info for basic profile
    search_domains["faculty_info"] = {
        "priority": 1,
        "max_docs": 2,
        "strategies": {"metadata_filter": [{"faculty_name": faculty_name}]},
    }

    # Search publications if query mentions publications/papers
    if any(
        word in query_lower
        for word in [
            "publications",
            "papers",
            "articles",
            "research papers",
            "published",
        ]
    ):
        search_domains["publications"] = {
            "priority": 2,
            "max_docs": 3,
            "strategies": {"metadata_filter": [{"faculty_name": faculty_name}]},
        }

    # Search research if query mentions research/interests
    if any(
        word in query_lower
        for word in ["research", "interests", "working on", "projects", "focus"]
    ):
        search_domains["research"] = {
            "priority": 2,
            "max_docs": 2,
            "strategies": {"metadata_filter": [{"faculty_name": faculty_name}]},
        }

    # Search labs if query mentions lab/team
    if any(word in query_lower for word in ["lab", "laboratory", "team", "group"]):
        search_domains["labs"] = {
            "priority": 2,
            "max_docs": 2,
            "strategies": {"metadata_filter": [{"faculty_name": faculty_name}]},
        }

    # Search staff if query mentions contact info
    if any(
        word in query_lower
        for word in [
            "email",
            "phone",
            "contact",
            "extension",
            "education",
            "background",
        ]
    ):
        search_domains["staff"] = {
            "priority": 2,
            "max_docs": 1,
            "strategies": {"metadata_filter": [{"faculty_name": faculty_name}]},
        }

    print(f"🎯 Will search domains: {list(search_domains.keys())}")

    # Search each domain
    for domain_name, config in search_domains.items():
        try:
            collection_path = os.path.join(VECTOR_DB_DIR, domain_name)
            if not os.path.exists(collection_path):
                print(f"   ⚠️ Domain {domain_name} not found, skipping...")
                continue

            vectorstore = Chroma(
                collection_name=domain_name,
                embedding_function=embedding_model,
                persist_directory=collection_path,
            )

            print(f"📂 Searching {domain_name} domain...")

            # Use STRICT metadata filtering for multi-domain search
            try:
                domain_docs = vectorstore.similarity_search(
                    query,
                    k=config["max_docs"],
                    filter={"faculty_name": faculty_name},  # EXACT match
                )

                if domain_docs:
                    print(f"   ✅ Found {len(domain_docs)} docs in {domain_name}")
                    # Add domain info to metadata for better tracking
                    for doc in domain_docs:
                        doc.metadata["search_domain"] = domain_name
                        doc.metadata["priority"] = config["priority"]

                    all_docs.extend(domain_docs)
                else:
                    print(f"   ⚠️ No docs found in {domain_name} for {faculty_name}")

            except Exception as e:
                print(f"   ❌ Error searching {domain_name}: {e}")
                continue

        except Exception as e:
            print(f"   ❌ Error accessing {domain_name}: {e}")
            continue

    # Remove duplicates and prioritize
    unique_docs = _deduplicate_documents(all_docs)
    prioritized_docs = _prioritize_faculty_documents(unique_docs, faculty_name)

    print(f"🎯 Multi-domain search found {len(prioritized_docs)} total documents")
    return prioritized_docs[:8]  # Return up to 8 documents for comprehensive info


def _prioritize_faculty_documents(
    docs: List[Document], faculty_name: str
) -> List[Document]:
    """
    Prioritize documents for comprehensive faculty information

    Args:
        docs: List of documents from multiple domains
        faculty_name: Name of faculty for verification

    Returns:
        List[Document]: Prioritized document list
    """
    faculty_docs = []
    research_docs = []
    publication_docs = []
    lab_docs = []
    staff_docs = []
    other_docs = []

    for doc in docs:
        search_domain = doc.metadata.get("search_domain", "")
        chunk_type = doc.metadata.get("chunk_type", "")

        # Verify document is about the right faculty
        doc_faculty = doc.metadata.get("faculty_name", "")
        if doc_faculty != faculty_name:
            print(f"   ⚠️ Skipping doc about {doc_faculty} (looking for {faculty_name})")
            continue

        # Categorize by domain
        if search_domain == "faculty_info" or "faculty" in chunk_type:
            faculty_docs.append(doc)
        elif search_domain == "research" or "research" in chunk_type:
            research_docs.append(doc)
        elif search_domain == "publications" or "publication" in chunk_type:
            publication_docs.append(doc)
        elif search_domain == "labs" or "lab" in chunk_type:
            lab_docs.append(doc)
        elif search_domain == "staff" or "staff" in chunk_type:
            staff_docs.append(doc)
        else:
            other_docs.append(doc)

    # Prioritize: Faculty profile first, then based on query relevance
    prioritized = []
    prioritized.extend(faculty_docs[:1])  # 1 faculty profile
    prioritized.extend(research_docs[:2])  # 2 research docs
    prioritized.extend(publication_docs[:3])  # 3 publication docs
    prioritized.extend(lab_docs[:1])  # 1 lab doc
    prioritized.extend(staff_docs[:1])  # 1 staff doc
    prioritized.extend(other_docs[:1])  # 1 other doc

    return prioritized


def get_comprehensive_director_info(query: str) -> List[Document]:
    """
    Gather comprehensive director information from multiple collections/sources.
    NOW USES UNIFIED SEARCH FUNCTION to reduce code duplication.

    Args:
        query (str): User query about the director

    Returns:
        List[Document]: Comprehensive set of documents about the director

    Purpose: Director queries need special handling because information about
    the director is scattered across multiple collections. Uses optimized
    search strategies to reduce redundant code.
    """
    print("🔍 Gathering comprehensive director information from multiple sources...")

    all_docs = []
    director_name = "Dr. Debasisa Mohanty"

    # Define collections to search
    collections = ["nii_info", "faculty_info", "research", "publications"]

    for collection_name in collections:
        try:
            collection_path = os.path.join(VECTOR_DB_DIR, collection_name)
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=embedding_model,
                persist_directory=collection_path,
            )

            print(f"📂 Searching {collection_name} collection...")

            # Define search strategies for this collection
            if collection_name == "nii_info":
                strategies = {
                    "metadata_filter": [{"faculty_name": director_name}],
                    "content_search": [
                        "directors page",
                        "appointed as director",
                        "director of the institute",
                        "Debasisa Mohanty director",
                    ],
                }
            else:
                strategies = {"metadata_filter": [{"faculty_name": director_name}]}

            # Use unified search function
            collection_docs = optimize_vectorstore_search(
                vectorstore, query, 3, strategies
            )
            all_docs.extend(collection_docs)

        except Exception as e:
            print(f"❌ Error accessing {collection_name}: {e}")
            continue

    # Process and prioritize results
    unique_docs = _deduplicate_documents(all_docs)
    prioritized_docs = _prioritize_director_documents(unique_docs)

    print(
        f"🎯 Comprehensive search found {len(prioritized_docs)} unique director documents"
    )
    return prioritized_docs[:6]


def handle_multiple_candidates(
    candidates: List[str], query: str, domain: str
) -> List[Document]:
    """
    Handle queries that match multiple faculty members by searching for all of them.

    Args:
        candidates (List[str]): List of faculty names to search for
        query (str): Original user query
        domain (str): Domain/collection to search in

    Returns:
        List[Document]: Combined documents from all candidates

    Purpose: When ambiguous queries match multiple people, gather information
    about all candidates and let the LLM decide which is most relevant.
    """
    print(f"🤔 Found multiple candidates: {candidates}")
    print("🔍 Searching for all candidates...")

    all_docs = []

    try:
        collection_path = os.path.join(VECTOR_DB_DIR, domain)
        vectorstore = Chroma(
            collection_name=domain,
            embedding_function=embedding_model,
            persist_directory=collection_path,
        )

        # Search for each candidate individually
        for candidate in candidates:
            try:
                docs = vectorstore.similarity_search(
                    query, k=2, filter={"faculty_name": candidate}
                )
                if docs:
                    print(f"   ✅ Found {len(docs)} docs for {candidate}")
                    all_docs.extend(docs)
            except Exception as e:
                print(f"   ⚠️ Search failed for {candidate}: {e}")

    except Exception as e:
        print(f"❌ Error in multi-candidate search: {e}")

    return all_docs


def _prioritize_director_documents(docs: List[Document]) -> List[Document]:
    """
    Prioritize documents to ensure the most important director information comes first.

    Args:
        docs (List[Document]): List of director-related documents

    Returns:
        List[Document]: Prioritized list with most important documents first

    Priority order:
    1. Director page documents (official role, appointment info)
    2. Faculty profile documents (basic info, contact)
    3. Research documents (current projects)
    4. Publication documents (research output)
    5. Other relevant documents
    """
    # Categorize documents by type and importance
    director_page_docs = []
    faculty_docs = []
    research_docs = []
    publication_docs = []
    other_docs = []

    for doc in docs:
        content_lower = doc.page_content.lower()
        title_lower = doc.metadata.get("title", "").lower()
        source = doc.metadata.get("source", "").lower()
        chunk_type = doc.metadata.get("chunk_type", "").lower()

        # Categorize based on content and metadata
        if "directors page" in title_lower or (
            "director" in content_lower
            and ("appointed" in content_lower or "august 2022" in content_lower)
        ):
            director_page_docs.append(doc)
        elif "faculty" in chunk_type or "faculty" in source or "profile" in chunk_type:
            faculty_docs.append(doc)
        elif "research" in chunk_type or "research" in source:
            research_docs.append(doc)
        elif "publication" in chunk_type or "publication" in source:
            publication_docs.append(doc)
        else:
            other_docs.append(doc)

    # Prioritize: Director page first, then faculty, research, publications, others
    prioritized = []
    prioritized.extend(director_page_docs[:2])  # Max 2 director page docs
    prioritized.extend(faculty_docs[:1])  # 1 faculty profile
    prioritized.extend(research_docs[:1])  # 1 research summary
    prioritized.extend(publication_docs[:2])  # 2 publication docs
    prioritized.extend(other_docs[:1])  # 1 other doc

    return prioritized
