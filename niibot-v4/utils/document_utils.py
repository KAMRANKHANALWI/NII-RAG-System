from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

# ==== Utility Functions ====


# ==== Unified Document Processing ====
def _deduplicate_documents(docs: List[Document]) -> List[Document]:
    """
    UNIFIED deduplication function for all document types.

    Args:
        docs (List[Document]): List of documents that may contain duplicates

    Returns:
        List[Document]: List with duplicates removed

    Method: Uses first 150 characters as content fingerprint for deduplication.
    This replaces multiple similar functions throughout the codebase.
    """
    if not docs:
        return []

    seen_content = set()
    unique_docs = []

    for doc in docs:
        # Create content identifier from first 150 characters
        content_id = doc.page_content[:150].replace("\n", " ").strip()
        if content_id not in seen_content:
            seen_content.add(content_id)
            unique_docs.append(doc)

    return unique_docs


def format_docs(docs, max_docs=3, max_chars_per_doc=800):
    """
    Format retrieved documents for inclusion in LLM prompts with integrated truncation.

    Args:
        docs: List of retrieved documents
        max_docs (int): Maximum number of documents to include
        max_chars_per_doc (int): Maximum characters per document

    Returns:
        str: Formatted string with numbered documents, ready for LLM

    Purpose: Combines document formatting and truncation in one optimized function.
    """
    if not docs:
        return "No relevant documents found."

    formatted_docs = []
    for i, doc in enumerate(docs[:max_docs]):
        # Truncate long documents to prevent token limits
        content = (
            doc.page_content[:max_chars_per_doc] + "..."
            if len(doc.page_content) > max_chars_per_doc
            else doc.page_content
        )
        formatted_docs.append(f"Document {i+1}:\n{content}")

    return "\n\n".join(formatted_docs)


def get_metadata_value(doc, keys, default="Unknown"):
    """
    Helper function to get metadata values with fallback keys.

    Args:
        doc: Document object
        keys (list): List of possible metadata keys to check
        default (str): Default value if none found

    Returns:
        str: First found metadata value or default

    Purpose: Eliminates repetitive metadata.get() chains throughout the code.
    """
    for key in keys:
        value = doc.metadata.get(key)
        if value:
            return value
    return default


def display_sources(docs):
    """
    Display source information for retrieved documents to provide transparency.

    Args:
        docs: List of retrieved documents

    Purpose: Shows users where the information came from, enhancing trust
    and allowing them to verify information or explore further.
    """
    if not docs:
        print("📎 No sources found")
        return

    print("📎 Sources:")
    for i, doc in enumerate(docs[:5], 1):  # Show up to 5 sources
        # Use helper function to reduce redundancy
        faculty_name = get_metadata_value(doc, ["faculty_name", "name"])
        source = get_metadata_value(doc, ["source", "Source"], "No link")
        title = doc.metadata.get("title", "")
        category = get_metadata_value(doc, ["category", "chunk_type"])

        # Enhanced source display with title if available
        source_info = f"{title} - " if title else ""
        print(f"   {i}. {source_info}{faculty_name} ({category}) — {source}")
