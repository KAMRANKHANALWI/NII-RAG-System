from functools import lru_cache
from typing import Optional


# ==== Performance Caching System ====
@lru_cache(maxsize=128)
def _cached_query_preprocessing(query_lower: str) -> tuple:
    """
    Cache expensive query preprocessing operations.

    Args:
        query_lower (str): Lowercase query string

    Returns:
        tuple: (has_pronouns, has_exit_words, has_faculty_list_patterns, has_director_patterns)

    Purpose: Avoid repeated regex operations and string searches for the same queries.
    This significantly speeds up repeated queries about the same topics.
    """
    # Check for pronouns
    pronouns = ["his", "her", "their", "its", "he", "she", "they"]
    has_pronouns = any(pronoun in query_lower.split() for pronoun in pronouns)

    # Check for exit words
    exit_words = ["bye", "goodbye", "exit", "quit", "thanks", "thank you"]
    has_exit_words = any(word in query_lower for word in exit_words)

    # Check for faculty list patterns
    faculty_list_patterns = [
        "faculty list",
        "list of faculty",
        "all faculty",
        "faculty members",
        "current faculty",
        "show faculty",
        "institute faculty",
    ]
    has_faculty_list_patterns = any(
        pattern in query_lower for pattern in faculty_list_patterns
    )

    # Check for director patterns
    director_patterns = [
        "director",
        "current director",
        "nii director",
        "institute director",
    ]
    has_director_patterns = any(pattern in query_lower for pattern in director_patterns)

    return (
        has_pronouns,
        has_exit_words,
        has_faculty_list_patterns,
        has_director_patterns,
    )


@lru_cache(maxsize=256)
def _cached_name_lookup(name_lower: str, lookup_type: str) -> Optional[str]:
    """
    Cache faculty name lookups to avoid repeated dictionary searches.

    Args:
        name_lower (str): Lowercase name to lookup
        lookup_type (str): Type of lookup ("known_faculty", "first_names", "ambiguous")

    Returns:
        Optional[str]: Found faculty name or None

    Purpose: Since faculty databases are static, we can cache lookups indefinitely.
    This is particularly useful for common names that get queried repeatedly.
    """
    # Lazy import to avoid circular imports
    # Initialize the extractor once and reuse its databases
    if not hasattr(_cached_name_lookup, "_extractor"):
        try:
            # Import here to avoid circular imports
            from core.faculty_extractor import FacultyNameExtractor

            _cached_name_lookup._extractor = FacultyNameExtractor()
        except ImportError as e:
            print(f"⚠️ Could not import FacultyNameExtractor: {e}")
            return None

    extractor = _cached_name_lookup._extractor

    try:
        if lookup_type == "known_faculty":
            return extractor.known_faculty.get(name_lower)
        elif lookup_type == "first_names":
            return extractor.first_names.get(name_lower)
        elif lookup_type == "ambiguous":
            return extractor.ambiguous_names.get(name_lower)
    except AttributeError as e:
        print(f"⚠️ Error accessing extractor attributes: {e}")
        return None

    return None
