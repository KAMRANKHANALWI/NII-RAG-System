import re
from typing import List, Optional, Callable
from functools import lru_cache

# Import centralized faculty data
from config.faculty_data import (
    KNOWN_FACULTY,
    AMBIGUOUS_NAMES,
    FIRST_NAMES,
    SPECIAL_POSITIONS,
    FALSE_POSITIVES,
)


def _fallback_query_preprocessing(query_lower: str) -> tuple:
    """
    Core query analysis logic shared between cached and non-cached implementations.

    Args:
        query_lower (str): Lowercase query string for pattern matching

    Returns:
        tuple: Boolean flags (has_pronouns, has_exit_words, has_faculty_list_patterns, has_director_patterns)

    Architecture Note: This function exists outside the class to prevent code duplication
    between the LRU-cached version and the fallback version. This design ensures
    consistent behavior regardless of caching availability.

    Purpose: Avoid repeated regex operations and string searches for the same queries.
    This significantly speeds up repeated queries about the same topics when cached.
    """
    pronouns = ["his", "her", "their", "its", "he", "she", "they"]
    has_pronouns = any(pronoun in query_lower.split() for pronoun in pronouns)

    exit_words = ["bye", "goodbye", "exit", "quit", "thanks", "thank you"]
    has_exit_words = any(word in query_lower for word in exit_words)

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


def _fallback_name_lookup(
    name_lower: str, lookup_type: str, extractor_instance
) -> Optional[str]:
    """
    Core name resolution logic shared between cached and non-cached implementations.

    Args:
        name_lower (str): Normalized name string for lookup
        lookup_type (str): Database type ("known_faculty", "first_names", "ambiguous")
        extractor_instance (FacultyNameExtractor): Reference to extractor instance for database access

    Returns:
        Optional[str]: Matched faculty name or None if not found

    Architecture Note: Requires extractor_instance parameter to access faculty databases.
    This design maintains single responsibility while enabling dependency injection.

    Purpose: Centralized name resolution logic that can be used with or without caching,
    eliminating code duplication between cached and non-cached implementations.
    """
    if lookup_type == "known_faculty":
        return extractor_instance.known_faculty.get(name_lower)
    elif lookup_type == "first_names":
        return extractor_instance.first_names.get(name_lower)
    elif lookup_type == "ambiguous":
        return extractor_instance.ambiguous_names.get(name_lower)
    return None


class FacultyNameExtractor:
    """
    Faculty name recognition system with dependency injection pattern.

    Design Rationale:
    - Uses dependency injection to enable LRU caching without circular imports
    - Falls back gracefully to non-cached implementations
    - Maintains consistent behavior across different deployment scenarios
    - Separates data from logic for better maintainability
    """

    def __init__(
        self,
        query_preprocessor: Optional[Callable] = None,
        name_lookup: Optional[Callable] = None,
    ):
        """
        Initialize faculty name extractor with optional dependency injection for performance optimization.

        Args:
            query_preprocessor (Optional[Callable]): External query analysis function (usually cached)
            name_lookup (Optional[Callable]): External name resolution function (usually cached)

        Returns:
            None

        Architecture: Constructor accepts external functions to enable caching strategies
        without tight coupling to specific implementations. Falls back to shared logic
        when no external functions provided.

        Purpose: Enable high-performance caching when available while maintaining
        functionality when caching is unavailable due to import restrictions.
        """
        # Load faculty data from external configuration
        self.known_faculty = KNOWN_FACULTY
        self.ambiguous_names = AMBIGUOUS_NAMES
        self.first_names = FIRST_NAMES
        self.special_positions = SPECIAL_POSITIONS
        self.false_positives = FALSE_POSITIVES

        # Dependency injection pattern: Use provided functions or fall back to defaults
        self.query_preprocessor = (
            query_preprocessor if query_preprocessor else _fallback_query_preprocessing
        )

        # Lambda wrapper maintains interface consistency while passing instance reference
        self.name_lookup = (
            name_lookup
            if name_lookup
            else lambda name, lookup_type: _fallback_name_lookup(
                name, lookup_type, self
            )
        )

    @lru_cache(maxsize=512)
    def _cached_extract_names(self, query: str) -> tuple:
        """
        Internal caching layer for name extraction results.

        Args:
            query (str): User query string

        Returns:
            tuple: Extracted faculty names as tuple (required for lru_cache hashability)

        Performance: Prevents redundant processing of identical queries within the same session.
        Note: Returns tuple for hashability required by lru_cache.
        """
        return tuple(self._extract_names_internal(query))

    def extract_names(self, query: str) -> List[str]:
        """
        Main entry point for faculty name extraction from user queries.

        Args:
            query (str): User's natural language query

        Returns:
            List[str]: List of matched faculty names in standardized format

        Architecture: Delegates to cached internal method, then converts to list format
        expected by downstream components.

        Purpose: Provides clean public API while leveraging internal caching mechanisms
        for optimal performance on repeated queries.
        """
        cached_result = self._cached_extract_names(query)
        return list(cached_result)

    def _extract_names_internal(self, query: str) -> List[str]:
        """
        Core name extraction algorithm using injected preprocessing functions.

        Args:
            query (str): User's natural language query

        Returns:
            List[str]: List of extracted faculty names

        Strategy:
        1. Early exit for non-person queries (exit commands, faculty lists)
        2. Special handling for director queries
        3. Exact match lookup in known faculty database
        4. First name matching with context validation
        5. Ambiguous name disambiguation

        Design: Uses injected functions to enable caching without code duplication.
        """
        names = []
        query_lower = query.lower()

        # Query classification using injected preprocessing function
        (
            has_pronouns,
            has_exit_words,
            has_faculty_list_patterns,
            has_director_patterns,
        ) = self.query_preprocessor(query_lower)

        # Early termination for non-person queries
        if has_exit_words or has_faculty_list_patterns:
            return []

        # Special case: Director queries always resolve to current director
        if has_director_patterns:
            names.append(self.special_positions.get("director"))
            return names

        # Primary strategy: Exact match in comprehensive faculty database
        for known_name in self.known_faculty.keys():
            if known_name in query_lower:
                result = self.name_lookup(known_name, "known_faculty")
                if result:
                    names.append(result)
                    return names

        # Secondary strategy: First name matches with context validation
        for first_name in self.first_names.keys():
            if first_name in query_lower and self._is_valid_name_context(
                query_lower, first_name
            ):
                result = self.name_lookup(first_name, "first_names")
                if result:
                    names.append(result)
                    return names

        # Tertiary strategy: Ambiguous name handling with disambiguation
        for ambiguous_name in self.ambiguous_names.keys():
            if ambiguous_name in query_lower:
                candidates = self.name_lookup(ambiguous_name, "ambiguous")
                if candidates:
                    disambiguated = self._disambiguate_names(query, candidates)
                    if disambiguated:
                        names.extend(disambiguated)
                        return names

        return list(set(names))

    def _is_valid_name_context(self, query_lower: str, name: str) -> bool:
        """
        Validate that a name appears in a valid context using false positive detection.

        Args:
            query_lower (str): Lowercase query string
            name (str): Name to validate context for

        Returns:
            bool: True if name appears in valid context, False otherwise

        Purpose: Prevents false positive matches by checking if the query contains
        known false positive phrases that indicate non-person contexts.
        """
        # Use centralized false positive detection instead of redundant indicators
        for false_positive in self.false_positives:
            if false_positive in query_lower:
                return False

        # Additional check: if name appears in context with academic titles, it's valid
        academic_indicators = ["dr ", "prof ", "professor ", "doctor "]
        name_context = query_lower[
            max(0, query_lower.find(name) - 10) : query_lower.find(name)
            + len(name)
            + 10
        ]

        return (
            any(indicator in name_context for indicator in academic_indicators)
            or name in query_lower
        )

    def _disambiguate_names(self, query: str, candidates: List[str]) -> List[str]:
        """
        Resolve ambiguous names using contextual clues from the query.

        Args:
            query (str): Original user query
            candidates (List[str]): List of possible faculty matches

        Returns:
            List[str]: Disambiguated faculty names or original candidates if no clear winner

        Purpose: When surnames match multiple faculty, use domain knowledge to pick the most likely candidate.
        """
        query_lower = query.lower()

        # Domain-specific disambiguation clues
        disambiguation_clues = {
            "Dr. Sarika Gupta": ["neurodegenerative", "protein misfolding", "sarika"],
            "Dr. Nimesh Gupta": ["virus", "immunology", "vaccine", "nimesh"],
            "Dr. Anil Kumar": ["microbiome", "cancer", "anil"],
            "Dr. Rajesh Kumar Yadav": ["epigenetics", "cancer biology", "rajesh"],
            "Dr. Narendra Kumar": ["narendra", "narender"],
        }

        best_candidate = None
        best_score = 0

        for candidate in candidates:
            score = 0
            clues = disambiguation_clues.get(candidate, [])
            for clue in clues:
                if clue in query_lower:
                    score += 2 if len(clue) > 5 else 1

            if score > best_score:
                best_score = score
                best_candidate = candidate

        return [best_candidate] if best_candidate and best_score > 0 else candidates


def create_faculty_extractor_with_cache():
    """
    Factory function implementing dependency injection with graceful degradation.

    Returns:
        FacultyNameExtractor: Extractor instance with caching enabled when available

    Architecture Decision: This pattern solves circular import issues while maintaining
    optimal performance when caching is available. The factory abstracts the complexity
    of dependency resolution from client code.

    Fallback Strategy: If cached functions unavailable (due to import issues),
    falls back to shared logic without caching - ensuring system reliability.

    Purpose: Primary factory function that provides best available performance
    while maintaining system stability across different deployment scenarios.
    """
    try:
        from core.caching import _cached_query_preprocessing, _cached_name_lookup

        print("✅ Using CACHED functions for optimal performance")
        return FacultyNameExtractor(
            query_preprocessor=_cached_query_preprocessing,
            name_lookup=_cached_name_lookup,
        )
    except ImportError:
        print("⚠️ Using FALLBACK functions (shared logic, no caching)")
        return FacultyNameExtractor()


def create_faculty_extractor_no_cache():
    """
    Factory function for explicit non-cached extractor creation.

    Returns:
        FacultyNameExtractor: Extractor instance without caching

    Use Cases:
    - Testing environments requiring deterministic behavior
    - Memory-constrained systems avoiding cache overhead
    - Debugging scenarios where cache effects need isolation
    - Development environments with frequent code changes

    Purpose: Provides guaranteed non-cached behavior when caching side effects
    are undesirable or when maximum memory efficiency is required.

    Note: Uses same shared logic as cached version, ensuring consistent results.
    """
    return FacultyNameExtractor()  # Always uses fallback functions
