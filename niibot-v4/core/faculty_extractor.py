"""Enhanced Faculty Name Extractor with Multi-Strategy Recognition.

This module provides sophisticated faculty name recognition with comprehensive
pattern matching, designed to work seamlessly with the 5-tier retrieval system.

Key Features:
    - Multi-database name recognition (KNOWN_FACULTY, FIRST_NAMES, AMBIGUOUS_NAMES)
    - Enhanced partial matching for incomplete queries
    - Context validation to prevent false positives
    - Dependency injection pattern for optimal caching
    - Graceful degradation with fallback implementations
"""

import re
from typing import List, Optional, Callable
from functools import lru_cache

from config.faculty_data import (
    KNOWN_FACULTY,
    AMBIGUOUS_NAMES,
    FIRST_NAMES,
    SPECIAL_POSITIONS,
    FALSE_POSITIVES,
)


def _fallback_query_preprocessing(query_lower: str) -> tuple:
    """Core query analysis logic for pattern recognition.

    Analyzes query patterns to determine processing strategy and early
    exit conditions for non-person queries.

    Args:
        query_lower (str): Lowercase version of user query

    Returns:
        tuple: (has_pronouns, has_exit_words, has_faculty_list_patterns,
               has_director_patterns)
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
    """Core name resolution logic with database access.

    Provides centralized lookup across different faculty databases
    with consistent interface for cached and non-cached implementations.

    Args:
        name_lower (str): Lowercase name to lookup
        lookup_type (str): Type of lookup ('known_faculty', 'first_names', 'ambiguous')
        extractor_instance: Instance of FacultyNameExtractor

    Returns:
        Optional[str]: Full faculty name if found, None otherwise
    """
    if lookup_type == "known_faculty":
        return extractor_instance.known_faculty.get(name_lower)
    elif lookup_type == "first_names":
        return extractor_instance.first_names.get(name_lower)
    elif lookup_type == "ambiguous":
        return extractor_instance.ambiguous_names.get(name_lower)
    return None


class FacultyNameExtractor:
    """Advanced faculty name recognition system with multi-strategy approach.

    Implements comprehensive name extraction using multiple databases and
    recognition strategies to handle various query patterns:
        - Formal names: "Dr. Monica Sundd"
        - Casual names: "monica", "sarika"
        - Partial names: "dr monica", "tanmay research"
        - Complex patterns: Ambiguous surnames with context disambiguation
    """

    def __init__(
        self,
        query_preprocessor: Optional[Callable] = None,
        name_lookup: Optional[Callable] = None,
    ):
        """Initialize extractor with optional dependency injection for caching.

        Uses dependency injection pattern to enable high-performance caching
        while maintaining functionality when caching is unavailable.

        Args:
            query_preprocessor (Optional[Callable]): Custom query preprocessing function
            name_lookup (Optional[Callable]): Custom name lookup function
        """
        # Load comprehensive faculty databases
        self.known_faculty = KNOWN_FACULTY
        self.ambiguous_names = AMBIGUOUS_NAMES
        self.first_names = FIRST_NAMES
        self.special_positions = SPECIAL_POSITIONS
        self.false_positives = FALSE_POSITIVES

        # Dependency injection with fallback implementations
        self.query_preprocessor = query_preprocessor or _fallback_query_preprocessing
        self.name_lookup = name_lookup or self._default_name_lookup

    def _default_name_lookup(self, name: str, lookup_type: str) -> Optional[str]:
        """Default fallback for name lookup when no custom function provided.

        Args:
            name (str): Name to lookup
            lookup_type (str): Type of lookup to perform

        Returns:
            Optional[str]: Found faculty name or None
        """
        return _fallback_name_lookup(name, lookup_type, self)

    @lru_cache(maxsize=512)
    def _cached_extract_names(self, query: str) -> tuple:
        """Internal caching layer for name extraction optimization.

        Prevents redundant processing of identical queries within session.
        Returns tuple for lru_cache hashability requirement.

        Args:
            query (str): User query to process

        Returns:
            tuple: Tuple of extracted faculty names for caching
        """
        return tuple(self._extract_names_internal(query))

    def extract_names(self, query: str) -> List[str]:
        """Main entry point for faculty name extraction from user queries.

        Processes natural language queries to identify faculty members using
        comprehensive multi-strategy approach with automatic fallbacks.

        Args:
            query (str): User's natural language query

        Returns:
            List[str]: List of standardized faculty names found in query

        Examples:
            >>> extractor.extract_names("monica email")
            ['Dr. Monica Sundd']
            >>> extractor.extract_names("dr sarika research")
            ['Dr. Sarika Gupta']
        """
        cached_result = self._cached_extract_names(query)
        return list(cached_result)

    def _extract_names_internal(self, query: str) -> List[str]:
        """Core name extraction with enhanced multi-strategy recognition.

        Implements comprehensive extraction strategy:
            1. Query preprocessing and early exits
            2. Special position handling (director)
            3. Exact known faculty matching
            4. Enhanced partial name matching
            5. First name database lookup
            6. Ambiguous name disambiguation

        Args:
            query (str): User query to process

        Returns:
            List[str]: List of extracted faculty names
        """
        names = []
        query_lower = query.lower()

        # Query classification and early exits
        (
            has_pronouns,
            has_exit_words,
            has_faculty_list_patterns,
            has_director_patterns,
        ) = self.query_preprocessor(query_lower)

        # Early termination for non-person queries
        if has_exit_words or has_faculty_list_patterns:
            return []

        # Special case: Director queries
        if has_director_patterns:
            director_name = self.special_positions.get("director")
            if director_name:
                names.append(director_name)
            return names

        # Strategy 1: Exact known faculty matching
        extracted_name = self._try_exact_faculty_matching(query_lower)
        if extracted_name:
            names.append(extracted_name)
            return names

        # Strategy 2: First name database lookup
        extracted_name = self._try_first_name_matching(query_lower)
        if extracted_name:
            names.append(extracted_name)
            return names

        # Strategy 3: Enhanced partial name matching
        extracted_name = self._try_enhanced_partial_matching(query_lower)
        if extracted_name:
            names.append(extracted_name)
            return names

        # Strategy 4: Ambiguous name disambiguation
        disambiguated_names = self._try_ambiguous_name_resolution(query, query_lower)
        if disambiguated_names:
            names.extend(disambiguated_names)
            return names

        return list(set(names))

    def _try_exact_faculty_matching(self, query_lower: str) -> Optional[str]:
        """Attempt exact matching against known faculty database.

        Provides highest precision by matching complete name patterns
        from the KNOWN_FACULTY database.

        Args:
            query_lower (str): Lowercase user query

        Returns:
            Optional[str]: Matched faculty name or None
        """
        for known_name in self.known_faculty.keys():
            if known_name in query_lower:
                result = self.name_lookup(known_name, "known_faculty")
                if result:
                    print(f"Exact faculty match: '{known_name}' → '{result}'")
                    return result
        return None

    def _try_enhanced_partial_matching(self, query_lower: str) -> Optional[str]:
        """Enhanced partial name matching for incomplete faculty queries.

        Handles patterns like "dr monica", "sarika contact" by extracting
        potential name components and matching against faculty databases.

        Args:
            query_lower (str): Lowercase user query

        Returns:
            Optional[str]: Matched faculty name or None
        """
        # Extract potential name parts from query
        query_words = self._extract_query_words(query_lower)

        # Try combinations of consecutive words as potential names
        for i in range(len(query_words)):
            # Max 3-word combinations
            for j in range(i + 1, min(i + 4, len(query_words) + 1)):
                potential_name = " ".join(query_words[i:j])

                # Skip very short or invalid combinations
                if len(potential_name) < 3 or potential_name in self.false_positives:
                    continue

                # Check against known faculty patterns
                matched_faculty = self._match_partial_name(potential_name)
                if matched_faculty:
                    print(f"Partial match: '{potential_name}' → '{matched_faculty}'")
                    return matched_faculty

        return None

    def _try_first_name_matching(self, query_lower: str) -> Optional[str]:
        """Direct first name database lookup for casual queries.

        Handles informal patterns like "monica email", "sarika research"
        using the FIRST_NAMES database for immediate faculty identification.

        Args:
            query_lower (str): Lowercase user query

        Returns:
            Optional[str]: Matched faculty name or None
        """
        query_words = query_lower.split()

        # Check each word and phrase against first names database
        for first_name in self.first_names.keys():
            if (
                first_name in query_words or first_name in query_lower
            ) and self._is_valid_name_context(query_lower, first_name):

                result = self.name_lookup(first_name, "first_names")
                if result:
                    print(f"First name match: '{first_name}' → '{result}'")
                    return result

        return None

    def _try_ambiguous_name_resolution(self, query: str, query_lower: str) -> List[str]:
        """Handle ambiguous names with context-based disambiguation.

        When surnames match multiple faculty members, uses contextual
        clues to identify the most likely candidate.

        Args:
            query (str): Original user query
            query_lower (str): Lowercase user query

        Returns:
            List[str]: List of disambiguated faculty names
        """
        for ambiguous_name in self.ambiguous_names.keys():
            if ambiguous_name in query_lower:
                candidates = self.name_lookup(ambiguous_name, "ambiguous")
                if candidates:
                    disambiguated = self._disambiguate_names(query, candidates)
                    if disambiguated:
                        print(f"Disambiguated: '{ambiguous_name}' → {disambiguated}")
                        return disambiguated
        return []

    def _extract_query_words(self, query_lower: str) -> List[str]:
        """Extract meaningful words from query for name matching.

        Removes titles, stop words, and non-name tokens while preserving
        potential name components.

        Args:
            query_lower (str): Lowercase user query

        Returns:
            List[str]: List of meaningful words for name matching
        """
        # Remove titles and normalize
        cleaned_query = query_lower.replace("dr.", "").replace("prof.", "").strip()

        # Split into words
        words = cleaned_query.split()

        # Filter out stop words and very short words
        stop_words = {
            "the",
            "of",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "is",
            "are",
            "was",
            "were",
        }
        meaningful_words = [
            word for word in words if len(word) > 2 and word not in stop_words
        ]

        return meaningful_words

    def _match_partial_name(self, potential_name: str) -> Optional[str]:
        """Match potential name against faculty databases with flexible matching.

        Checks if the potential name appears in any faculty database keys
        and returns the corresponding full faculty name.

        Args:
            potential_name (str): Potential name extracted from query

        Returns:
            Optional[str]: Matched full faculty name or None
        """
        # Check against known faculty database
        for known_name, full_name in self.known_faculty.items():
            if potential_name in known_name or self._is_name_similarity(
                potential_name, known_name
            ):
                return full_name

        # Check against first names database
        for first_name, full_name in self.first_names.items():
            if potential_name in first_name or first_name in potential_name:
                return full_name

        return None

    def _is_name_similarity(self, potential_name: str, known_name: str) -> bool:
        """Check for name similarity using flexible matching criteria.

        Uses word-level matching to handle variations in name formatting
        and partial matches.

        Args:
            potential_name (str): Potential name to check
            known_name (str): Known faculty name from database

        Returns:
            bool: True if names are similar enough to match
        """
        potential_words = set(potential_name.split())
        known_words = set(known_name.split())

        # Calculate word overlap
        overlap = len(potential_words.intersection(known_words))
        min_words = min(len(potential_words), len(known_words))

        # Require significant overlap for similarity
        return overlap > 0 and overlap >= min_words * 0.6

    def _is_valid_name_context(self, query_lower: str, name: str) -> bool:
        """Validate name appears in appropriate academic context.

        Prevents false positives by checking for academic indicators
        and avoiding known false positive patterns.

        Args:
            query_lower (str): Lowercase user query
            name (str): Name to validate

        Returns:
            bool: True if context is appropriate for faculty search
        """
        # Check for false positive patterns
        for false_positive in self.false_positives:
            if false_positive in query_lower:
                return False

        # Look for academic context indicators
        academic_indicators = [
            "dr",
            "prof",
            "professor",
            "doctor",
            "faculty",
            "email",
            "contact",
            "research",
            "publication",
            "lab",
            "about",
            "tell me",
            "working on",
            "interests",
            "phone",
            "extension",
        ]

        # Check for academic context around the name
        name_position = query_lower.find(name)
        if name_position >= 0:
            # Check context around the name (10 characters before and after)
            context_start = max(0, name_position - 10)
            context_end = min(len(query_lower), name_position + len(name) + 10)
            name_context = query_lower[context_start:context_end]

            if any(indicator in name_context for indicator in academic_indicators):
                return True

        # Check for academic indicators anywhere in query
        return any(indicator in query_lower for indicator in academic_indicators)

    def _disambiguate_names(self, query: str, candidates: List[str]) -> List[str]:
        """Resolve ambiguous names using contextual clues and domain knowledge.

        When multiple faculty members match a surname, analyzes query context
        to identify the most likely candidate based on research areas and specialties.

        Args:
            query (str): Original user query
            candidates (List[str]): List of candidate faculty names

        Returns:
            List[str]: List of best matching faculty names
        """
        query_lower = query.lower()

        # Enhanced disambiguation clues with broader context
        disambiguation_clues = {
            "Dr. Sarika Gupta": [
                "neurodegenerative",
                "protein",
                "misfolding",
                "sarika",
                "chemical biology",
            ],
            "Dr. Nimesh Gupta": [
                "virus",
                "immunology",
                "vaccine",
                "infection",
                "nimesh",
            ],
            "Dr. Anil Kumar": ["microbiome", "cancer", "biology", "anil", "genetics"],
            "Dr. Rajesh Kumar Yadav": ["epigenetics", "cancer", "chromatin", "rajesh"],
            "Dr. Narendra Kumar": ["narendra", "narender", "structural", "protein"],
        }

        best_candidate = None
        best_score = 0

        for candidate in candidates:
            score = 0
            clues = disambiguation_clues.get(candidate, [])

            for clue in clues:
                if clue in query_lower:
                    # Weight longer clues higher
                    if len(clue) > 8:
                        score += 3
                    elif len(clue) > 5:
                        score += 2
                    else:
                        score += 1

            if score > best_score:
                best_score = score
                best_candidate = candidate

        # Return best candidate if confident, otherwise return all candidates
        if best_candidate and best_score > 0:
            return [best_candidate]
        else:
            print(f"Ambiguous match - returning all candidates: {candidates}")
            return candidates


def create_faculty_extractor_with_cache():
    """Factory function for cache-optimized extractor creation.

    Attempts to use cached functions for optimal performance with
    graceful degradation to non-cached implementations if unavailable.

    Returns:
        FacultyNameExtractor: Extractor instance with caching if available
    """
    try:
        from core.caching import _cached_query_preprocessing, _cached_name_lookup

        print("Using CACHED functions for optimal performance")
        return FacultyNameExtractor(
            query_preprocessor=_cached_query_preprocessing,
            name_lookup=_cached_name_lookup,
        )
    except ImportError:
        print("Using FALLBACK functions (no caching available)")
        return FacultyNameExtractor()


def create_faculty_extractor_no_cache():
    """Factory function for explicit non-cached extractor creation.

    Provides guaranteed non-cached behavior for testing, debugging,
    or memory-constrained environments.

    Returns:
        FacultyNameExtractor: Extractor instance without caching
    """
    return FacultyNameExtractor()  # Uses fallback functions
