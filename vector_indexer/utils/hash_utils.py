"""Utilities for content hashing."""

import hashlib


def content_hash(text: str) -> str:
    """
    Generate MD5 hash for given text content.

    Args:
        text: Input text to hash

    Returns:
        MD5 hash as hexadecimal string
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()
