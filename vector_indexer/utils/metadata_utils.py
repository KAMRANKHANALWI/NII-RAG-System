"""Utilities for metadata processing and safety."""

from typing import Any, Dict, Union


def flatten_metadata(
    metadata: Dict[str, Any],
) -> Dict[str, Union[str, int, float, bool]]:
    """
    Convert complex metadata to safe format for vector storage.

    Args:
        metadata: Raw metadata dictionary

    Returns:
        Flattened metadata with safe types only
    """
    safe_metadata = {}
    for k, v in metadata.items():
        if isinstance(v, list):
            safe_metadata[k] = ", ".join(map(str, v))
        elif isinstance(v, (str, int, float, bool)):
            safe_metadata[k] = v
        else:
            safe_metadata[k] = str(v)
    return safe_metadata
