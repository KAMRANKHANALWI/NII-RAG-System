"""Utilities for content processing and formatting."""

import json
from typing import Any, Union


def flatten_content(content: Any) -> str:
    """
    Convert various content types to flattened string format.

    Args:
        content: Content to flatten (str, dict, or other)

    Returns:
        Flattened string representation
    """
    if isinstance(content, str):
        return content.strip()
    elif isinstance(content, dict):
        lines = []
        for key, value in content.items():
            if isinstance(value, list):
                value = ", ".join(map(str, value))
            elif isinstance(value, dict):
                value = json.dumps(value, indent=2)
            lines.append(f"**{key.replace('_', ' ').title()}**: {value}")
        return "\n".join(lines).strip()
    else:
        return str(content).strip()
