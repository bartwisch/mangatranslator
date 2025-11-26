"""Utilities for managing UI states across the Streamlit app."""

from typing import Any


def should_display_thumbnails(translation_in_progress: bool) -> bool:
    """Determine whether page thumbnails should be rendered.

    Args:
        translation_in_progress (bool): Flag indicating if the translation workflow is active.

    Returns:
        bool: True if thumbnails should be shown, otherwise False.

    Raises:
        ValueError: If the provided flag is not a boolean.
    """
    if not isinstance(translation_in_progress, bool):
        raise ValueError("translation_in_progress must be a boolean")

    return not translation_in_progress
