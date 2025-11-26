import pytest

from src.ui_state import should_display_thumbnails


def test_should_display_thumbnails_returns_true_when_not_in_progress():
    """Thumbnails render while translation is idle."""
    assert should_display_thumbnails(False) is True


def test_should_display_thumbnails_returns_false_during_progress():
    """Thumbnails stay hidden once translation starts."""
    assert should_display_thumbnails(True) is False


def test_should_display_thumbnails_rejects_non_boolean_input():
    """Helper enforces boolean input to catch state bugs early."""
    with pytest.raises(ValueError):
        should_display_thumbnails("yes")
