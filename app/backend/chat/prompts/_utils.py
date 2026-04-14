from __future__ import annotations


def pick_example_classes(
    classes_info: list[dict],
) -> tuple[dict | None, dict | None]:
    """Pick one trackable and one non-trackable class for example generation."""
    trackable = next((c for c in classes_info if c["trackable"]), None)
    non_trackable = next((c for c in classes_info if not c["trackable"]), None)
    return trackable, non_trackable
