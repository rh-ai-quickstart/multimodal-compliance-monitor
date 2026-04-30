from __future__ import annotations


def pick_example_classes(
    classes_info: list[dict],
    trackable_count: int = 1,
    non_trackable_count: int = 1,
) -> tuple[list[dict], list[dict]]:
    """Pick up to *trackable_count* trackable and *non_trackable_count*
    non-trackable classes for example generation.

    Returns fewer items when the source list doesn't have enough.
    """
    trackable = [c for c in classes_info if c["trackable"]][:trackable_count]
    non_trackable = [c for c in classes_info if not c["trackable"]][
        :non_trackable_count
    ]
    return trackable, non_trackable


def pick_example_class(
    classes_info: list[dict],
) -> tuple[dict | None, dict | None]:
    """Convenience wrapper returning a single trackable and non-trackable class."""
    trackable, non_trackable = pick_example_classes(classes_info, 1, 1)
    return trackable[0] if trackable else None, non_trackable[
        0
    ] if non_trackable else None


def english_join(items: list[str], conjunction: str = "and") -> str:
    """Join items with commas and a final conjunction.

    >>> english_join(["a helmet"])
    'a helmet'
    >>> english_join(["a helmet", "a vest"])
    'a helmet and a vest'
    >>> english_join(["a helmet", "a vest", "a boots"])
    'a helmet, a vest, and a boots'
    """
    if len(items) <= 1:
        return items[0] if items else ""
    if len(items) == 2:
        return f"{items[0]} {conjunction} {items[1]}"
    return f"{', '.join(items[:-1])}, {conjunction} {items[-1]}"


def compact_classes(classes_info: list[dict] | str | None) -> str | None:
    """Render classes as ``- name (trackable)`` lines to save tokens."""
    if not classes_info:
        return None
    if isinstance(classes_info, str):
        return classes_info
    return "\n".join(
        f"- {c['name']} ({'trackable' if c.get('trackable') else 'non-trackable'})"
        for c in classes_info
    )
