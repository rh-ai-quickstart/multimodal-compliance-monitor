from __future__ import annotations

_CLASS_TO_ATTR: dict[str, str] = {
    "Hardhat": "hardhat",
    "NO-Hardhat": "hardhat",
    "Safety Vest": "vest",
    "NO-Safety Vest": "vest",
    "Mask": "mask",
    "NO-Mask": "mask",
}


def attr_keys_for_classes(classes_info: list[dict]) -> list[str]:
    """Return unique JSONB attribute keys for the non-trackable classes."""
    seen: set[str] = set()
    keys: list[str] = []
    for c in classes_info:
        if c["trackable"]:
            continue
        attr = _CLASS_TO_ATTR.get(c["name"], c["name"].lower())
        if attr not in seen:
            seen.add(attr)
            keys.append(attr)
    return keys


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


def extract_metric_attributes(
    metric: str,
    classes_info: list[dict],
) -> tuple[list[str], list[str]]:
    """Return trackable class names and non-trackable JSONB attribute keys
    mentioned in *metric*.

    Trackable names are returned as-is (used in ``dc.name = '...'``).
    Non-trackable names are mapped to their JSONB attribute keys via
    ``_CLASS_TO_ATTR`` and deduplicated so that e.g. "Hardhat" and
    "NO-Hardhat" both resolve to a single ``"hardhat"`` entry.

    Returns ``(trackable_names, non_trackable_attr_keys)``.
    """
    metric_lower = metric.lower()
    trackable = [
        c["name"]
        for c in classes_info
        if c["trackable"] and c["name"].lower() in metric_lower
    ]
    seen: set[str] = set()
    non_trackable: list[str] = []
    for c in classes_info:
        if c["trackable"]:
            continue
        attr = _CLASS_TO_ATTR.get(c["name"], c["name"].lower())
        if attr in metric_lower and attr not in seen:
            seen.add(attr)
            non_trackable.append(attr)

    return trackable, non_trackable
