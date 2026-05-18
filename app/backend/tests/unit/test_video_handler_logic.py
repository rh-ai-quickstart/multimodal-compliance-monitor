"""Unit tests for VideoHandler business logic: description formatting and compliance summary."""

from collections import deque

import pytest

from video_processing.video_handler import VideoHandler

PPE_CLASS_ORDER = [
    "Hardhat",
    "Mask",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
    "Person",
    "Safety Cone",
    "Safety Vest",
    "machinery",
    "vehicle",
]

DESC_FULL_COMPLIANCE = "Detected: Hardhat: 2, Safety Vest: 2, Mask: 2, Person: 2"
DESC_VIOLATION = "Detected: NO-Hardhat: 1, NO-Safety Vest: 1, NO-Mask: 1, Person: 2, Hardhat: 1, Safety Vest: 1, Mask: 1"
DESC_NO_PEOPLE = "Detected: Safety Cone: 3, machinery: 1"


@pytest.fixture
def handler():
    """Create a VideoHandler without running __init__ (no DB, no threads)."""
    obj = object.__new__(VideoHandler)
    return obj


# ── _format_detection_description ───────────────────────────────────────────


class TestFormatDetectionDescription:
    def test_happy_path(self, handler):
        counts = {"Person": 2, "Hardhat": 2, "Safety Vest": 2}
        result = handler._format_detection_description(counts, PPE_CLASS_ORDER)
        assert "Hardhat: 2" in result
        assert "Safety Vest: 2" in result
        assert "Person: 2" in result

    def test_all_zero_counts(self, handler):
        counts = {"Person": 0, "Hardhat": 0, "Safety Vest": 0}
        result = handler._format_detection_description(counts, PPE_CLASS_ORDER)
        assert result == "Detected:"

    def test_single_class(self, handler):
        counts = {"Person": 1}
        result = handler._format_detection_description(counts, PPE_CLASS_ORDER)
        assert result == "Detected: Person: 1"

    def test_zero_count_skipped(self, handler):
        counts = {"Person": 2, "Safety Cone": 0, "Hardhat": 1}
        result = handler._format_detection_description(counts, PPE_CLASS_ORDER)
        assert "Safety Cone" not in result
        assert "Hardhat: 1" in result

    def test_always_starts_with_detected(self, handler):
        for counts in [{}, {"Person": 3}, {"Hardhat": 0}]:
            result = handler._format_detection_description(counts, PPE_CLASS_ORDER)
            assert result.startswith("Detected:")

    def test_never_ends_with_comma(self, handler):
        counts = {"Person": 1, "Hardhat": 2, "Safety Vest": 3}
        result = handler._format_detection_description(counts, PPE_CLASS_ORDER)
        assert not result.endswith(", ")
        assert not result.endswith(",")

    def test_respects_class_ordering(self, handler):
        counts = {"Person": 1, "Hardhat": 1, "Safety Vest": 1}
        result = handler._format_detection_description(counts, PPE_CLASS_ORDER)
        hardhat_pos = result.index("Hardhat")
        person_pos = result.index("Person")
        vest_pos = result.index("Safety Vest")
        assert hardhat_pos < person_pos < vest_pos


# ── _generate_summary ───────────────────────────────────────────────────────


class TestGenerateSummary:
    def test_full_compliance_good(self, handler):
        descriptions = deque([DESC_FULL_COMPLIANCE] * 50)
        result = handler._generate_summary(descriptions)
        assert "Good compliance observed" in result
        assert "100.00%" in result

    def test_violation_critical(self, handler):
        descriptions = deque([DESC_VIOLATION] * 50)
        result = handler._generate_summary(descriptions)
        assert "Critical" in result

    def test_no_people_detected(self, handler):
        descriptions = deque([DESC_NO_PEOPLE] * 50)
        result = handler._generate_summary(descriptions)
        assert "No people detected" in result

    def test_empty_deque_no_crash(self, handler):
        descriptions = deque()
        result = handler._generate_summary(descriptions)
        assert "Safety Trends Summary" in result
        assert "0 frames" in result

    def test_single_frame(self, handler):
        descriptions = deque([DESC_FULL_COMPLIANCE])
        result = handler._generate_summary(descriptions)
        assert "1 frames" in result
        assert "100.00%" in result

    def test_warning_tier(self, handler):
        # 9 compliant + 1 violation per 10 -> ~90% compliance per category
        compliant = "Detected: Hardhat: 9, Safety Vest: 9, Mask: 9, Person: 9"
        violation = "Detected: NO-Hardhat: 1, NO-Safety Vest: 1, NO-Mask: 1, Person: 1"
        descriptions = deque([compliant] * 9 + [violation] * 1)
        result = handler._generate_summary(descriptions)
        assert "Warning" in result or "Good compliance" in result

    def test_below_80_triggers_critical(self, handler):
        # Heavy violations -> overall < 80% -> Critical message
        compliant = "Detected: Hardhat, Safety Vest, Mask, Person"
        violation = "Detected: NO-Hardhat, NO-Safety Vest, NO-Mask, Person"
        descriptions = deque([compliant] * 1 + [violation] * 4)
        result = handler._generate_summary(descriptions)
        assert "Critical" in result

    def test_above_95_triggers_good(self, handler):
        # 19 compliant + 1 violation -> overall > 95% -> Good message
        compliant = "Detected: Hardhat, Safety Vest, Mask, Person"
        violation = "Detected: NO-Hardhat, NO-Safety Vest, NO-Mask, Person"
        descriptions = deque([compliant] * 19 + [violation] * 1)
        result = handler._generate_summary(descriptions)
        assert "Good compliance observed" in result

    def test_always_contains_header(self, handler):
        for descs in [
            deque(),
            deque([DESC_FULL_COMPLIANCE]),
            deque([DESC_NO_PEOPLE] * 10),
        ]:
            result = handler._generate_summary(descs)
            assert "Safety Trends Summary" in result
