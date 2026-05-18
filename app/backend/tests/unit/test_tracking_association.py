"""Unit tests for PPE association and detection conversion in tracking.py."""

import numpy as np
import supervision as sv

from video_processing.tracking import (
    _batch_associate_ppe,
    _dicts_to_sv_detections,
    _sv_to_boosttrack_dets,
)

# Realistic bounding boxes (xyxy) from a 640x640 inference frame
PERSON_1_XYXY = [100, 50, 280, 450]
PERSON_2_XYXY = [350, 60, 520, 440]
HARDHAT_XYXY = [120, 30, 260, 100]
NO_HARDHAT_XYXY = [370, 40, 500, 110]
SAFETY_VEST_XYXY = [110, 150, 270, 350]
NO_SAFETY_VEST_XYXY = [360, 150, 510, 350]
CONE_XYXY = [550, 400, 600, 470]


# ── _batch_associate_ppe ────────────────────────────────────────────────────


class TestBatchAssociatePpe:
    def test_empty_persons(self):
        persons = np.empty((0, 4), dtype=np.float32)
        all_xyxy = np.array([HARDHAT_XYXY], dtype=np.float32)
        result = _batch_associate_ppe(persons, all_xyxy, ["Hardhat"])
        assert result == []

    def test_no_ppe_classes_all_none(self):
        persons = np.array([PERSON_1_XYXY, PERSON_2_XYXY], dtype=np.float32)
        all_xyxy = np.array([CONE_XYXY, [400, 300, 450, 350]], dtype=np.float32)
        result = _batch_associate_ppe(persons, all_xyxy, ["Safety Cone", "vehicle"])
        assert len(result) == 2
        for r in result:
            assert r == {"hardhat": None, "vest": None, "mask": None}

    def test_happy_path_two_persons(self):
        persons = np.array([PERSON_1_XYXY, PERSON_2_XYXY], dtype=np.float32)
        all_xyxy = np.array(
            [HARDHAT_XYXY, NO_HARDHAT_XYXY, SAFETY_VEST_XYXY, NO_SAFETY_VEST_XYXY],
            dtype=np.float32,
        )
        classes = ["Hardhat", "NO-Hardhat", "Safety Vest", "NO-Safety Vest"]
        result = _batch_associate_ppe(persons, all_xyxy, classes)
        assert result[0] == {"hardhat": True, "vest": True, "mask": None}
        assert result[1] == {"hardhat": False, "vest": False, "mask": None}

    def test_no_overlap_all_remain_none(self):
        persons = np.array([PERSON_1_XYXY, PERSON_2_XYXY], dtype=np.float32)
        far_hardhat = np.array([[550, 400, 600, 420]], dtype=np.float32)
        result = _batch_associate_ppe(persons, far_hardhat, ["Hardhat"])
        for r in result:
            assert r["hardhat"] is None

    def test_touching_edge_counts_as_overlap(self):
        persons = np.array([PERSON_1_XYXY], dtype=np.float32)  # x2=280
        touching_box = np.array([[280, 30, 350, 100]], dtype=np.float32)
        result = _batch_associate_ppe(persons, touching_box, ["Hardhat"])
        assert result[0]["hardhat"] is True

    def test_just_miss_no_overlap(self):
        persons = np.array([PERSON_1_XYXY], dtype=np.float32)  # x2=280
        miss_box = np.array([[281, 30, 350, 100]], dtype=np.float32)
        result = _batch_associate_ppe(persons, miss_box, ["Hardhat"])
        assert result[0]["hardhat"] is None

    def test_first_match_wins_hardhat_first(self):
        persons = np.array([PERSON_1_XYXY], dtype=np.float32)
        both_overlap = np.array([HARDHAT_XYXY, HARDHAT_XYXY], dtype=np.float32)
        result = _batch_associate_ppe(persons, both_overlap, ["Hardhat", "NO-Hardhat"])
        assert result[0]["hardhat"] is True

    def test_first_match_wins_no_hardhat_first(self):
        persons = np.array([PERSON_1_XYXY], dtype=np.float32)
        both_overlap = np.array([HARDHAT_XYXY, HARDHAT_XYXY], dtype=np.float32)
        result = _batch_associate_ppe(persons, both_overlap, ["NO-Hardhat", "Hardhat"])
        assert result[0]["hardhat"] is False

    def test_single_person_all_ppe_populated(self):
        persons = np.array([PERSON_1_XYXY], dtype=np.float32)
        box = HARDHAT_XYXY
        all_xyxy = np.array([box, box, box, box, box, box], dtype=np.float32)
        classes = [
            "Hardhat",
            "NO-Hardhat",
            "Safety Vest",
            "NO-Safety Vest",
            "Mask",
            "NO-Mask",
        ]
        result = _batch_associate_ppe(persons, all_xyxy, classes)
        assert result[0]["hardhat"] is True
        assert result[0]["vest"] is True
        assert result[0]["mask"] is True

    def test_output_length_equals_persons(self):
        n = 5
        persons = np.array([PERSON_1_XYXY] * n, dtype=np.float32)
        all_xyxy = np.array([HARDHAT_XYXY], dtype=np.float32)
        result = _batch_associate_ppe(persons, all_xyxy, ["Hardhat"])
        assert len(result) == n

    def test_output_dict_keys_invariant(self):
        persons = np.array([PERSON_1_XYXY, PERSON_2_XYXY], dtype=np.float32)
        all_xyxy = np.array([HARDHAT_XYXY, SAFETY_VEST_XYXY], dtype=np.float32)
        result = _batch_associate_ppe(persons, all_xyxy, ["Hardhat", "Safety Vest"])
        for r in result:
            assert set(r.keys()) == {"hardhat", "vest", "mask"}
            for v in r.values():
                assert v in (True, False, None)


# ── _dicts_to_sv_detections ─────────────────────────────────────────────────


class TestDictsToSvDetections:
    def test_empty_list(self):
        result = _dicts_to_sv_detections([])
        assert len(result) == 0

    def test_three_detections(self):
        dets = [
            {
                "bbox": (100, 50, 280, 450),
                "confidence": 0.92,
                "class_id": 5,
                "class_name": "Person",
            },
            {
                "bbox": (120, 30, 260, 100),
                "confidence": 0.88,
                "class_id": 0,
                "class_name": "Hardhat",
            },
            {
                "bbox": (550, 400, 600, 470),
                "confidence": 0.75,
                "class_id": 6,
                "class_name": "Safety Cone",
            },
        ]
        result = _dicts_to_sv_detections(dets)
        assert result.xyxy.shape == (3, 4)
        np.testing.assert_array_equal(result.class_id, [5, 0, 6])
        np.testing.assert_allclose(result.confidence, [0.92, 0.88, 0.75])

    def test_confidence_dtype_float32(self):
        dets = [
            {
                "bbox": (10, 20, 30, 40),
                "confidence": 0.5,
                "class_id": 0,
                "class_name": "Hardhat",
            }
        ]
        result = _dicts_to_sv_detections(dets)
        assert result.confidence.dtype == np.float32

    def test_single_detection(self):
        dets = [
            {
                "bbox": (100, 50, 280, 450),
                "confidence": 0.9,
                "class_id": 5,
                "class_name": "Person",
            }
        ]
        result = _dicts_to_sv_detections(dets)
        assert result.xyxy.shape == (1, 4)


# ── _sv_to_boosttrack_dets ──────────────────────────────────────────────────


class TestSvToBoosttrackDets:
    def test_empty_detections(self):
        result = _sv_to_boosttrack_dets(sv.Detections.empty())
        assert result.shape == (0, 6)

    def test_two_detections(self):
        sv_dets = sv.Detections(
            xyxy=np.array([[100, 50, 280, 450], [350, 60, 520, 440]], dtype=np.float32),
            confidence=np.array([0.92, 0.85], dtype=np.float32),
            class_id=np.array([5, 0], dtype=int),
        )
        result = _sv_to_boosttrack_dets(sv_dets)
        assert result.shape == (2, 6)
        np.testing.assert_allclose(result[:, :4], sv_dets.xyxy)
        np.testing.assert_allclose(result[:, 4], sv_dets.confidence)
        np.testing.assert_array_equal(result[:, 5], sv_dets.class_id)

    def test_dtype_always_float32(self):
        sv_dets = sv.Detections(
            xyxy=np.array([[1, 2, 3, 4]], dtype=np.float64),
            confidence=np.array([0.5], dtype=np.float64),
            class_id=np.array([0], dtype=int),
        )
        result = _sv_to_boosttrack_dets(sv_dets)
        assert result.dtype == np.float32

    def test_always_six_columns(self):
        for n in [1, 3, 10]:
            sv_dets = sv.Detections(
                xyxy=np.random.rand(n, 4).astype(np.float32),
                confidence=np.random.rand(n).astype(np.float32),
                class_id=np.zeros(n, dtype=int),
            )
            result = _sv_to_boosttrack_dets(sv_dets)
            assert result.shape[1] == 6
