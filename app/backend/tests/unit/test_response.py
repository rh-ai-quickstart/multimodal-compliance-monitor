"""Unit tests for response.py: YOLO postprocessing and detection formatting."""

import numpy as np
import pytest

from response import (
    Detection,
    _raw_prediction_tensor,
    _sigmoid,
    _apply_class_sigmoid,
    _predictions_matrix,
    postprocess_image,
    process_detections,
)

PPE_CLASSES = {
    0: "Hardhat",
    1: "Mask",
    2: "NO-Hardhat",
    3: "NO-Mask",
    4: "NO-Safety Vest",
    5: "Person",
    6: "Safety Cone",
    7: "Safety Vest",
    8: "machinery",
    9: "vehicle",
}

NC = len(PPE_CLASSES)
FEAT = 4 + NC  # 14


# ── _raw_prediction_tensor ──────────────────────────────────────────────────


class TestRawPredictionTensor:
    def test_dict_with_output0(self):
        arr = np.zeros((1, 14, 5))
        result = _raw_prediction_tensor({"output0": arr})
        np.testing.assert_array_equal(result, arr)

    def test_dict_without_output0_uses_first_value(self):
        arr = np.ones((1, 14, 5))
        result = _raw_prediction_tensor({"predictions": arr})
        np.testing.assert_array_equal(result, arr)

    def test_list_input(self):
        arr = np.zeros((1, 14, 5))
        result = _raw_prediction_tensor([arr])
        np.testing.assert_array_equal(result, arr)

    def test_tuple_input(self):
        arr = np.zeros((1, 14, 5))
        result = _raw_prediction_tensor((arr,))
        np.testing.assert_array_equal(result, arr)

    def test_bare_ndarray(self):
        arr = np.zeros((1, 14, 5))
        result = _raw_prediction_tensor(arr)
        np.testing.assert_array_equal(result, arr)

    def test_output_is_always_ndarray(self):
        for inp in [{"output0": [[1, 2]]}, [[1, 2]], np.array([1, 2])]:
            assert isinstance(_raw_prediction_tensor(inp), np.ndarray)


# ── _sigmoid ────────────────────────────────────────────────────────────────


class TestSigmoid:
    def test_zero_gives_half(self):
        result = _sigmoid(np.array([0.0]))
        assert abs(result[0] - 0.5) < 1e-10

    def test_large_negative_clipped(self):
        result = _sigmoid(np.array([-500.0]))
        assert result[0] < 1e-100

    def test_large_positive_clipped(self):
        result = _sigmoid(np.array([500.0]))
        assert result[0] > 1.0 - 1e-10

    @pytest.mark.parametrize("x", [-5.0, -1.0, 0.0, 1.0, 5.0])
    def test_matches_formula(self, x):
        expected = 1.0 / (1.0 + np.exp(-x))
        result = _sigmoid(np.array([x]))
        assert abs(result[0] - expected) < 1e-10

    def test_output_always_in_0_1_moderate_values(self):
        inputs = np.array([-30, -10, -1, 0, 1, 10, 30], dtype=float)
        result = _sigmoid(inputs)
        assert np.all(result > 0)
        assert np.all(result < 1)

    def test_extreme_values_saturate(self):
        result = _sigmoid(np.array([-1000.0, 1000.0]))
        assert result[0] == 0.0 or result[0] < 1e-200
        assert result[1] == 1.0 or result[1] > 1.0 - 1e-10


# ── _apply_class_sigmoid ────────────────────────────────────────────────────


class TestApplyClassSigmoid:
    @pytest.mark.parametrize("env_val", ["true", "1", "yes"])
    def test_force_on(self, monkeypatch, env_val):
        monkeypatch.setenv("YOLO_CLASS_SIGMOID", env_val)
        scores = np.array([[0.5, 0.8]])
        result, applied = _apply_class_sigmoid(scores)
        assert applied is True
        expected = 1.0 / (1.0 + np.exp(-scores.astype(np.float64)))
        np.testing.assert_allclose(result, expected, atol=1e-10)

    @pytest.mark.parametrize("env_val", ["false", "0", "no"])
    def test_force_off(self, monkeypatch, env_val):
        monkeypatch.setenv("YOLO_CLASS_SIGMOID", env_val)
        scores = np.array([[2.5, -1.3]])
        result, applied = _apply_class_sigmoid(scores)
        assert applied is False
        np.testing.assert_array_equal(result, scores)

    def test_auto_applies_for_logits(self, monkeypatch):
        monkeypatch.delenv("YOLO_CLASS_SIGMOID", raising=False)
        scores = np.array([[-3.0, 0.0, 3.0]])
        result, applied = _apply_class_sigmoid(scores)
        assert applied is True
        assert np.all(result > 0) and np.all(result < 1)

    def test_auto_skips_for_probabilities(self, monkeypatch):
        monkeypatch.delenv("YOLO_CLASS_SIGMOID", raising=False)
        scores = np.array([[0.1, 0.5, 0.9]])
        result, applied = _apply_class_sigmoid(scores)
        assert applied is False
        np.testing.assert_array_equal(result, scores)

    def test_empty_array_returns_false(self, monkeypatch):
        monkeypatch.delenv("YOLO_CLASS_SIGMOID", raising=False)
        scores = np.array([]).reshape(0, NC)
        result, applied = _apply_class_sigmoid(scores)
        assert applied is False
        assert result.size == 0

    def test_auto_boundary_just_above_one(self, monkeypatch):
        monkeypatch.delenv("YOLO_CLASS_SIGMOID", raising=False)
        scores = np.array([[0.5, 1.0001]])
        _, applied = _apply_class_sigmoid(scores)
        assert applied is True

    def test_auto_boundary_exactly_one(self, monkeypatch):
        monkeypatch.delenv("YOLO_CLASS_SIGMOID", raising=False)
        scores = np.array([[0.5, 1.0]])
        _, applied = _apply_class_sigmoid(scores)
        assert applied is False


# ── _predictions_matrix ─────────────────────────────────────────────────────


class TestPredictionsMatrix:
    @pytest.mark.parametrize(
        "shape",
        [(1, FEAT, 8400), (1, 8400, FEAT), (8400, FEAT)],
    )
    def test_normalizes_common_shapes(self, shape):
        raw = np.random.rand(*shape).astype(np.float32)
        result = _predictions_matrix(raw, NC)
        assert result.shape == (8400, FEAT)

    def test_single_prediction_feat_by_1(self):
        raw = np.random.rand(FEAT, 1).astype(np.float32)
        result = _predictions_matrix(raw, NC)
        assert result.shape == (1, FEAT)

    def test_single_prediction_batch_wrapped(self):
        raw = np.random.rand(1, FEAT, 1).astype(np.float32)
        result = _predictions_matrix(raw, NC)
        assert result.shape == (1, FEAT)

    def test_wrong_feat_dim_raises(self):
        raw = np.random.rand(5, 5).astype(np.float32)
        with pytest.raises(ValueError, match="feature dimension mismatch"):
            _predictions_matrix(raw, NC)

    def test_mismatched_nc_raises(self):
        raw = np.random.rand(1, 15, 8400).astype(np.float32)
        with pytest.raises(ValueError, match="feature dimension mismatch"):
            _predictions_matrix(raw, NC)

    def test_output_always_has_feat_columns(self):
        for shape in [(1, FEAT, 50), (50, FEAT), (FEAT, 50)]:
            result = _predictions_matrix(np.random.rand(*shape).astype(np.float32), NC)
            assert result.shape[1] == FEAT

    def test_output_dtype_float32(self):
        raw = np.random.rand(1, FEAT, 10).astype(np.float64)
        result = _predictions_matrix(raw, NC)
        assert result.dtype == np.float32


# ── postprocess_image ───────────────────────────────────────────────────────


def _make_yolo_tensor(boxes_and_classes: list[tuple], nc: int = NC) -> np.ndarray:
    """Build a (1, 4+nc, N) YOLO output tensor with planted detections.

    Each entry: (cx, cy, w, h, class_id, score).
    """
    n = len(boxes_and_classes)
    feat = 4 + nc
    tensor = np.zeros((feat, n), dtype=np.float32)
    for i, (cx, cy, w, h, class_id, score) in enumerate(boxes_and_classes):
        tensor[0, i] = cx
        tensor[1, i] = cy
        tensor[2, i] = w
        tensor[3, i] = h
        tensor[4 + class_id, i] = score
    return tensor[np.newaxis, :, :]


class TestPostprocessImage:
    def test_happy_path_three_detections(self, monkeypatch):
        monkeypatch.setenv("YOLO_CLASS_SIGMOID", "false")
        tensor = _make_yolo_tensor(
            [
                (320, 320, 100, 300, 5, 0.95),  # Person
                (330, 200, 80, 50, 0, 0.90),  # Hardhat
                (550, 550, 40, 40, 6, 0.85),  # Safety Cone
            ]
        )
        dets = postprocess_image(tensor, scale=1.0, classes=PPE_CLASSES)
        class_names = {d.class_name for d in dets}
        assert "Person" in class_names
        assert "Hardhat" in class_names
        assert "Safety Cone" in class_names

    def test_boundary_score_at_threshold_passes(self, monkeypatch):
        monkeypatch.setenv("YOLO_CLASS_SIGMOID", "false")
        tensor = _make_yolo_tensor([(320, 320, 100, 100, 5, 0.25)])
        dets = postprocess_image(tensor, scale=1.0, classes=PPE_CLASSES)
        assert len(dets) == 1

    def test_boundary_score_below_threshold_filtered(self, monkeypatch):
        monkeypatch.setenv("YOLO_CLASS_SIGMOID", "false")
        tensor = _make_yolo_tensor([(320, 320, 100, 100, 5, 0.24)])
        dets = postprocess_image(tensor, scale=1.0, classes=PPE_CLASSES)
        assert len(dets) == 0

    def test_all_below_threshold_empty(self, monkeypatch):
        monkeypatch.setenv("YOLO_CLASS_SIGMOID", "false")
        tensor = _make_yolo_tensor(
            [
                (100, 100, 50, 50, 0, 0.1),
                (200, 200, 50, 50, 1, 0.15),
            ]
        )
        dets = postprocess_image(tensor, scale=1.0, classes=PPE_CLASSES)
        assert dets == []

    def test_single_detection(self, monkeypatch):
        monkeypatch.setenv("YOLO_CLASS_SIGMOID", "false")
        tensor = _make_yolo_tensor([(300, 300, 80, 200, 5, 0.92)])
        dets = postprocess_image(tensor, scale=1.5, classes=PPE_CLASSES)
        assert len(dets) == 1
        assert dets[0].class_name == "Person"
        assert dets[0].scale == 1.5

    def test_nms_suppresses_overlapping_boxes(self, monkeypatch):
        monkeypatch.setenv("YOLO_CLASS_SIGMOID", "false")
        tensor = _make_yolo_tensor(
            [
                (320, 320, 100, 300, 5, 0.95),  # Person high conf
                (325, 325, 100, 300, 5, 0.80),  # Person lower conf, ~same location
            ]
        )
        dets = postprocess_image(tensor, scale=1.0, classes=PPE_CLASSES)
        assert len(dets) == 1
        assert dets[0].confidence == pytest.approx(0.95)

    def test_confidence_above_nms_threshold(self, monkeypatch):
        monkeypatch.setenv("YOLO_CLASS_SIGMOID", "false")
        tensor = _make_yolo_tensor(
            [(100, 100, 50, 50, i, 0.5 + i * 0.05) for i in range(NC)]
        )
        dets = postprocess_image(tensor, scale=1.0, classes=PPE_CLASSES)
        for d in dets:
            assert d.confidence >= 0.20

    def test_class_name_always_valid(self, monkeypatch):
        monkeypatch.setenv("YOLO_CLASS_SIGMOID", "false")
        tensor = _make_yolo_tensor(
            [(100 + i * 60, 100, 40, 40, i, 0.8) for i in range(NC)]
        )
        dets = postprocess_image(tensor, scale=1.0, classes=PPE_CLASSES)
        for d in dets:
            assert d.class_name in PPE_CLASSES.values()


# ── process_detections ──────────────────────────────────────────────────────


class TestProcessDetections:
    def test_happy_path_scaling(self):
        dets = [
            Detection(
                class_id=5,
                class_name="Person",
                confidence=0.9,
                bbox=[100, 50, 200, 350],
                scale=2.0,
            )
        ]
        result, counts = process_detections(dets)
        assert len(result) == 1
        assert result[0]["bbox"] == (200, 100, 600, 800)
        assert counts["Person"] == 1

    def test_empty_list(self):
        result, counts = process_detections([])
        assert result == []
        assert dict(counts) == {}

    def test_include_in_counts_filtering(self):
        dets = [
            Detection(
                class_id=5,
                class_name="Person",
                confidence=0.9,
                bbox=[100, 50, 200, 350],
                scale=1.0,
            ),
            Detection(
                class_id=6,
                class_name="Safety Cone",
                confidence=0.7,
                bbox=[400, 400, 40, 40],
                scale=1.0,
            ),
        ]
        result, counts = process_detections(
            dets, include_in_counts_by_class_id={5: True, 6: False}
        )
        assert len(result) == 1
        assert result[0]["class_name"] == "Person"
        assert "Safety Cone" not in counts

    def test_scale_one_unchanged(self):
        dets = [
            Detection(
                class_id=0,
                class_name="Hardhat",
                confidence=0.85,
                bbox=[10, 20, 30, 40],
                scale=1.0,
            )
        ]
        result, _ = process_detections(dets)
        assert result[0]["bbox"] == (10, 20, 40, 60)

    def test_include_none_includes_all(self):
        dets = [
            Detection(
                class_id=5,
                class_name="Person",
                confidence=0.9,
                bbox=[0, 0, 10, 10],
                scale=1.0,
            ),
            Detection(
                class_id=6,
                class_name="Safety Cone",
                confidence=0.7,
                bbox=[50, 50, 10, 10],
                scale=1.0,
            ),
        ]
        result, counts = process_detections(dets, include_in_counts_by_class_id=None)
        assert len(result) == 2
        assert counts["Person"] == 1
        assert counts["Safety Cone"] == 1

    def test_count_equals_detection_length(self):
        dets = [
            Detection(
                class_id=5,
                class_name="Person",
                confidence=0.9,
                bbox=[0, 0, 10, 10],
                scale=1.0,
            ),
            Detection(
                class_id=5,
                class_name="Person",
                confidence=0.8,
                bbox=[50, 50, 10, 10],
                scale=1.0,
            ),
            Detection(
                class_id=0,
                class_name="Hardhat",
                confidence=0.7,
                bbox=[100, 100, 10, 10],
                scale=1.0,
            ),
        ]
        result, counts = process_detections(dets)
        assert len(result) == sum(counts.values())

    def test_bbox_coordinates_are_integers(self):
        dets = [
            Detection(
                class_id=5,
                class_name="Person",
                confidence=0.9,
                bbox=[10.7, 20.3, 30.5, 40.1],
                scale=1.5,
            )
        ]
        result, _ = process_detections(dets)
        for coord in result[0]["bbox"]:
            assert isinstance(coord, int)
