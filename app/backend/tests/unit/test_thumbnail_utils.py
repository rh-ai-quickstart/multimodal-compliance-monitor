"""Unit tests for thumbnail_utils: is_s3_video_path, parse_s3_video_path."""

import pytest

from thumbnail_utils import is_s3_video_path, parse_s3_video_path


# ── is_s3_video_path ────────────────────────────────────────────────────────


class TestIsS3VideoPath:
    def test_ppe_video_source(self):
        assert (
            is_s3_video_path("s3://config/uploads/combined-video-no-gap-rooftop.mp4")
            is True
        )

    def test_bird_video_source(self):
        assert is_s3_video_path("s3://config/uploads/bluejayclear.mp4") is True

    def test_traffic_video_source(self):
        assert is_s3_video_path("s3://config/uploads/cars.mp4") is True

    def test_rtsp_returns_false(self):
        assert is_s3_video_path("rtsp://video-stream:8554/live") is False

    def test_local_path_returns_false(self):
        assert is_s3_video_path("/tmp/local-video.mp4") is False

    def test_empty_string_returns_false(self):
        assert is_s3_video_path("") is False

    def test_none_returns_false(self):
        assert is_s3_video_path(None) is False

    def test_scheme_only_returns_true(self):
        assert is_s3_video_path("s3://") is True

    @pytest.mark.parametrize(
        "path",
        [
            "http://example.com/video.mp4",
            "https://bucket.s3.amazonaws.com/key",
            "file:///tmp/video.mp4",
        ],
    )
    def test_other_schemes_return_false(self, path):
        assert is_s3_video_path(path) is False

    def test_uppercase_s3_returns_false(self):
        # Implementation uses startswith("s3://") which is case-sensitive
        assert is_s3_video_path("S3://config/uploads/video.mp4") is False

    def test_leading_whitespace_handled(self):
        assert is_s3_video_path("  s3://config/uploads/video.mp4") is True


# ── parse_s3_video_path ─────────────────────────────────────────────────────


class TestParseS3VideoPath:
    def test_ppe_video_source(self):
        result = parse_s3_video_path(
            "s3://config/uploads/combined-video-no-gap-rooftop.mp4"
        )
        assert result == ("config", "uploads/combined-video-no-gap-rooftop.mp4")

    def test_bird_video_source(self):
        result = parse_s3_video_path("s3://config/uploads/bluejayclear.mp4")
        assert result == ("config", "uploads/bluejayclear.mp4")

    def test_data_bucket(self):
        result = parse_s3_video_path("s3://data/cars.mp4")
        assert result == ("data", "cars.mp4")

    def test_nested_path(self):
        result = parse_s3_video_path("s3://bucket/a/b/c/d.mp4")
        assert result == ("bucket", "a/b/c/d.mp4")

    def test_bucket_only_no_key_returns_none(self):
        # "s3://bucket" has no slash after bucket -> split gives only 1 part
        result = parse_s3_video_path("s3://bucket")
        assert result is None

    def test_bucket_with_trailing_slash(self):
        result = parse_s3_video_path("s3://bucket/")
        assert result == ("bucket", "")

    def test_special_chars_in_key(self):
        result = parse_s3_video_path("s3://config/uploads/file with spaces.mp4")
        assert result == ("config", "uploads/file with spaces.mp4")

    def test_none_returns_none(self):
        assert parse_s3_video_path(None) is None

    def test_empty_string_returns_none(self):
        assert parse_s3_video_path("") is None

    def test_non_s3_path_returns_none(self):
        assert parse_s3_video_path("rtsp://video-stream:8554/live") is None

    def test_leading_whitespace_handled(self):
        result = parse_s3_video_path("  s3://config/uploads/video.mp4")
        assert result == ("config", "uploads/video.mp4")
