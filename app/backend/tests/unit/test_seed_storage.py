"""Demo storage retries and idempotency with boto3 exceptions."""

from unittest.mock import Mock

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

import seed_demo_configs as seed
import thumbnail_utils as thumbnails


def test_existing_seed_object_is_not_overwritten(monkeypatch):
    monkeypatch.setattr(seed, "object_exists", lambda *_args: True)
    copy = Mock()
    monkeypatch.setattr(seed, "copy_object", copy)
    seed._ensure_object_with_retry("config", "uploads/video.mp4", "data", "video.mp4")
    copy.assert_not_called()


def test_seed_waits_for_source_then_copies(monkeypatch):
    # First attempt: destination and source missing. Second: source is ready.
    monkeypatch.setattr(
        seed, "object_exists", Mock(side_effect=[False, False, False, True])
    )
    copy = Mock()
    sleep = Mock()
    monkeypatch.setattr(seed, "copy_object", copy)
    monkeypatch.setattr(seed.time, "sleep", sleep)
    seed._ensure_object_with_retry(
        "config", "uploads/video.mp4", "data", "video.mp4", max_retries=2
    )
    copy.assert_called_once_with("config", "uploads/video.mp4", "data", "video.mp4")
    sleep.assert_called_once_with(2.0)


@pytest.mark.parametrize(
    "error",
    [
        ClientError({"Error": {"Code": "SlowDown"}}, "CopyObject"),
        EndpointConnectionError(endpoint_url="http://storage.example:7480"),
    ],
)
def test_seed_retries_boto3_copy_failures(monkeypatch, error):
    monkeypatch.setattr(
        seed, "object_exists", Mock(side_effect=[False, True, False, True])
    )
    copy = Mock(side_effect=[error, None])
    monkeypatch.setattr(seed, "copy_object", copy)
    monkeypatch.setattr(seed.time, "sleep", Mock())
    seed._ensure_object_with_retry(
        "config", "uploads/video.mp4", "data", "video.mp4", max_retries=2
    )
    assert copy.call_count == 2


def test_seed_retries_destination_head_errors(monkeypatch):
    error = EndpointConnectionError(endpoint_url="http://storage.example:7480")
    monkeypatch.setattr(seed, "object_exists", Mock(side_effect=[error, False, True]))
    copy = Mock()
    monkeypatch.setattr(seed, "copy_object", copy)
    monkeypatch.setattr(seed.time, "sleep", Mock())
    seed._ensure_object_with_retry(
        "config", "uploads/video.mp4", "data", "video.mp4", max_retries=2
    )
    copy.assert_called_once()


def test_seed_exhausted_retries_raise(monkeypatch):
    monkeypatch.setattr(seed, "object_exists", lambda *_args: False)
    monkeypatch.setattr(seed.time, "sleep", Mock())
    with pytest.raises(RuntimeError, match="after 2 attempts"):
        seed._ensure_object_with_retry(
            "config", "uploads/video.mp4", "data", "video.mp4", max_retries=2
        )


def test_readiness_retries_boto3_errors(monkeypatch):
    client = Mock()
    client.list_buckets.side_effect = [
        EndpointConnectionError(endpoint_url="http://storage.example:7480"),
        {},
    ]
    monkeypatch.setattr(seed, "get_s3_client", lambda: client)
    monkeypatch.setattr(seed.time, "sleep", Mock())
    seed._ping_storage(max_attempts=2)
    assert client.list_buckets.call_count == 2


def test_optional_thumbnail_failure_does_not_abort_config_creation(monkeypatch):
    error = ClientError({"Error": {"Code": "AccessDenied"}}, "HeadObject")
    monkeypatch.setattr(thumbnails, "object_exists", Mock(side_effect=error))
    assert (
        thumbnails.generate_thumbnail_for_video_source("s3://config/uploads/video.mp4")
        is None
    )
