"""S3 client contracts without a live object store or application services."""

import io
from unittest.mock import Mock

import boto3
import pytest
from boto3.exceptions import RetriesExceededError
from botocore.exceptions import ClientError, EndpointConnectionError
from botocore.response import StreamingBody
from botocore.stub import Stubber

import s3_client as storage


@pytest.fixture
def stubbed_client(monkeypatch):
    client = boto3.client(
        "s3",
        endpoint_url="http://storage.example:7480",
        aws_access_key_id="test-access",
        aws_secret_access_key="test-secret",
        region_name="us-east-1",
    )
    monkeypatch.setattr(storage, "get_s3_client", lambda: client)
    with Stubber(client) as stubber:
        yield stubber
        stubber.assert_no_pending_responses()
    client.close()


@pytest.mark.parametrize(
    "endpoint", ["http://storage.example:7480", "https://storage.example"]
)
def test_client_uses_aws_settings_and_path_style(monkeypatch, endpoint):
    for name, value in {
        "AWS_ENDPOINT_URL": endpoint,
        "AWS_ACCESS_KEY_ID": "test-access",
        "AWS_SECRET_ACCESS_KEY": "test-secret",
        "AWS_DEFAULT_REGION": "us-west-2",
    }.items():
        monkeypatch.setenv(name, value)
    factory = Mock()
    monkeypatch.setattr(storage.boto3, "client", factory)
    assert storage.get_s3_client() is factory.return_value
    assert factory.call_args.args == ("s3",)
    kwargs = factory.call_args.kwargs
    assert kwargs["endpoint_url"] == endpoint
    assert kwargs["aws_access_key_id"] == "test-access"
    assert kwargs["aws_secret_access_key"] == "test-secret"
    assert kwargs["region_name"] == "us-west-2"
    assert kwargs["config"].signature_version == "s3v4"
    assert kwargs["config"].s3 == {"addressing_style": "path"}


def test_client_defaults_match_local_storage(monkeypatch):
    for name in (
        "AWS_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    ):
        monkeypatch.delenv(name, raising=False)
    factory = Mock()
    monkeypatch.setattr(storage.boto3, "client", factory)
    storage.get_s3_client()
    kwargs = factory.call_args.kwargs
    assert kwargs["endpoint_url"] == "http://aws-compatible-storage:7480"
    assert kwargs["aws_access_key_id"] == "s4admin"
    assert kwargs["aws_secret_access_key"] == "s4secret"
    assert kwargs["region_name"] == "us-east-1"


def test_object_exists(stubbed_client):
    stubbed_client.add_response(
        "head_object", {}, {"Bucket": "config", "Key": "uploads/video.mp4"}
    )
    assert storage.object_exists("config", "uploads/video.mp4")


@pytest.mark.parametrize("code", ["404", "NoSuchKey", "NotFound"])
def test_missing_object_returns_false(stubbed_client, code):
    stubbed_client.add_client_error(
        "head_object",
        service_error_code=code,
        http_status_code=404,
        expected_params={"Bucket": "config", "Key": "missing"},
    )
    assert not storage.object_exists("config", "missing")


@pytest.mark.parametrize("code,status", [("AccessDenied", 403), ("InternalError", 500)])
def test_storage_errors_are_not_reported_as_missing(stubbed_client, code, status):
    stubbed_client.add_client_error(
        "head_object",
        service_error_code=code,
        http_status_code=status,
        expected_params={"Bucket": "config", "Key": "video.mp4"},
    )
    with pytest.raises(ClientError):
        storage.object_exists("config", "video.mp4")


def test_get_object_returns_closeable_body(stubbed_client):
    raw = io.BytesIO(b"jpeg")
    body = StreamingBody(raw, 4)
    stubbed_client.add_response(
        "get_object",
        {"Body": body},
        {"Bucket": "config", "Key": "thumbnails/video.jpg"},
    )
    stream = storage.get_object_stream("config", "thumbnails/video.jpg")
    assert stream.read() == b"jpeg"
    stream.close()
    assert raw.closed


@pytest.mark.parametrize("content_type", [None, "video/mp4"])
def test_upload_file_preserves_bucket_key_and_mime_type(monkeypatch, content_type):
    client = Mock()
    monkeypatch.setattr(storage, "get_s3_client", lambda: client)
    storage.upload_file("config", "uploads/video.mp4", "/tmp/video.mp4", content_type)
    client.upload_file.assert_called_once_with(
        "/tmp/video.mp4",
        "config",
        "uploads/video.mp4",
        ExtraArgs={"ContentType": content_type} if content_type else {},
    )


@pytest.mark.parametrize("content_type", [None, "image/jpeg"])
def test_upload_bytes_preserves_payload_and_mime_type(monkeypatch, content_type):
    client = Mock()
    monkeypatch.setattr(storage, "get_s3_client", lambda: client)
    storage.upload_bytes("config", "thumbnails/video.jpg", b"jpeg", content_type)
    args, kwargs = client.upload_fileobj.call_args
    assert args[0].read() == b"jpeg"
    assert args[1:] == ("config", "thumbnails/video.jpg")
    assert kwargs == {
        "ExtraArgs": {"ContentType": content_type} if content_type else {}
    }


def test_copy_uses_managed_transfer_for_large_objects(monkeypatch):
    client = Mock()
    monkeypatch.setattr(storage, "get_s3_client", lambda: client)
    storage.copy_object(
        "config", "uploads/video with spaces.mp4", "data", "video with spaces.mp4"
    )
    client.copy.assert_called_once_with(
        {"Bucket": "data", "Key": "video with spaces.mp4"},
        "config",
        "uploads/video with spaces.mp4",
    )


@pytest.mark.parametrize(
    "error",
    [
        ClientError({"Error": {"Code": "SlowDown"}}, "GetObject"),
        EndpointConnectionError(endpoint_url="http://storage.example:7480"),
        RetriesExceededError(OSError("connection lost")),
    ],
)
def test_download_retries_storage_failures(monkeypatch, tmp_path, error):
    client = Mock()
    client.download_file.side_effect = [error, None]
    sleep = Mock()
    monkeypatch.setattr(storage, "get_s3_client", lambda: client)
    monkeypatch.setattr(storage.time, "sleep", sleep)
    dest = str(tmp_path / "nested" / "video.mp4")
    assert (
        storage.download_file("data", "video.mp4", dest, max_retries=2, retry_delay=1)
        == dest
    )
    assert (tmp_path / "nested").is_dir()
    assert client.download_file.call_count == 2
    client.download_file.assert_called_with("data", "video.mp4", dest)
    sleep.assert_called_once_with(1)


def test_download_failure_propagates_after_last_attempt(monkeypatch, tmp_path):
    client = Mock()
    client.download_file.side_effect = EndpointConnectionError(
        endpoint_url="http://storage.example:7480"
    )
    monkeypatch.setattr(storage, "get_s3_client", lambda: client)
    sleep = Mock()
    monkeypatch.setattr(storage.time, "sleep", sleep)
    with pytest.raises(EndpointConnectionError):
        storage.download_file(
            "data", "video.mp4", str(tmp_path / "video.mp4"), max_retries=2
        )
    assert client.download_file.call_count == 2
    sleep.assert_called_once()


def test_download_supports_a_filename_without_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    client = Mock()
    monkeypatch.setattr(storage, "get_s3_client", lambda: client)
    assert storage.download_file("data", "video.mp4", "video.mp4") == "video.mp4"
