"""Upload and thumbnail API contracts, including boto3 stream cleanup."""

import io
from unittest.mock import Mock

import pytest
from botocore.exceptions import ClientError
from botocore.response import StreamingBody

from test_alert_endpoints import _load_app_module


def test_upload_returns_same_s3_uri(monkeypatch):
    module = _load_app_module(monkeypatch)
    upload = Mock()
    monkeypatch.setattr(module, "upload_bytes", upload)
    response = module.app.test_client().post(
        "/api/config/upload",
        data={"file": (io.BytesIO(b"video"), "sample.mp4")},
    )
    assert response.status_code == 200
    assert response.json == {
        "path": "s3://config/uploads/sample.mp4",
        "filename": "sample.mp4",
    }
    upload.assert_called_once_with(
        "config", "uploads/sample.mp4", b"video", content_type="video/mp4"
    )


@pytest.mark.parametrize("read_error", [False, True])
def test_thumbnail_closes_boto3_stream_even_on_failure(monkeypatch, read_error):
    module = _load_app_module(monkeypatch)
    raw = io.BytesIO(b"jpeg")
    stream = StreamingBody(raw, 4)
    if read_error:
        monkeypatch.setattr(stream, "read", Mock(side_effect=OSError("read failed")))
    monkeypatch.setattr(module, "object_exists", lambda *_args: True)
    monkeypatch.setattr(module, "get_object_stream", lambda *_args: stream)
    response = module.app.test_client().get("/api/thumbnails/sample.jpg")
    assert raw.closed
    if read_error:
        assert response.status_code == 500
        assert response.json == {"error": "Failed to load thumbnail"}
    else:
        assert response.status_code == 200
        assert response.data == b"jpeg"
        assert response.content_type == "image/jpeg"


def test_missing_thumbnail_returns_404(monkeypatch):
    module = _load_app_module(monkeypatch)
    response = module.app.test_client().get("/api/thumbnails/missing.jpg")
    assert response.status_code == 404


def test_storage_access_error_returns_500_not_404(monkeypatch):
    module = _load_app_module(monkeypatch)
    monkeypatch.setattr(
        module,
        "object_exists",
        Mock(
            side_effect=ClientError(
                {"Error": {"Code": "AccessDenied"}},
                "HeadObject",
            )
        ),
    )
    response = module.app.test_client().get("/api/thumbnails/sample.jpg")
    assert response.status_code == 500
    assert response.json == {"error": "Failed to load thumbnail"}
