"""Unit tests for alert update/delete API endpoints."""

from __future__ import annotations

import importlib
import sys
import types


class _DummyVideoHandler:
    def __init__(self):
        self._active_config_id = None
        self.video_source = ""

    def frame_generator(self):
        if False:
            yield b""

    def get_latested_description(self):
        return ""

    def get_latest_summary(self):
        return ""

    def switch_video_source(self, video_source, config_id):
        self.video_source = video_source
        self._active_config_id = config_id

    def stop_streaming_if_active_config(self, _config_id):
        return None


class _DummyLLMAlert:
    def create_alert(self, alert_text, app_config_id, classes_info=None):
        return f"SELECT '{alert_text}' AS rule, {app_config_id} AS app_config_id"


def _load_app_module(monkeypatch):
    """Import backend app with lightweight stubs for external dependencies."""
    tracing_mod = types.ModuleType("tracing")
    tracing_mod.init_tracing = lambda: None
    monkeypatch.setitem(sys.modules, "tracing", tracing_mod)

    logger_mod = types.ModuleType("logger")

    class _DummyLogger:
        def info(self, *args, **kwargs):
            return None

        def debug(self, *args, **kwargs):
            return None

        def exception(self, *args, **kwargs):
            return None

    logger_mod.get_logger = lambda _name: _DummyLogger()
    monkeypatch.setitem(sys.modules, "logger", logger_mod)

    minio_mod = types.ModuleType("minio_client")
    minio_mod.get_config_bucket = lambda: "config"
    minio_mod.upload_bytes = lambda *args, **kwargs: None
    minio_mod.get_object_stream = lambda *args, **kwargs: None
    minio_mod.object_exists = lambda *args, **kwargs: False
    monkeypatch.setitem(sys.modules, "minio_client", minio_mod)

    db_mod = types.ModuleType("database")
    db_mod.count_app_configs = lambda: 1
    db_mod.execute_query = lambda _query: [{"count": 0}]
    db_mod.get_all_configs = lambda: []
    db_mod.get_classes_for_config = (
        lambda _app_config_id: {"0": {"name": "Person", "trackable": True}}
    )
    db_mod.get_config_by_id = lambda _cid: None
    db_mod.insert_config = lambda *_args, **_kwargs: 1
    db_mod.delete_config = lambda *_args, **_kwargs: False
    db_mod.replace_detection_classes = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "database", db_mod)

    chat_mod = types.ModuleType("chat")

    class _DummyLLMChat:
        def chat(self, **_kwargs):
            return "ok"

        def clear_history(self, _session_id):
            return None

    chat_mod.LLMChat = _DummyLLMChat
    monkeypatch.setitem(sys.modules, "chat", chat_mod)

    seed_mod = types.ModuleType("seed_demo_configs")
    seed_mod.insert_demo_configs = lambda: None
    monkeypatch.setitem(sys.modules, "seed_demo_configs", seed_mod)

    thumb_mod = types.ModuleType("thumbnail_utils")
    thumb_mod.generate_thumbnail_for_video_source = lambda *_args, **_kwargs: None
    thumb_mod.is_s3_video_path = lambda _path: False
    monkeypatch.setitem(sys.modules, "thumbnail_utils", thumb_mod)

    video_handler_mod = types.ModuleType("video_processing.video_handler")
    video_handler_mod.VideoHandler = _DummyVideoHandler
    monkeypatch.setitem(sys.modules, "video_processing.video_handler", video_handler_mod)

    alert_graph_mod = types.ModuleType("alert.graph")
    alert_graph_mod.LLMAlert = _DummyLLMAlert
    monkeypatch.setitem(sys.modules, "alert.graph", alert_graph_mod)

    sys.modules.pop("app", None)
    return importlib.import_module("app")


def _seed_alert(app_module, *, app_config_id=7, alert_id="a1"):
    entry = app_module.AlertEntry(
        id=alert_id,
        app_config_id=app_config_id,
        rule="alert when no hardhat",
        severity="medium",
        status="done",
        sql_query="SELECT 1",
    )
    app_module.alerts = app_module.AlertsStore(configs={app_config_id: {alert_id: entry}})
    return entry


def test_patch_alert_updates_rule_and_severity(monkeypatch):
    app_module = _load_app_module(monkeypatch)
    _seed_alert(app_module, app_config_id=7, alert_id="a1")

    client = app_module.app.test_client()
    res = client.patch(
        "/api/alerts/7/a1",
        json={
            "rule": "alert when more than 3 people have no vests",
            "severity": "high",
        },
    )

    assert res.status_code == 200
    payload = res.get_json()
    assert payload["id"] == "a1"
    assert payload["app_config_id"] == 7
    assert payload["severity"] == "high"
    assert payload["rule"] == "alert when more than 3 people have no vests"
    assert payload["status"] == "done"
    assert "SELECT 'alert when more than 3 people have no vests'" in payload["sql_query"]


def test_patch_alert_rejects_invalid_severity(monkeypatch):
    app_module = _load_app_module(monkeypatch)
    _seed_alert(app_module, app_config_id=9, alert_id="a2")

    client = app_module.app.test_client()
    res = client.patch("/api/alerts/9/a2", json={"severity": "critical"})

    assert res.status_code == 400
    assert "severity must be" in res.get_json()["error"]


def test_patch_alert_rejects_empty_rule(monkeypatch):
    app_module = _load_app_module(monkeypatch)
    _seed_alert(app_module, app_config_id=5, alert_id="a3")

    client = app_module.app.test_client()
    res = client.patch("/api/alerts/5/a3", json={"rule": "   "})

    assert res.status_code == 400
    assert "cannot be empty" in res.get_json()["error"]


def test_delete_alert_removes_existing_alert(monkeypatch):
    app_module = _load_app_module(monkeypatch)
    _seed_alert(app_module, app_config_id=3, alert_id="a4")

    client = app_module.app.test_client()
    res = client.delete("/api/alerts/3/a4")

    assert res.status_code == 200
    assert res.get_json()["message"] == "Alert deleted"
    assert "a4" not in app_module.alerts.configs.get(3, {})


def test_delete_alert_returns_404_for_missing_alert(monkeypatch):
    app_module = _load_app_module(monkeypatch)
    app_module.alerts = app_module.AlertsStore(configs={})

    client = app_module.app.test_client()
    res = client.delete("/api/alerts/42/missing")

    assert res.status_code == 404
    assert res.get_json()["error"] == "Alert not found"
