"""Unit tests for MCP execute_sql scoping guard in tools/mcp_tools.py."""

import asyncio

import pytest

from tools.mcp_tools import _wrap_execute_sql, current_app_config_id, _SCOPED_TABLES


class _FakeTool:
    """Minimal fake tool that returns the SQL it receives."""

    name = "execute_sql"
    description = "Execute SQL"

    async def ainvoke(self, args: dict) -> str:
        return f"OK: {args['sql']}"


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture
def wrapped_tool():
    return _wrap_execute_sql(_FakeTool())


@pytest.fixture(autouse=True)
def reset_context_var():
    token = current_app_config_id.set(None)
    yield
    current_app_config_id.reset(token)


# ── With current_app_config_id = 2 (PPE config active) ─────────────────────


class TestScopingWithConfigId:
    @pytest.fixture(autouse=True)
    def set_config_id(self):
        token = current_app_config_id.set(2)
        yield
        current_app_config_id.reset(token)

    def test_rejects_detection_tracks_without_filter(self, wrapped_tool):
        result = _run(wrapped_tool.coroutine("SELECT * FROM detection_tracks"))
        assert "ERROR" in result
        assert "app_config_id = 2" in result

    def test_rejects_multi_table_without_filter(self, wrapped_tool):
        sql = (
            "SELECT obs.track_id, obs.attributes FROM detection_observations obs "
            "JOIN detection_tracks dt ON obs.track_id = dt.track_id"
        )
        result = _run(wrapped_tool.coroutine(sql))
        assert "ERROR" in result

    def test_passes_with_correct_filter(self, wrapped_tool):
        sql = (
            "SELECT dt.track_id, dc.name FROM detection_tracks dt "
            "JOIN detection_classes dc ON dt.detection_classes_id = dc.id "
            "WHERE dc.app_config_id = 2"
        )
        result = _run(wrapped_tool.coroutine(sql))
        assert result.startswith("OK:")

    def test_passes_unscoped_table(self, wrapped_tool):
        result = _run(wrapped_tool.coroutine("SELECT * FROM app_config WHERE id = 2"))
        assert result.startswith("OK:")

    @pytest.mark.parametrize("table", sorted(_SCOPED_TABLES))
    def test_each_scoped_table_rejected_without_filter(self, wrapped_tool, table):
        result = _run(wrapped_tool.coroutine(f"SELECT * FROM {table}"))
        assert "ERROR" in result

    def test_wrong_config_id_rejected(self, wrapped_tool):
        sql = (
            "SELECT * FROM detection_tracks WHERE detection_classes.app_config_id = 99"
        )
        result = _run(wrapped_tool.coroutine(sql))
        assert "ERROR" in result

    def test_like_instead_of_equals_rejected(self, wrapped_tool):
        sql = "SELECT * FROM detection_tracks WHERE app_config_id LIKE '2'"
        result = _run(wrapped_tool.coroutine(sql))
        assert "ERROR" in result

    def test_table_name_as_substring_triggers(self, wrapped_tool):
        # Known behavior: simple `in` check matches substrings
        sql = "SELECT * FROM my_detection_tracks_backup"
        result = _run(wrapped_tool.coroutine(sql))
        assert "ERROR" in result


# ── With current_app_config_id unset (None) ─────────────────────────────────


class TestScopingWithoutConfigId:
    def test_detection_tracks_passes(self, wrapped_tool):
        result = _run(wrapped_tool.coroutine("SELECT * FROM detection_tracks"))
        assert result.startswith("OK:")

    def test_detection_observations_passes(self, wrapped_tool):
        result = _run(wrapped_tool.coroutine("SELECT * FROM detection_observations"))
        assert result.startswith("OK:")
