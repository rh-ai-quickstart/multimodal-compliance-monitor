"""Unit tests for database.execute_query SQL safety guard."""

from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from database import execute_query


@pytest.fixture(autouse=True)
def mock_db_connection(monkeypatch):
    """Patch get_connection so passing queries don't need a real DB."""
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [{"track_id": 1}]
    mock_cursor.description = [("track_id",)]

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    @contextmanager
    def fake_connection():
        yield mock_conn

    monkeypatch.setattr("database.get_connection", fake_connection)


# ── Rejection cases (raise ValueError before DB is touched) ─────────────────


class TestExecuteQueryRejection:
    def test_rejects_update(self):
        with pytest.raises(ValueError, match="Only SELECT"):
            execute_query(
                "UPDATE detection_tracks SET last_seen = NOW() WHERE track_id = 15323"
            )

    def test_rejects_insert(self):
        with pytest.raises(ValueError, match="Only SELECT"):
            execute_query(
                "INSERT INTO detection_observations (track_id, timestamp, attributes) VALUES (1, NOW(), '{}')"
            )

    def test_rejects_delete(self):
        with pytest.raises(ValueError, match="Only SELECT"):
            execute_query("DELETE FROM detection_tracks WHERE track_id = 15323")

    @pytest.mark.parametrize(
        "keyword",
        [
            "DROP",
            "DELETE",
            "UPDATE",
            "INSERT",
            "ALTER",
            "TRUNCATE",
            "CREATE",
            "GRANT",
            "REVOKE",
        ],
    )
    def test_rejects_forbidden_keyword_in_select(self, keyword):
        sql = f"SELECT * FROM detection_tracks; {keyword} TABLE detection_tracks"
        with pytest.raises(ValueError, match="forbidden keyword"):
            execute_query(sql)

    def test_bypass_case_still_caught(self):
        with pytest.raises(ValueError, match="forbidden keyword"):
            execute_query("select * from detection_tracks; drop table detection_tracks")

    def test_bypass_leading_whitespace(self):
        with pytest.raises(ValueError, match="Only SELECT"):
            execute_query("   UPDATE detection_tracks SET last_seen = NOW()")

    def test_bypass_subquery_with_delete(self):
        with pytest.raises(ValueError, match="forbidden keyword"):
            execute_query("SELECT * FROM (DELETE FROM detection_tracks RETURNING *) x")

    def test_bypass_comment_with_drop(self):
        with pytest.raises(ValueError, match="forbidden keyword"):
            execute_query("SELECT /* DROP */ * FROM detection_tracks")

    def test_empty_string_rejected(self):
        with pytest.raises(ValueError, match="Only SELECT"):
            execute_query("")

    def test_whitespace_only_rejected(self):
        with pytest.raises(ValueError, match="Only SELECT"):
            execute_query("   ")


# ── Pass cases (valid SELECT queries reach the mock DB) ─────────────────────


class TestExecuteQueryPass:
    def test_real_llm_query_passes(self):
        sql = (
            "SELECT track_id, timestamp, attributes FROM detection_observations obs "
            "JOIN detection_tracks dt ON obs.track_id = dt.track_id "
            "JOIN detection_classes dc ON dt.detection_classes_id = dc.id "
            "WHERE dc.app_config_id = 2 AND obs.attributes != '{}'"
        )
        result = execute_query(sql)
        assert isinstance(result, list)

    def test_real_production_query_passes(self):
        sql = (
            "SELECT dt.track_id, dc.name as class_name, dt.first_seen, dt.last_seen "
            "FROM detection_tracks dt "
            "JOIN detection_classes dc ON dt.detection_classes_id = dc.id "
            "ORDER BY dt.track_id DESC LIMIT 15"
        )
        result = execute_query(sql)
        assert isinstance(result, list)

    def test_keyword_in_string_value_is_known_limitation(self):
        sql = "SELECT * FROM detection_classes WHERE name = 'DROP zone'"
        with pytest.raises(ValueError, match="forbidden keyword"):
            execute_query(sql)
