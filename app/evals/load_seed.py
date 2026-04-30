"""Save/restore live database and load eval seed data.

Provides a save/restore pattern so evals can temporarily replace the live
database contents with a known seed, then put everything back afterwards.

The snapshot is written to a volume-mounted path (SNAPSHOT_DIR env var,
default ``/snapshots``) so it survives container crashes and can be
manually restored if the ``finally`` block never ran.
"""

import os
from datetime import datetime, date
from decimal import Decimal
from pathlib import Path

from database import get_connection

TABLES = [
    "app_config",
    "detection_classes",
    "detection_tracks",
    "detection_observations",
]

TRUNCATE_ORDER = (
    "detection_observations, detection_tracks, detection_classes, app_config"
)

SNAPSHOT_DIR = Path(os.getenv("SNAPSHOT_DIR", "/snapshots"))
SNAPSHOT_FILE = SNAPSHOT_DIR / "live_backup.sql"

SKIP_PREFIXES = ("--", "\\", "SET ", "SELECT pg_catalog.set_config")

_SEQUENCES = [
    "app_config_id_seq",
    "detection_classes_id_seq",
    "detection_observations_id_seq",
]


def _sql_literal(value) -> str:
    """Convert a Python value to a SQL literal string."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, Decimal)):
        return str(value)
    if isinstance(value, datetime):
        return f"'{value.isoformat()}'"
    if isinstance(value, date):
        return f"'{value.isoformat()}'"
    if isinstance(value, dict):
        import json

        return f"'{json.dumps(value)}'"
    s = str(value).replace("'", "''")
    return f"'{s}'"


def _dump_tables_to_file(path: Path) -> int:
    """Dump all 4 tables as INSERT statements to *path*. Returns statement count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with get_connection() as conn, open(path, "w") as f:
        cur = conn.cursor()
        f.write("--\n-- PostgreSQL database dump\n--\n\n")
        for table in TABLES:
            f.write(f"--\n-- Data for Name: {table}\n--\n\n")
            cur.execute(f"SELECT * FROM {table}")
            cols = [desc[0] for desc in cur.description]
            col_list = ", ".join(f'"{c}"' if c in ("timestamp",) else c for c in cols)
            for row in cur.fetchall():
                vals = ", ".join(_sql_literal(v) for v in row)
                f.write(f"INSERT INTO public.{table} ({col_list}) VALUES ({vals});\n")
                count += 1

        f.write("\n")
        for seq in _SEQUENCES:
            try:
                cur.execute(f"SELECT last_value FROM {seq}")
                val = cur.fetchone()[0]
                f.write(f"SELECT pg_catalog.setval('public.{seq}', {val}, true);\n")
                count += 1
            except Exception:
                conn.rollback()

        f.write("\n--\n-- PostgreSQL database dump complete\n--\n")
    return count


def _load_sql_file(sql_path: Path) -> dict[str, int]:
    """Truncate all tables and replay INSERT/setval statements from *sql_path*.

    Returns per-table row counts after loading.
    """
    statements: list[str] = []
    setval_stmts: list[str] = []

    with open(sql_path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or any(stripped.startswith(p) for p in SKIP_PREFIXES):
                if stripped.startswith("SELECT pg_catalog.setval"):
                    setval_stmts.append(stripped.rstrip(";") + ";")
                continue
            if stripped.startswith("INSERT INTO"):
                statements.append(stripped)

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(f"TRUNCATE {TRUNCATE_ORDER} CASCADE")

        for stmt in statements:
            cur.execute(stmt)

        for sv in setval_stmts:
            cur.execute(sv)

        counts: dict[str, int] = {}
        for table in TABLES:
            cur.execute(f"SELECT count(*) FROM {table}")
            counts[table] = cur.fetchone()[0]

        conn.commit()

    return counts


def save_snapshot(path: Path = SNAPSHOT_FILE) -> int:
    """Dump current live data to a SQL file on the mounted volume.

    Returns the number of statements written.
    """
    return _dump_tables_to_file(path)


def _shift_timestamps_to_now() -> None:
    """Shift all seed timestamps so the newest observation is NOW() - 1 second.

    The seed data spans ~2 minutes. After shifting, all data falls within
    the last ~2:13 which keeps every time-window query (last 3 min etc.)
    valid regardless of when the eval actually runs.
    """
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE detection_observations
            SET "timestamp" = "timestamp" + (
                NOW() - INTERVAL '1 second'
                - (SELECT MAX("timestamp") FROM detection_observations)
            )
        """)
        cur.execute("""
            UPDATE detection_tracks
            SET first_seen = first_seen + (
                NOW() - INTERVAL '1 second'
                - (SELECT MAX(last_seen) FROM detection_tracks)
            ),
            last_seen = last_seen + (
                NOW() - INTERVAL '1 second'
                - (SELECT MAX(last_seen) FROM detection_tracks)
            )
        """)
        conn.commit()


def load_seed(sql_path: Path) -> dict[str, int]:
    """Truncate the 4 eval tables and load seed data from *sql_path*.

    After loading, shifts all timestamps to be relative to NOW() so that
    time-based queries work regardless of when the eval runs.

    Returns per-table row counts.
    """
    counts = _load_sql_file(sql_path)
    _shift_timestamps_to_now()
    return counts


def restore_snapshot(path: Path = SNAPSHOT_FILE) -> dict[str, int]:
    """Restore live data from the snapshot file on the volume.

    Returns per-table row counts after restoring.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Snapshot file not found: {path}. Cannot restore live data."
        )
    return _load_sql_file(path)
