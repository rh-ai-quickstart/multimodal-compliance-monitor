"""
PostgreSQL database module for PPE compliance tracking.

Schema:
- app_config: User-defined configs (model_url, model_name for OVMS, video_source) — classes come from detection_classes
- detection_classes: Class definitions (model_class_index, name, trackable, include_in_counts) per config - FK to app_config
- detection_tracks: Generic tracks for detected objects
- detection_observations: Per-track observations with flexible JSONB attributes
"""

import os
import queue as queue_mod
import threading
from contextlib import contextmanager
from datetime import datetime

import psycopg2
from psycopg2.extras import RealDictCursor

from logger import get_logger

log = get_logger(__name__)


# Database connection settings from environment variables
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "ppe_tracking")
DB_USER = os.getenv("DB_USER", "ppe_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "ppe_password")


def get_connection_string():
    """Get the PostgreSQL connection string."""
    return f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD}"


@contextmanager
def get_connection():
    """Context manager for database connections."""
    conn = psycopg2.connect(get_connection_string())
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def get_readonly_connection():
    """Context manager for read-only database connections.

    Sets the session to readonly so PostgreSQL itself rejects any writes,
    regardless of what SQL is passed in.
    """
    conn = psycopg2.connect(get_connection_string())
    try:
        conn.set_session(readonly=True, autocommit=False)
        yield conn
    finally:
        conn.rollback()
        conn.close()


def init_database():
    """
    Initialize the database schema.
    Creates tables if they don't exist.
    Retries connection for Kubernetes (backend may start before PostgreSQL is ready).
    """
    import time

    max_retries = 10
    for attempt in range(max_retries):
        try:
            _init_schema()
            return
        except Exception as e:
            if attempt < max_retries - 1:
                log.warning(
                    "Database not ready (attempt %s/%s): %s. Retrying in 3s...",
                    attempt + 1,
                    max_retries,
                    e,
                )
                time.sleep(3)
            else:
                raise


def _init_schema():
    """Create tables if they don't exist. Data persists across restarts."""
    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS app_config (
                id SERIAL PRIMARY KEY,
                model_url VARCHAR(512) NOT NULL,
                video_source VARCHAR(1024) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            ALTER TABLE app_config
            ADD COLUMN IF NOT EXISTS model_name VARCHAR(128) NOT NULL DEFAULT 'ppe'
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detection_classes (
                id SERIAL PRIMARY KEY,
                app_config_id INTEGER NOT NULL REFERENCES app_config(id) ON DELETE CASCADE,
                model_class_index INTEGER NOT NULL,
                name VARCHAR(128) NOT NULL,
                trackable BOOLEAN NOT NULL DEFAULT false,
                CONSTRAINT uq_detection_classes_config_index UNIQUE (app_config_id, model_class_index)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_detection_classes_app_config_id
            ON detection_classes(app_config_id)
        """)
        cursor.execute("""
            ALTER TABLE detection_classes
            ADD COLUMN IF NOT EXISTS include_in_counts BOOLEAN NOT NULL DEFAULT true
        """)

        # Create detection_tracks table - generic tracks for any detection type
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detection_tracks (
                track_id INTEGER PRIMARY KEY,
                detection_classes_id INTEGER NOT NULL REFERENCES detection_classes(id)
                    ON DELETE CASCADE,
                first_seen TIMESTAMP NOT NULL,
                last_seen TIMESTAMP NOT NULL
            )
        """)

        # Create detection_observations table - flexible attributes via JSONB
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detection_observations (
                id SERIAL PRIMARY KEY,
                track_id INTEGER NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                attributes JSONB NOT NULL DEFAULT '{}',
                FOREIGN KEY (track_id) REFERENCES detection_tracks(track_id)
                    ON DELETE CASCADE
            )
        """)

        # Create indexes for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_detection_observations_track_id
            ON detection_observations(track_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_detection_observations_timestamp
            ON detection_observations(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_detection_observations_attributes
            ON detection_observations USING GIN (attributes)
        """)

        # Upgrade old DBs: FKs without ON DELETE CASCADE block deleting app_config when
        # tracks/observations exist. Recreate with CASCADE (idempotent for new installs).
        cursor.execute(
            "ALTER TABLE detection_tracks DROP CONSTRAINT IF EXISTS "
            "detection_tracks_detection_classes_id_fkey"
        )
        cursor.execute("""
            ALTER TABLE detection_tracks
            ADD CONSTRAINT detection_tracks_detection_classes_id_fkey
            FOREIGN KEY (detection_classes_id)
            REFERENCES detection_classes(id) ON DELETE CASCADE
        """)
        cursor.execute(
            "ALTER TABLE detection_observations DROP CONSTRAINT IF EXISTS "
            "detection_observations_track_id_fkey"
        )
        cursor.execute("""
            ALTER TABLE detection_observations
            ADD CONSTRAINT detection_observations_track_id_fkey
            FOREIGN KEY (track_id)
            REFERENCES detection_tracks(track_id) ON DELETE CASCADE
        """)

        conn.commit()
        log.debug("Schema created or verified (tables persist across restarts)")
        log.info("PostgreSQL database initialized: %s:%s/%s", DB_HOST, DB_PORT, DB_NAME)


# ----- Write Operations (used by tracker) -----


def insert_detection_track(
    track_id: int,
    detection_classes_id: int,
    first_seen: datetime,
    last_seen: datetime,
):
    """Insert a new detection track, or update last_seen if already exists."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO detection_tracks (track_id, detection_classes_id, first_seen, last_seen)
               VALUES (%s, %s, %s, %s)
               ON CONFLICT (track_id) DO UPDATE SET last_seen = EXCLUDED.last_seen""",
            (track_id, detection_classes_id, first_seen, last_seen),
        )
        conn.commit()


def update_detection_track_last_seen(track_id: int, last_seen: datetime):
    """Update the last_seen timestamp for an existing detection track."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE detection_tracks SET last_seen = %s WHERE track_id = %s",
            (last_seen, track_id),
        )
        conn.commit()


def insert_detection_observation(
    track_id: int,
    timestamp: datetime,
    attributes: dict,
):
    """Insert a new detection observation with flexible JSONB attributes."""
    import json

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO detection_observations (track_id, timestamp, attributes)
               VALUES (%s, %s, %s)""",
            (track_id, timestamp, json.dumps(attributes)),
        )
        conn.commit()


# ----- Async DB Writer Thread -----

_INSERT_TRACK_SQL = (
    "INSERT INTO detection_tracks (track_id, detection_classes_id, first_seen, last_seen) "
    "VALUES (%s, %s, %s, %s) "
    "ON CONFLICT (track_id) DO UPDATE SET last_seen = EXCLUDED.last_seen"
)
_UPDATE_LAST_SEEN_SQL = "UPDATE detection_tracks SET last_seen = %s WHERE track_id = %s"
_INSERT_OBSERVATION_SQL = (
    "INSERT INTO detection_observations (track_id, timestamp, attributes) "
    "VALUES (%s, %s, %s)"
)

OP_INSERT_TRACK = "insert_track"
OP_UPDATE_LAST_SEEN = "update_last_seen"
OP_INSERT_OBSERVATION = "insert_observation"

_SQL_BY_OP = {
    OP_INSERT_TRACK: _INSERT_TRACK_SQL,
    OP_UPDATE_LAST_SEEN: _UPDATE_LAST_SEEN_SQL,
    OP_INSERT_OBSERVATION: _INSERT_OBSERVATION_SQL,
}

# Execution order: tracks must exist before observations reference them.
_OP_ORDER = (OP_INSERT_TRACK, OP_UPDATE_LAST_SEEN, OP_INSERT_OBSERVATION)


class DbWriterThread:
    """Background thread that batches DB writes from a queue.

    The inference loop enqueues lightweight (op, args) tuples via ``enqueue``.
    A single daemon thread drains the queue in batches, groups by operation
    type, and executes each group with ``executemany`` inside one transaction.
    A persistent connection is reused across batches and reconnected on failure.
    """

    def __init__(self, max_batch: int = 50, poll_timeout: float = 0.05):
        self._queue: queue_mod.Queue = queue_mod.Queue(maxsize=5000)
        self._stop = threading.Event()
        self._max_batch = max_batch
        self._poll_timeout = poll_timeout
        self._conn = None
        self._thread = threading.Thread(
            target=self._run,
            name="db-writer",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout)

    def enqueue(self, op: str, args: tuple) -> None:
        try:
            self._queue.put_nowait((op, args))
        except queue_mod.Full:
            log.warning("DB writer queue full, dropping %s", op)

    # -- internals --

    def _ensure_conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(get_connection_string())
        return self._conn

    def _close_conn(self):
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def _drain_batch(self) -> list[tuple[str, tuple]]:
        """Block for one item, then greedily pull up to max_batch more."""
        batch: list[tuple[str, tuple]] = []
        try:
            item = self._queue.get(timeout=self._poll_timeout)
            batch.append(item)
        except queue_mod.Empty:
            return batch
        for _ in range(self._max_batch - 1):
            try:
                batch.append(self._queue.get_nowait())
            except queue_mod.Empty:
                break
        return batch

    def _execute_batch(self, batch: list[tuple[str, tuple]]) -> None:
        groups: dict[str, list[tuple]] = {}
        for op, args in batch:
            groups.setdefault(op, []).append(args)

        conn = self._ensure_conn()
        cursor = conn.cursor()
        try:
            for op in _OP_ORDER:
                rows = groups.get(op)
                if rows:
                    cursor.executemany(_SQL_BY_OP[op], rows)
            conn.commit()
        except psycopg2.OperationalError:
            log.warning("DB writer: connection lost, reconnecting")
            self._close_conn()
            try:
                conn = self._ensure_conn()
                cursor = conn.cursor()
                for op in _OP_ORDER:
                    rows = groups.get(op)
                    if rows:
                        cursor.executemany(_SQL_BY_OP[op], rows)
                conn.commit()
            except Exception:
                log.exception("DB writer: retry failed, dropping %d items", len(batch))
                self._close_conn()
        except Exception:
            log.exception("DB writer: batch failed, dropping %d items", len(batch))
            try:
                conn.rollback()
            except Exception:
                self._close_conn()

    def _run(self) -> None:
        log.info("DB writer thread started")
        try:
            while not self._stop.is_set():
                batch = self._drain_batch()
                if batch:
                    self._execute_batch(batch)

            # Drain remaining items after stop signal
            remaining: list[tuple[str, tuple]] = []
            while True:
                try:
                    remaining.append(self._queue.get_nowait())
                except queue_mod.Empty:
                    break
            if remaining:
                log.info("DB writer: flushing %d remaining items", len(remaining))
                self._execute_batch(remaining)
        except Exception:
            log.exception("DB writer thread crashed")
        finally:
            self._close_conn()
            log.info("DB writer thread exited")


# ----- Detection Classes Operations -----


def replace_detection_classes(
    app_config_id: int, entries: list[tuple[int, str, bool, bool]]
) -> None:
    """
    Replace all detection classes for a config. Deletes existing, inserts new.
    entries: list of (model_class_index, name, trackable, include_in_counts).
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM detection_classes WHERE app_config_id = %s",
            (app_config_id,),
        )
        for model_class_index, name, trackable, include_in_counts in entries:
            cursor.execute(
                """INSERT INTO detection_classes
                   (app_config_id, model_class_index, name, trackable, include_in_counts)
                   VALUES (%s, %s, %s, %s, %s)""",
                (
                    app_config_id,
                    model_class_index,
                    name.strip(),
                    bool(trackable),
                    bool(include_in_counts),
                ),
            )
        conn.commit()


def get_detection_classes_for_config(app_config_id: int) -> dict[int, str]:
    """Return {model_class_index: name} for the given config (used by Runtime)."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT model_class_index, name FROM detection_classes
               WHERE app_config_id = %s ORDER BY model_class_index""",
            (app_config_id,),
        )
        return {row[0]: row[1] for row in cursor.fetchall()}


def get_include_in_counts_by_class_index(app_config_id: int) -> dict[int, bool]:
    """Return {model_class_index: include_in_counts} for process_detections."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT model_class_index, include_in_counts FROM detection_classes
               WHERE app_config_id = %s ORDER BY model_class_index""",
            (app_config_id,),
        )
        return {row[0]: bool(row[1]) for row in cursor.fetchall()}


def get_detection_classes_pipeline_maps(
    app_config_id: int,
) -> tuple[dict[int, str], dict[int, bool], dict[int, bool], dict[str, int]]:
    """
    Single query for inference startup: class names for Runtime, flags for
    process_detections, and detection_classes.id by class name (for track FK).
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT id, model_class_index, name, include_in_counts, trackable
               FROM detection_classes
               WHERE app_config_id = %s ORDER BY model_class_index""",
            (app_config_id,),
        )
        rows = cursor.fetchall()
    classes: dict[int, str] = {}
    include_in_counts: dict[int, bool] = {}
    trackable: dict[int, bool] = {}
    name_to_id: dict[str, int] = {}
    for row_id, model_class_index, name, inc, trk in rows:
        classes[model_class_index] = name
        include_in_counts[model_class_index] = bool(inc)
        trackable[model_class_index] = bool(trk)
        name_to_id[name.strip()] = int(row_id)
    return classes, include_in_counts, trackable, name_to_id


def get_classes_for_config(app_config_id: int) -> dict:
    """
    Build classes dict from detection_classes for API response.
    Returns {"0":{"name","trackable","include_in_counts"}, ...}.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT model_class_index, name, trackable, include_in_counts
               FROM detection_classes
               WHERE app_config_id = %s ORDER BY model_class_index""",
            (app_config_id,),
        )
        return {
            str(row[0]): {
                "name": row[1],
                "trackable": row[2],
                "include_in_counts": bool(row[3]),
            }
            for row in cursor.fetchall()
        }


def get_detection_class_by_name_and_config(
    name: str, app_config_id: int
) -> dict | None:
    """Return detection class row as dict, or None if not found."""
    with get_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            """SELECT id, app_config_id, model_class_index, name, trackable, include_in_counts
               FROM detection_classes WHERE name = %s AND app_config_id = %s""",
            (name.strip(), app_config_id),
        )
        row = cursor.fetchone()
        return dict(row) if row else None


# ----- App Config Operations -----


def count_app_configs() -> int:
    """Return the number of rows in app_config (for idempotent demo seeding)."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM app_config")
        return int(cursor.fetchone()[0])


def get_all_configs() -> list:
    """Return all app_config rows as list of dicts, with classes built from detection_classes."""
    with get_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            """SELECT id, model_url, model_name, video_source, created_at
               FROM app_config ORDER BY id"""
        )
        configs = [dict(row) for row in cursor.fetchall()]
    for c in configs:
        c["classes"] = get_classes_for_config(c["id"])
    return configs


def get_config_by_id(config_id: int) -> dict | None:
    """Return a single config by id, or None if not found."""
    with get_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            """SELECT id, model_url, model_name, video_source, created_at
               FROM app_config WHERE id = %s""",
            (config_id,),
        )
        row = cursor.fetchone()
    if row is None:
        return None
    c = dict(row)
    c["classes"] = get_classes_for_config(c["id"])
    return c


def insert_config(
    model_url: str,
    video_source: str,
    model_name: str,
) -> int:
    """Insert a new config and return the inserted id. Classes go to detection_classes via replace_detection_classes."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO app_config (model_url, video_source, model_name)
               VALUES (%s, %s, %s) RETURNING id""",
            (model_url, video_source, model_name),
        )
        row_id = cursor.fetchone()[0]
        conn.commit()
        return row_id


def delete_config(config_id: int) -> bool:
    """Delete a config row. Cascades to detection_classes, detection_tracks, detection_observations."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM app_config WHERE id = %s", (config_id,))
        deleted = cursor.rowcount > 0
        conn.commit()
        return deleted


def clear_all_data() -> None:
    """Remove all rows from all tables. Resets sequences. For fresh testing."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM detection_observations")
        cursor.execute("DELETE FROM detection_tracks")
        cursor.execute("DELETE FROM detection_classes")
        cursor.execute("DELETE FROM app_config")
        # Reset sequences so new inserts get id=1, etc.
        cursor.execute("ALTER SEQUENCE app_config_id_seq RESTART WITH 1")
        cursor.execute("ALTER SEQUENCE detection_classes_id_seq RESTART WITH 1")
        cursor.execute("ALTER SEQUENCE detection_observations_id_seq RESTART WITH 1")
        conn.commit()
        log.info("All tables cleared")


# ----- Text-to-SQL Operations (used by chatbot) -----


def execute_query(sql: str) -> list:
    """
    Execute a SELECT query and return results as list of dicts.
    Used by the Text-to-SQL chatbot to run LLM-generated queries.

    Safety: Only SELECT queries are allowed. Dangerous keywords are blocked.
    """
    sql_upper = sql.strip().upper()

    # Only allow SELECT queries
    if not sql_upper.startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed")

    # Block dangerous keywords
    dangerous_keywords = [
        "DROP",
        "DELETE",
        "UPDATE",
        "INSERT",
        "ALTER",
        "TRUNCATE",
        "CREATE",
        "GRANT",
        "REVOKE",
    ]
    for keyword in dangerous_keywords:
        if keyword in sql_upper:
            raise ValueError(f"Query contains forbidden keyword: {keyword}")

    sql_preview = sql.strip()[:120] + "..." if len(sql.strip()) > 120 else sql.strip()
    log.debug("Executing query: %s", sql_preview)

    with get_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(sql)
        results = cursor.fetchall()
        log.debug("Query returned %s row(s)", len(results))
        return [dict(row) for row in results]


def get_schema_description() -> str:
    """
    Return the database schema description for the LLM prompt.
    Used by the Text-to-SQL chatbot to understand the data model.
    """
    return """
DATABASE SCHEMA:

Table: app_config
- id (SERIAL, PRIMARY KEY): Config ID
- model_url (VARCHAR): OVMS gRPC/REST host (endpoint URL)
- model_name (VARCHAR): OVMS served model id (must match model name in OVMS config)
- video_source (VARCHAR): Video path or RTSP URL
- created_at (TIMESTAMP): When config was created
- Inference input tensor name: backend env MODEL_INPUT_NAME only (not in this table; Runtime defaults to x if unset)
- classes: Derived from detection_classes (model_class_index, name, trackable)

Table: detection_classes
- id (SERIAL, PRIMARY KEY): Class ID
- app_config_id (INTEGER, FOREIGN KEY → app_config.id): Config that defines this class
- model_class_index (INTEGER): Model output index (0, 1, 2, ...)
- name (VARCHAR): Class name, e.g. "Person", "Hardhat"
- trackable (BOOLEAN): Whether to track this class for unique counting

Table: detection_tracks
- track_id (INTEGER, PRIMARY KEY): Unique identifier for each tracked detection
- detection_classes_id (INTEGER, FOREIGN KEY → detection_classes.id): Links to the detection class (e.g. Person)
- first_seen (TIMESTAMP): When the track was first detected
- last_seen (TIMESTAMP): When the track was last detected

Table: detection_observations
- id (SERIAL, PRIMARY KEY): Auto-incrementing observation ID
- track_id (INTEGER, FOREIGN KEY → detection_tracks.track_id): Links to the track
- timestamp (TIMESTAMP): When this observation was recorded
- attributes (JSONB): Flexible attributes. For PPE persons: {"hardhat": true/false, "vest": true/false, "mask": true/false}

NOTES:
- For PPE: detection_classes_id points to Person. A "violation" means attributes->>'hardhat'='false', etc.
- Query PPE: (attributes->>'hardhat')::boolean = false for hardhat violations
- Use COUNT(DISTINCT track_id) to count unique people
- Use CURRENT_TIMESTAMP for current time, CURRENT_DATE for today
- Use INTERVAL for date math: CURRENT_DATE - INTERVAL '7 days'
- Use EXTRACT(DOW FROM timestamp) for day of week (0=Sunday, 6=Saturday)
- Use DATE_TRUNC('day', timestamp) to group by date
- Use TO_CHAR(timestamp, 'Day') to get day name
"""
