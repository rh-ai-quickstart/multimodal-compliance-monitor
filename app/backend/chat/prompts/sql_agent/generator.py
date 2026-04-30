from __future__ import annotations

from chat.prompts._utils import compact_classes, pick_example_class
from database import get_schema_description
from logger import get_logger

logger = get_logger(__name__)

_SQL_GENERATION_RULES = """\
SQL GENERATION RULES — read carefully:
1. Write a SINGLE plain SQL SELECT statement.
2. Never wrap SQL in markdown fences (```), quotes, or any extra text.
3. Return exactly ONE statement — no semicolons separating multiple statements.
4. Output ONLY the raw SQL, nothing else.
5. Make sure majority vote is used for each track_id.
"""

_SQL_GENERATION_NOTE = """\
NOTES:
- Each track has many noisy observations. Classify a track by majority vote, not by any single observation.
- Use GROUP BY track_id with COUNT(*) FILTER (WHERE (attributes->>'attr')::boolean = false/true) to tally per-track.
- Use HAVING with majority vote in BOTH directions (false > true AND true > false) — never shortcut either with COUNT(DISTINCT) + a WHERE filter.
- Observations where the attribute key is absent are naturally ignored by FILTER (NULL fails the boolean cast). Use the exact JSONB key from the schema — never infer or rename keys from class display names.
- Use COUNT(DISTINCT track_id) to count unique tracked objects
- Use CURRENT_TIMESTAMP for current time, CURRENT_DATE for today
- Use INTERVAL for date math: CURRENT_DATE - INTERVAL '7 days'
- Use EXTRACT(DOW FROM timestamp) for day of week (0=Sunday, 6=Saturday)
- Use DATE_TRUNC('day', timestamp) to group by date
- Use TO_CHAR(timestamp, 'Day') to get day name
"""


_SQL_GENERATION_MAJORITY_VOTE_NOTE = """\
NOTES:
- Make sure majority vote is used for each track_id.
"""


def _build_query_patterns(
    classes_info: list[dict] | None,
    app_config_id: int | None,
) -> str:
    """Build worked SQL examples tailored to the current detection classes."""
    if not classes_info or app_config_id is None:
        return ""

    trackable, non_trackable = pick_example_class(classes_info)
    if not trackable:
        return ""

    ppe_attrs = [c["name"].lower() for c in classes_info if not c["trackable"]]

    parts = ["\nQUERY PATTERNS (adapt these, don't copy blindly):\n"]

    cfg = app_config_id
    trk_name = trackable["name"]

    if ppe_attrs:
        attr = ppe_attrs[0]
        parts.append(
            f"-- Unique {trk_name} where majority of observations show no {attr} (last N minutes):\n"
            f"SELECT COUNT(*) AS violators\n"
            f"FROM (\n"
            f"  SELECT dt.track_id\n"
            f"  FROM detection_tracks dt\n"
            f"  JOIN detection_classes dc ON dc.id = dt.detection_classes_id\n"
            f"  JOIN detection_observations do_ ON do_.track_id = dt.track_id\n"
            f"  WHERE dc.app_config_id = {cfg}\n"
            f"    AND dc.name = '{trk_name}'\n"
            f"    AND do_.timestamp >= NOW() - INTERVAL '10 minutes'\n"
            f"  GROUP BY dt.track_id\n"
            f"  HAVING COUNT(*) FILTER (WHERE (do_.attributes->>'{attr}')::boolean = false)\n"
            f"       > COUNT(*) FILTER (WHERE (do_.attributes->>'{attr}')::boolean = true)\n"
            f") sub\n\n"
        )

    if ppe_attrs:
        attr = ppe_attrs[0]
        parts.append(
            f"-- Total unique {trk_name} with {attr} violation today (majority vote):\n"
            f"SELECT COUNT(*) AS violators\n"
            f"FROM (\n"
            f"  SELECT dt.track_id\n"
            f"  FROM detection_tracks dt\n"
            f"  JOIN detection_classes dc ON dc.id = dt.detection_classes_id\n"
            f"  JOIN detection_observations do_ ON do_.track_id = dt.track_id\n"
            f"  WHERE dc.app_config_id = {cfg}\n"
            f"    AND dc.name = '{trk_name}'\n"
            f"    AND dt.first_seen >= CURRENT_DATE\n"
            f"  GROUP BY dt.track_id\n"
            f"  HAVING COUNT(*) FILTER (WHERE (do_.attributes->>'{attr}')::boolean = false)\n"
            f"       > COUNT(*) FILTER (WHERE (do_.attributes->>'{attr}')::boolean = true)\n"
            f") sub\n\n"
        )
    else:
        parts.append(
            f"-- Total unique {trk_name} detected today:\n"
            f"SELECT COUNT(DISTINCT dt.track_id)\n"
            f"FROM detection_tracks dt\n"
            f"JOIN detection_classes dc ON dc.id = dt.detection_classes_id\n"
            f"WHERE dc.app_config_id = {cfg}\n"
            f"  AND dc.name = '{trk_name}'\n"
            f"  AND dt.first_seen >= CURRENT_DATE\n\n"
        )

    if len(ppe_attrs) >= 2:
        attr2 = ppe_attrs[1]
        parts.append(
            f"-- Per-{trk_name} classification by majority vote (last N minutes):\n"
            f"SELECT dt.track_id,\n"
            f"       CASE WHEN COUNT(*) FILTER (WHERE (do_.attributes->>'{attr}')::boolean = false)\n"
            f"                 > COUNT(*) FILTER (WHERE (do_.attributes->>'{attr}')::boolean = true)\n"
            f"            THEN true ELSE false END AS {attr}_violation,\n"
            f"       CASE WHEN COUNT(*) FILTER (WHERE (do_.attributes->>'{attr2}')::boolean = false)\n"
            f"                 > COUNT(*) FILTER (WHERE (do_.attributes->>'{attr2}')::boolean = true)\n"
            f"            THEN true ELSE false END AS {attr2}_violation\n"
            f"FROM detection_tracks dt\n"
            f"JOIN detection_classes dc ON dc.id = dt.detection_classes_id\n"
            f"JOIN detection_observations do_ ON do_.track_id = dt.track_id\n"
            f"WHERE dc.app_config_id = {cfg}\n"
            f"  AND dc.name = '{trk_name}'\n"
            f"  AND do_.timestamp >= NOW() - INTERVAL '10 minutes'\n"
            f"GROUP BY dt.track_id\n\n"
        )
    elif ppe_attrs:
        attr = ppe_attrs[0]
        parts.append(
            f"-- Per-{trk_name} classification by majority vote (last N minutes):\n"
            f"SELECT dt.track_id,\n"
            f"       CASE WHEN COUNT(*) FILTER (WHERE (do_.attributes->>'{attr}')::boolean = false)\n"
            f"                 > COUNT(*) FILTER (WHERE (do_.attributes->>'{attr}')::boolean = true)\n"
            f"            THEN true ELSE false END AS {attr}_violation\n"
            f"FROM detection_tracks dt\n"
            f"JOIN detection_classes dc ON dc.id = dt.detection_classes_id\n"
            f"JOIN detection_observations do_ ON do_.track_id = dt.track_id\n"
            f"WHERE dc.app_config_id = {cfg}\n"
            f"  AND dc.name = '{trk_name}'\n"
            f"  AND do_.timestamp >= NOW() - INTERVAL '10 minutes'\n"
            f"GROUP BY dt.track_id\n\n"
        )

    return "".join(parts)


def build_sql_generator_prompt(
    app_config_id: int | None = None,
    classes_info: list[dict] | None = None,
) -> str:
    """System prompt for the inner SQL-generation LLM.

    Contains the DB schema, SQL syntax rules, query patterns, and
    app_config constraints.  The LLM should return ONLY a raw SQL
    SELECT statement for the requested metric.
    """
    parts = [
        "You are a SQL query generator for a detection-monitoring database.\n"
        "Given a metric description (and optionally a previous error), "
        "return ONLY a single raw SQL SELECT statement. No explanation.\n",
        _SQL_GENERATION_RULES,
        get_schema_description(),
        _SQL_GENERATION_NOTE,
    ]

    if app_config_id is not None:
        constraint = (
            f"\nIMPORTANT: The user is viewing app_config id={app_config_id}. "
            f"ALL SQL queries MUST join or filter through "
            f"Never query data from other configs.\n"
            f"When counting tracks via majority-vote (GROUP BY + HAVING), "
            f"ALWAYS wrap the grouped query in a subquery: SELECT COUNT(*) FROM (...) sub. "
            f"Never reference inner aliases in the outer SELECT. "
            f"In HAVING, always compare true_count vs false_count — never use > 0 or = 0 thresholds.\n"
        )
        compact = compact_classes(classes_info)
        if compact:
            constraint += f"Detection classes for this config:\n{compact}\n"
        parts.append(constraint)

        parts.append(_build_query_patterns(classes_info, app_config_id))
    else:
        logger.warning("No app_config_id provided for SQL generator")

    parts.append(_SQL_GENERATION_MAJORITY_VOTE_NOTE)
    return "".join(parts)
