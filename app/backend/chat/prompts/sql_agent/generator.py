from __future__ import annotations

from chat.prompts._utils import compact_classes, extract_metric_attributes
from database import get_schema_description
from logger import get_logger

logger = get_logger(__name__)

_SQL_GENERATION_RULES = """\
SQL GENERATION RULES — read carefully:
1. Write a SINGLE plain SQL SELECT statement.
2. Never wrap SQL in markdown fences (```), quotes, or any extra text.
3. Return exactly ONE statement — no semicolons separating multiple statements.
4. Output ONLY the raw SQL, nothing else.
5. NEVER alias detection_observations as "do" — it is a PostgreSQL reserved keyword. Use "obs" instead.
"""

_SQL_GENERAL_NOTES = """\
NOTES:
- Use COUNT(DISTINCT track_id) to count unique tracked objects
- Use CURRENT_TIMESTAMP for current time, CURRENT_DATE for today
- Use INTERVAL for date math: CURRENT_DATE - INTERVAL '7 days'
- Time columns: detection_tracks has first_seen/last_seen; detection_observations has timestamp. No other tables have time columns.
- For time filters on tracks, use dt.first_seen or dt.last_seen — NEVER dt.timestamp (that column does not exist).
- Use EXTRACT(DOW FROM obs.timestamp) for day of week (0=Sunday, 6=Saturday)
- DATE_TRUNC('day', obs.timestamp) to group by date
- TO_CHAR(obs.timestamp, 'Day') to get day name
"""

_SQL_MAJORITY_VOTE_NOTES = """\
MAJORITY-VOTE NOTES (apply when classifying tracks by observation attributes):
- Each track has many noisy observations. Classify a track by majority vote, not by any single observation.
- Use GROUP BY track_id with COUNT(*) FILTER (WHERE (attributes->>'<attribute_name>')::boolean = false/true) to tally per-track.
- Use HAVING with majority vote in BOTH directions (false > true AND true > false) — never shortcut either with COUNT(DISTINCT) + a WHERE filter.
- Observations where the attribute key is absent are naturally ignored by FILTER (NULL fails the boolean cast). Use the exact JSONB key from the schema — never infer or rename keys from class display names.
- When counting tracks via majority-vote (GROUP BY + HAVING), ALWAYS wrap the grouped query in a subquery: SELECT COUNT(*) FROM (...) sub. Never reference inner aliases in the outer SELECT.
- In HAVING, always compare true_count vs false_count — never use > 0 or = 0 thresholds.
"""


def _build_query_patterns(
    trackable_names: list[str],
    non_trackable_names: list[str],
    classes_info: list[dict] | None,
    app_config_id: int | None,
) -> str:
    """Build worked SQL examples tailored to classes mentioned in *metric*."""
    if not classes_info or app_config_id is None:
        logger.warning("No classes_info or app_config_id provided for SQL generator")
        return ""
    non_trackable_names = [
        n
        for n in non_trackable_names
        if not (n.startswith("NO-") and n.removeprefix("NO-") in non_trackable_names)
    ]
    if not trackable_names:
        logger.warning("No trackable_names provided for SQL generator")
        return ""
    logger.info(f"non_trackable_names: {non_trackable_names}")
    trackable_attr = trackable_names[0]
    non_trackable_attr = non_trackable_names[0] if non_trackable_names else ""
    non_trackable_attr_2 = (
        non_trackable_names[1] if len(non_trackable_names) >= 2 else ""
    )

    parts = ["\nQUERY PATTERNS (adapt these, don't copy blindly):\n"]

    cfg = app_config_id

    # example of trackble only
    parts.append(
        f"-- Total unique {trackable_attr} detected in the last 10 hours:\n"
        f"SELECT COUNT(DISTINCT dt.track_id)\n"
        f"FROM detection_tracks dt\n"
        f"JOIN detection_classes dc ON dc.id = dt.detection_classes_id\n"
        f"WHERE dc.app_config_id = {cfg}\n"
        f"  AND dc.name = '{trackable_attr}'\n"
        f"  AND dt.first_seen >= NOW() - INTERVAL '10 hours'\n\n"
    )
    # example of trackable with a non-trackable attribute
    if non_trackable_names:
        parts.append(
            f"-- Unique {trackable_attr} where observations show no {non_trackable_attr} in the last 10 minutes:\n"
            f"SELECT COUNT(*) AS violators\n"
            f"FROM (\n"
            f"  SELECT dt.track_id\n"
            f"  FROM detection_tracks dt\n"
            f"  JOIN detection_classes dc ON dc.id = dt.detection_classes_id\n"
            f"  JOIN detection_observations obs ON obs.track_id = dt.track_id\n"
            f"  WHERE dc.app_config_id = {cfg}\n"
            f"    AND dc.name = '{trackable_attr}'\n"
            f"    AND obs.timestamp >= NOW() - INTERVAL '10 minutes'\n"
            f"  GROUP BY dt.track_id\n"
            f"  HAVING COUNT(*) FILTER (WHERE (obs.attributes->>'{non_trackable_attr.lower()}')::boolean = false)\n"
            f"       > COUNT(*) FILTER (WHERE (obs.attributes->>'{non_trackable_attr.lower()}')::boolean = true)\n"
            f") sub\n\n"
        )
    # example of trackable with two non-trackable attributes
    if non_trackable_attr_2:
        parts.append(
            f"-- unique {trackable_attr} with a {non_trackable_attr} and a {non_trackable_attr_2} in the last 2 days:\n"
            f"SELECT COUNT(*) AS compliant_count\n"
            f"FROM (\n"
            f"  SELECT dt.track_id\n"
            f"  FROM detection_tracks dt\n"
            f"  JOIN detection_classes dc ON dc.id = dt.detection_classes_id\n"
            f"  JOIN detection_observations obs ON obs.track_id = dt.track_id\n"
            f"  WHERE dc.app_config_id = {cfg}\n"
            f"    AND dc.name = '{trackable_attr}'\n"
            f"    AND obs.timestamp >= NOW() - INTERVAL '2 days'\n"
            f"  GROUP BY dt.track_id\n"
            f"  HAVING COUNT(*) FILTER (WHERE (obs.attributes->>'{non_trackable_attr.lower()}')::boolean = true)\n"
            f"       > COUNT(*) FILTER (WHERE (obs.attributes->>'{non_trackable_attr.lower()}')::boolean = false)\n"
            f"     AND COUNT(*) FILTER (WHERE (obs.attributes->>'{non_trackable_attr_2.lower()}')::boolean = true)\n"
            f"       > COUNT(*) FILTER (WHERE (obs.attributes->>'{non_trackable_attr_2.lower()}')::boolean = false)\n"
            f") sub\n\n"
        )

    return "".join(parts)


def build_sql_generator_prompt(
    metric: str = "",
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
        _SQL_GENERAL_NOTES,
    ]

    if app_config_id is not None:
        constraint = (
            f"\nIMPORTANT: The user is viewing app_config id={app_config_id}. "
            f"ALL SQL queries MUST filter on dc.app_config_id = {app_config_id}. "
            f"Never query data from other configs.\n"
        )
        trackable_names, non_trackable_names = extract_metric_attributes(
            metric, classes_info or []
        )
        if non_trackable_names:
            parts.append(_SQL_MAJORITY_VOTE_NOTES)

        compact = compact_classes(classes_info)
        if compact:
            constraint += f"Detection classes for this config:\n{compact}\n"
        parts.append(constraint)

        parts.append(
            _build_query_patterns(
                trackable_names, non_trackable_names, classes_info, app_config_id
            )
        )
    else:
        logger.warning("No app_config_id provided for SQL generator")

    return "".join(parts)
