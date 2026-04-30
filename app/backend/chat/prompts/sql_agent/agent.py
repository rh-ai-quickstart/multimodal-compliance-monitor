from __future__ import annotations

_AGENT_ORCHESTRATION_RULES = """\
TOOL USAGE RULES — read carefully:
1. For each metric, call query_metric with the metric description. The tool generates and executes the SQL internally.
2. If the tool returns an error, call query_metric again with the same metric AND the error in error_context so it can fix the query.
3. After all metrics are fetched, reply with EXACTLY one line per metric:
   <metric text>: <raw tool result>
   Nothing else — no intro, no summary.
"""


def build_sql_agent_prompt(
    app_config_id: int | None = None,
    classes_info: list[dict] | None = None,
) -> str:
    """Lightweight orchestration prompt for the SQL agent ReAct node.

    The agent uses query_metric (which internally generates and executes
    SQL) as its only tool.  Schema knowledge lives in the inner
    generator LLM, not here.
    """
    parts = [
        "You are a data-fetching orchestrator for a detection-monitoring database.\n"
        "You have one tool which you MUST to use for each metric:\n"
        "- query_metric: generates a SQL query for a metric and executes it, returning the result.\n",
        _AGENT_ORCHESTRATION_RULES,
    ]

    if app_config_id is not None:
        parts.append(f"\nThe user is viewing app_config id={app_config_id}.\n")

    return "".join(parts)
