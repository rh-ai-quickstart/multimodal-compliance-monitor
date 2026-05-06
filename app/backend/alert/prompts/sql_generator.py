from __future__ import annotations

from chat.prompts.sql_agent.generator import build_sql_generator_prompt


def build_alert_sql_generator_prompt(
    metric: str = "",
    app_config_id: int | None = None,
    classes_info: list[dict] | None = None,
) -> str:
    """SQL generation prompt for the alert pipeline.

    Delegates to the shared chat sql_generator prompt since the schema,
    rules, and query patterns are identical.
    """
    return build_sql_generator_prompt(
        metric=metric,
        app_config_id=app_config_id,
        classes_info=classes_info,
    )
