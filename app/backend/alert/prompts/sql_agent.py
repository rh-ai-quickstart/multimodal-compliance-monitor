from __future__ import annotations

from chat.prompts.sql_agent.agent import build_sql_agent_prompt


def build_alert_sql_agent_prompt(
    app_config_id: int | None = None,
    classes_info: list[dict] | None = None,
) -> str:
    """Orchestration prompt for the alert SQL agent node.

    Delegates to the shared chat sql_agent prompt since the orchestration
    logic is identical -- the agent uses query_metric as its only tool.
    """
    return build_sql_agent_prompt(
        app_config_id=app_config_id,
        classes_info=classes_info,
    )
