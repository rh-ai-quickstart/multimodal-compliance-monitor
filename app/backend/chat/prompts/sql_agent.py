from database import get_schema_description


def build_sql_agent_prompt(
    app_config_id: int | None = None,
    classes_info: list[dict] | None = None,
    metrics: list[str] | None = None,
) -> str:
    """Build the full system prompt for the SQL agent node.

    Combines the DB schema, app_config_id constraint, and the metric
    checklist produced by the planner.
    """
    parts = [
        "You are a data-fetching agent for a detection-monitoring database.\n",
        "Your ONLY job is to execute SQL queries and return the raw results. "
        "Do NOT interpret or summarise — just fetch the data.\n\n",
        get_schema_description(),
    ]

    if app_config_id is not None:
        constraint = (
            f"\nIMPORTANT: The user is viewing app_config id={app_config_id}. "
            f"ALL SQL queries MUST join or filter through "
            f"detection_classes.app_config_id = {app_config_id}. "
            f"Never query data from other configs.\n"
        )
        if classes_info:
            class_lines = ", ".join(
                f"{c['name']} (trackable={c['trackable']})" for c in classes_info
            )
            constraint += f"Detection classes for this config: {class_lines}\n"
        parts.append(constraint)

    if metrics:
        checklist = "\n".join(f"  {i + 1}. {m}" for i, m in enumerate(metrics))
        parts.append(
            f"\nMetrics to fetch (query each one):\n{checklist}\n\n"
            "Return a short structured summary with each metric and its value."
        )

    return "".join(parts)
