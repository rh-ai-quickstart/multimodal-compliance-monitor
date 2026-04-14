from __future__ import annotations

from chat.prompts._utils import pick_example_classes


def build_sql_planner_prompt(classes_info: list[dict] | None = None) -> str:
    parts = [
        "You are a query planner for a detection-monitoring database.\n\n"
        "Given a user question that requires historical data, decompose it into "
        "a list of discrete, measurable metrics that must be fetched from the database "
        "to fully answer the question.\n\n"
        "Each metric should be a short, specific data point description.\n\n"
    ]

    if classes_info:
        trackable, non_trackable = pick_example_classes(classes_info)

        parts.append("Examples:\n")
        if trackable and non_trackable:
            parts.append(
                f'- Question: "What\'s the {non_trackable["name"]} rate today?"\n'
                f'  Metrics: ["total unique {trackable["name"]} detected today", '
                f'"unique {trackable["name"]} without {non_trackable["name"]} today"]\n\n'
            )
        if trackable:
            parts.append(
                f'- Question: "How many {trackable["name"]} were detected in the last hour?"\n'
                f'  Metrics: ["unique {trackable["name"]} count in the last hour"]\n\n'
            )
    else:
        parts.append(
            "Examples:\n"
            '- Question: "What\'s the detection rate today?"\n'
            '  Metrics: ["total unique objects detected today"]\n\n'
            '- Question: "How many objects were detected in the last hour?"\n'
            '  Metrics: ["unique object count in the last hour"]\n\n'
        )

    parts.append(
        "Return only the list of metrics needed. Be exhaustive — include every "
        "data point required so no follow-up queries are needed."
    )

    return "".join(parts)
