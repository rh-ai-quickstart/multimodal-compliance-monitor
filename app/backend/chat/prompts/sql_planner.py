from __future__ import annotations

from chat.prompts._utils import compact_classes, english_join, pick_example_classes


def build_sql_planner_prompt(classes_info: list[dict] | None = None) -> str:
    parts = [
        "You are a query planner for a detection-monitoring database.\n\n"
        "Given a user question that requires historical data, decompose it into "
        "a list of discrete, measurable metrics that must be fetched from the database "
        "to fully answer the question.\n\n"
        "Each metric should be a short, specific data point description.\n\n"
        "When the user's question refers to multiple detection classes collectively "
        "without naming them individually, produce a separate metric for each "
        "relevant class from the known classes list below. "
        "If the user names a specific class, produce a metric for that class only. "
        "If the question only concerns trackable objects, do not expand to "
        "include non-trackable classes.\n\n"
    ]
    class_list = compact_classes(classes_info)
    if class_list:
        parts.append(f"Known classes:\n{class_list}\n\n")
    if classes_info:
        trackables, non_trackables = pick_example_classes(classes_info, 1, 4)

        parts.append("Examples:\n")
        if trackables and non_trackables:
            trackable = trackables[0]["name"]
            nt_names = [nt["name"] for nt in non_trackables]

            with_items = [f"a {n}" for n in nt_names[:-1]]
            without = nt_names[-1]

            if len(nt_names) >= 2:
                with_str = english_join(with_items)
                parts.append(
                    f'- Question: "In the last 2 hours, how many {trackable} were detected '
                    f'with {with_str} but without a {without}?"\n'
                    f'  Metrics: ["unique {trackable} with {with_str} '
                    f'but without a {without} in the last 2 hours"]\n\n'
                )
            else:
                parts.append(
                    f'- Question: "In the last 2 hours, how many {trackable} were detected '
                    f'without a {without}?"\n'
                    f'  Metrics: ["unique {trackable} without a {without} '
                    f'in the last 2 hours"]\n\n'
                )

            with_items = english_join([f"a {n}" for n in nt_names])
            parts.append(
                f'- Question: "how many {trackable} have {with_items} in the last 2 hours"\n'
                f'  Metrics: ["unique {trackable} with {with_items} '
                f'in the last 2 hours", unique {trackable} count in the last 2 hours"]\n\n'
            )

            parts.append(
                f'- Question: "How many {trackable} were detected in the last 4 minutes?"\n'
                f'  Metrics: ["unique {trackable} count in the last 4 minutes"]\n\n'
            )
        elif trackables:
            trackable = trackables[0]["name"]
            parts.append(
                f'- Question: "How many {trackable} were detected in the last hour?"\n'
                f'  Metrics: ["unique {trackable} count in the last hour"]\n\n'
            )
        elif non_trackables:
            nt = non_trackables[0]["name"]
            nt_names = [nt["name"] for nt in non_trackables]
            parts.append(
                f'- Question: "How many {nt} were detected in the last hour?"\n'
                f'  Metrics: ["unique {nt} count in the last hour"]\n\n'
            )
            with_items = english_join([f"a {n}" for n in nt_names])
            parts.append(
                f'- Question: "How many {with_items} in the last hour?"\n'
                f'  Metrics: ["unique {with_items} count in the last hour"]\n\n'
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
