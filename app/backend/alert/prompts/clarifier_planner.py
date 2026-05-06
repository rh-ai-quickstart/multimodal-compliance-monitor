from __future__ import annotations

from chat.prompts._utils import compact_classes, english_join, pick_example_classes


def build_alert_clarifier_planner_prompt(
    classes_info: list[dict] | None = None,
) -> str:
    """System prompt for the merged clarifier+planner node.

    Instructs the LLM to:
    1. Map user terms to the exact known detection class names.
    2. Extract the alert intent (condition, time window, threshold).
    3. Output exactly ONE metric string that fully describes the SQL query
       needed to evaluate the alert.

    All examples are generated dynamically from *classes_info* so the prompt
    works for any dataset (PPE, Bird, etc.).
    """
    parts = [
        "You are an alert-rule interpreter for a detection-monitoring database.\n\n"
        "The user provides a plain-English alert rule. Your job is to:\n"
        "1. Map any informal names to the exact detection class names listed below.\n"
        "2. Identify the condition the alert checks (count threshold, presence/absence, "
        "time window, etc.).\n"
        "3. Produce exactly ONE metric string — a concise, self-contained description "
        "of the single SQL query needed to evaluate this alert.\n\n"
        "The metric must include:\n"
        "- Which class(es) to query\n"
        "- The time window (e.g. 'in the last 10 minutes')\n"
        "- The aggregation (e.g. 'count of unique ...', 'number of ...')\n"
        "- Any attribute conditions (e.g. 'without a Hardhat')\n\n"
        "Output ONLY the metric string. No explanation, no preamble.\n\n"
    ]

    class_list = compact_classes(classes_info)
    if class_list:
        parts.append(f"Known detection classes:\n{class_list}\n\n")

    if classes_info:
        trackables, non_trackables = pick_example_classes(classes_info, 1, 4)

        parts.append("Examples:\n")
        if trackables and non_trackables:
            trackable = trackables[0]["name"]
            nt_names = [nt["name"] for nt in non_trackables]

            if len(nt_names) >= 2:
                without = nt_names[0]
                parts.append(
                    f'- Alert: "Notify me if any {trackable.lower()} is seen without '
                    f'a {without.lower()} in the last 5 minutes"\n'
                    f'  Metric: "unique {trackable} without a {without} '
                    f'in the last 5 minutes"\n\n'
                )

                with_items = english_join([f"a {n}" for n in nt_names[:2]])
                parts.append(
                    f'- Alert: "Alert when a {trackable.lower()} has '
                    f'{with_items.lower()} in the last hour"\n'
                    f'  Metric: "unique {trackable} with {with_items} '
                    f'in the last hour"\n\n'
                )
            else:
                without = nt_names[0]
                parts.append(
                    f'- Alert: "Warn me if a {trackable.lower()} appears without '
                    f'a {without.lower()} in the last 10 minutes"\n'
                    f'  Metric: "unique {trackable} without a {without} '
                    f'in the last 10 minutes"\n\n'
                )

            parts.append(
                f'- Alert: "Alert if more than 5 {trackable.lower()} detected '
                f'in the last 30 minutes"\n'
                f'  Metric: "unique {trackable} count in the last 30 minutes"\n\n'
            )
        elif trackables:
            trackable = trackables[0]["name"]
            parts.append(
                f'- Alert: "Notify me if any {trackable.lower()} is detected '
                f'in the last 10 minutes"\n'
                f'  Metric: "unique {trackable} count in the last 10 minutes"\n\n'
            )
        elif non_trackables:
            nt = non_trackables[0]["name"]
            parts.append(
                f'- Alert: "Alert when {nt.lower()} count exceeds 3 '
                f'in the last hour"\n'
                f'  Metric: "unique {nt} count in the last hour"\n\n'
            )
    else:
        parts.append(
            "Examples:\n"
            '- Alert: "Notify me if more than 10 objects are detected in the '
            'last 30 minutes"\n'
            '  Metric: "total unique objects detected in the last 30 minutes"\n\n'
        )

    return "".join(parts)
