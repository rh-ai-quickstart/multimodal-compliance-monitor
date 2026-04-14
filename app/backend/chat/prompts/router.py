from __future__ import annotations

from chat.prompts._utils import pick_example_classes


def build_router_prompt(classes_info: list[dict] | None = None) -> str:
    context_examples = "'What do you see?'"
    sql_examples = "'Show detection counts for the past week'"

    if classes_info:
        trackable, non_trackable = pick_example_classes(classes_info)
        if trackable:
            context_examples = f"'How many {trackable['name']} on screen?'"
        if non_trackable:
            sql_examples = f"'What was the {non_trackable['name']} rate yesterday?'"

    return (
        "You are a question classifier for a real-time object-detection monitoring system.\n\n"
        "Given the user's question and the current visual context (what the camera sees right now), "
        "decide which path can answer it:\n\n"
        "Route to 'context' when:\n"
        "- The question is about what is visible RIGHT NOW (present tense)\n"
        f"- Examples: {context_examples}\n"
        "- The visual context alone is sufficient to answer\n\n"
        "Route to 'sql' when:\n"
        "- The question asks about historical data, trends, or time ranges\n"
        f"- Examples: {sql_examples}\n"
        "- The question requires querying stored detection records\n\n"
        "Respond with exactly one route."
    )
