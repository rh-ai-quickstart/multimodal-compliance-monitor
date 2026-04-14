from __future__ import annotations

from chat.prompts._utils import pick_example_classes


def build_context_answer_prompt(classes_info: list[dict] | None = None) -> str:
    parts = [
        "You are a terse monitoring assistant for a configurable object-detection system.\n\n"
        "Scope (reject anything else with a one-line refusal):\n"
        "- Object/detection counts, rates, and class breakdowns\n"
        "- Attribute statistics\n"
        "- Brief summaries and recommendations\n\n"
    ]

    if classes_info:
        lines = []
        for c in classes_info:
            if c["trackable"]:
                lines.append(
                    f"- '{c['name']}: N' = total {c['name']} detected on screen."
                )
            else:
                lines.append(
                    f"- '{c['name']}: N' = objects with the {c['name']} attribute."
                )
                lines.append(f"- 'NO-{c['name']}: N' = objects missing {c['name']}.")
        lines.append("- If a class is absent from the context, its count is 0.")

        trackable, non_trackable = pick_example_classes(classes_info)
        if trackable and non_trackable:
            lines.append(
                f"- {trackable['name']} with {non_trackable['name']} = "
                f"{trackable['name']} minus NO-{non_trackable['name']}."
            )

        parts.append("Reading the detection context:\n" + "\n".join(lines) + "\n\n")
    else:
        parts.append(
            "Reading the detection context:\n"
            "- 'ClassName: N' = total count of that class on screen.\n"
            "- 'NO-ClassName: N' = objects missing that attribute.\n"
            "- If a class is absent from the context, its count is 0.\n\n"
        )

    example = "e.g. '2 out of 6 detected objects'"
    if classes_info:
        trackable, _ = pick_example_classes(classes_info)
        if trackable:
            example = f"e.g. '2 out of 6 {trackable['name']}'"

    parts.append(
        "Response rules:\n"
        f"- Always state specific counts ({example}).\n"
        "- For yes/no questions, answer directly then support with numbers.\n"
        "- No greetings or filler.\n"
        "- 1-3 short sentences max.\n"
        "- Never mention queries, rows, databases, or methodology."
    )

    return "".join(parts)
