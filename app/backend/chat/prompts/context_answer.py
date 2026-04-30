from __future__ import annotations

from chat.prompts._utils import english_join, pick_example_classes

_PREAMBLE = (
    "You are a terse monitoring assistant. You receive three inputs:\n"
    "1. The original user question.\n"
    "2. The context (what the users see right now) yo must use to answer the question.\n"
    "Your job is to synthesize a clean, user-facing answer by combining the context (what the users see right now) of the original question.\n\n"
    "Response rules:\n"
    "- For yes/no questions, answer directly then support with numbers.\n"
    "- No greetings or filler.\n"
    "- 1-3 short sentences max.\n"
    "- Present the information as if you observed it directly."
)

_TOPICS = [
    "Are all the {trackable} detected with a {nt}?",
    "How many {trackable} have a {nt}?",
    "Are there any {trackable} without a {nt}?",
    "How many {trackable} are currently detected?",
    "What is the {nt} count?",
]


def _build_context(trackable: str, selected_nts: list[str], total: int) -> str:
    """Build a numbered context string from sampled classes and counts."""
    parts: list[str] = []
    idx = 1
    for j, name in enumerate(selected_nts):
        has = j + 1
        missing = total - has
        parts.append(f"{idx}. {name}: {has}")
        idx += 1
        parts.append(f"{idx}. NO-{name}: {missing}")
        idx += 1
    parts.append(f"{idx}. {trackable}: {total}")
    return ", ".join(parts)


def _build_answer(
    trackable: str,
    selected_nts: list[str],
    total: int,
    topic_idx: int,
) -> str:
    """Build the expected answer for the given topic."""
    nt = selected_nts[topic_idx % len(selected_nts)]

    if topic_idx == 0:
        nt_summary = english_join(
            [
                f"{j + 1} with {n} ({total - j - 1} without)"
                for j, n in enumerate(selected_nts)
            ]
        )
        return (
            f"No, not all {trackable} have a {nt}. "
            f"There are {total} {trackable}. {nt_summary}."
        )

    if topic_idx == 1:
        has = 2
        missing = total - has
        return f"{has} out of {total} {trackable} have a {nt}. {missing} do not."

    if topic_idx == 2:
        has = 1
        missing = total - has
        return f"Yes, {has} out of {total} {trackable} do not have a {nt}."

    if topic_idx == 3:
        return f"There are {total} {trackable} currently detected."

    has = 2
    missing = total - has
    return f"{nt} count is {has}. {missing} {trackable} do not have a {nt}."


def build_context_answer_prompt(classes_info: list[dict] | None = None) -> str:
    if not classes_info:
        return _PREAMBLE

    trackables, non_trackables = pick_example_classes(classes_info, 1, 4)

    if not trackables or not non_trackables:
        return _PREAMBLE

    trackable_name = trackables[0]["name"]
    nt_names = [nt["name"] for nt in non_trackables]
    total = 6

    examples: list[str] = []
    for i in range(3):
        n_sample = min(2 + i, len(nt_names))
        selected = nt_names[:n_sample]

        topic_idx = i % len(_TOPICS)
        nt_for_question = selected[i % len(selected)]

        context_str = _build_context(trackable_name, selected, total)
        question = _TOPICS[topic_idx].format(
            trackable=trackable_name, nt=nt_for_question
        )
        answer = _build_answer(trackable_name, selected, total, topic_idx)

        examples.append(f"Example:\nContext: {context_str}\nQ: {question}\nA: {answer}")

    return "\n\n".join([_PREAMBLE, "---", *examples])
