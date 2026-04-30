from chat.prompts._utils import compact_classes


_CLARIFIER_BASE = (
    "You are a question rewriter for a monitoring assistant.\n\n"
    "Given the conversation history and the user's latest message, produce a single "
    "self-contained question that captures the user's intent without relying on "
    "prior context.\n\n"
    "Rules:\n"
    "- Resolve all pronouns and references ('it', 'that', 'the same', 'those') "
    "using the conversation history.\n"
    "- If the latest message is already clear and standalone, return it unchanged.\n"
    "- When the user mentions a detection object, map it to the closest matching "
    "class name from the list below. Use the exact class name in the rewritten "
    "question.\n"
    "- Preserve the original phrasing as much as possible — only add context "
    "needed to make the question standalone.\n"
    "- Output ONLY the clarified question. No explanation, no preamble."
)


def build_clarifier_prompt(classes_info: list[dict] | str | None = None) -> str:
    compact = compact_classes(classes_info)
    if compact:
        return f"{_CLARIFIER_BASE}\n\nKnown detection classes:\n{compact}"
    return _CLARIFIER_BASE
