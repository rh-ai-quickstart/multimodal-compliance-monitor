from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    context: str
    route: str
    app_config_id: int | None
    classes_info: list[dict] | None
    metrics: list[str]
    sql_result: str
