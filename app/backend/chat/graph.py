from __future__ import annotations

import asyncio
import os
from typing import Generator

from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from chat.nodes import (
    make_clarifier_node,
    make_context_answer_node,
    make_router_node,
    make_sql_agent_node,
    make_sql_answer_node,
    make_sql_planner_node,
)
from chat.state import ChatState
from logger import get_logger
from tools.mcp_tools import current_app_config_id, load_sql_tool_only

log = get_logger(__name__)


def _route_after_router(state: ChatState) -> str:
    return state["route"]


def _build_graph(llm: ChatOpenAI, sql_tools: list) -> StateGraph:
    graph = StateGraph(ChatState)

    graph.add_node("clarifier", make_clarifier_node(llm))
    graph.add_node("router", make_router_node(llm))
    graph.add_node("context_answer", make_context_answer_node(llm))
    graph.add_node("sql_planner", make_sql_planner_node(llm))
    graph.add_node("sql_agent", make_sql_agent_node(llm, sql_tools))
    graph.add_node("sql_answer", make_sql_answer_node(llm))

    graph.add_edge(START, "clarifier")
    graph.add_edge("clarifier", "router")
    graph.add_conditional_edges(
        "router",
        _route_after_router,
        {"context": "context_answer", "sql": "sql_planner"},
    )
    graph.add_edge("sql_planner", "sql_agent")
    graph.add_edge("sql_agent", "sql_answer")
    graph.add_edge("context_answer", END)
    graph.add_edge("sql_answer", END)

    return graph


class LLMChat:
    """Conversational LLM backed by a LangGraph router pipeline.

    Routes present-tense questions to a context-only answerer, and
    historical questions through a SQL planner -> agent -> answer chain.

    Maintains per-session chat history so the model sees the full conversation.
    """

    def __init__(self) -> None:
        endpoint = os.environ["OPENAI_API_ENDPOINT"]
        api_key = os.environ["OPENAI_API_TOKEN"]
        model = os.getenv("OPENAI_MODEL", "llama-4-scout-17b-16e-w4a16")
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))

        llm = ChatOpenAI(
            base_url=endpoint,
            api_key=api_key,
            model=model,
            temperature=temperature,
            streaming=True,
        )

        sql_tools = asyncio.run(load_sql_tool_only())

        self._memory = MemorySaver()
        graph = _build_graph(llm, sql_tools)
        self._app = graph.compile(checkpointer=self._memory)
        self._session_versions: dict[str, int] = {}

        log.info(
            "LLMChat initialised — endpoint=%s, model=%s, sql_tools=%d",
            endpoint,
            model,
            len(sql_tools),
        )

    def _thread_id(self, session_id: str) -> str:
        version = self._session_versions.get(session_id, 0)
        return f"{session_id}:{version}" if version else session_id

    def _build_input(
        self,
        question: str,
        context: str,
        app_config_id: int | None = None,
        classes_info: list[dict] | None = None,
    ) -> ChatState:
        return {
            "messages": [HumanMessage(content=question)],
            "question": question,
            "context": context,
            "route": "",
            "app_config_id": app_config_id,
            "classes_info": classes_info,
            "metrics": [],
            "sql_result": "",
        }

    def chat(
        self,
        question: str,
        context: str,
        session_id: str = "default",
        app_config_id: int | None = None,
        classes_info: list[dict] | None = None,
    ) -> str:
        """Send a question through the router pipeline.

        Every prior exchange in *session_id* is automatically included so the
        model can reference earlier questions and answers.
        """
        log.info(
            "chat called: question=%r, session_id=%r, app_config_id=%r, context_len=%d, context=%r",
            question,
            session_id,
            app_config_id,
            len(context) if context else 0,
            context,
        )

        token = current_app_config_id.set(app_config_id)
        try:
            inp = self._build_input(question, context, app_config_id, classes_info)
            response = asyncio.run(
                self._app.ainvoke(
                    inp,
                    config={"configurable": {"thread_id": self._thread_id(session_id)}},
                )
            )
            return response["messages"][-1].content
        finally:
            current_app_config_id.reset(token)

    def stream_question(
        self,
        question: str,
        context: str,
        session_id: str = "default",
        app_config_id: int | None = None,
        classes_info: list[dict] | None = None,
    ) -> Generator[str, None, None]:
        """Stream answer tokens one chunk at a time.

        Conversation history is updated automatically once the full stream
        has been consumed.
        """

        async def _astream():
            chunks = []
            async for msg, _metadata in self._app.astream(
                self._build_input(question, context, app_config_id, classes_info),
                config={"configurable": {"thread_id": self._thread_id(session_id)}},
                stream_mode="messages",
            ):
                if (
                    isinstance(msg, AIMessageChunk)
                    and msg.content
                    and not msg.tool_calls
                ):
                    chunks.append(msg.content)
            return chunks

        token = current_app_config_id.set(app_config_id)
        try:
            for chunk in asyncio.run(_astream()):
                yield chunk
        finally:
            current_app_config_id.reset(token)

    def clear_history(self, session_id: str = "default") -> None:
        """Clear the conversation history for a session."""
        self._session_versions[session_id] = (
            self._session_versions.get(session_id, 0) + 1
        )
