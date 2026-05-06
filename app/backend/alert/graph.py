from __future__ import annotations

import asyncio
import os

from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from alert.nodes import (
    make_alert_clarifier_planner_node,
    make_alert_sql_agent_node,
)
from alert.state import AlertState
from logger import get_logger
from tools.mcp_tools import current_app_config_id, load_execute_sql_tool

log = get_logger(__name__)


def _build_alert_graph(
    llm: ChatOpenAI,
    execute_sql_tool: StructuredTool,
) -> StateGraph:
    graph = StateGraph(AlertState)

    graph.add_node("clarifier_planner", make_alert_clarifier_planner_node(llm))
    graph.add_node("sql_agent", make_alert_sql_agent_node(llm, execute_sql_tool))

    graph.add_edge(START, "clarifier_planner")
    graph.add_edge("clarifier_planner", "sql_agent")
    graph.add_edge("sql_agent", END)

    return graph


class LLMAlert:
    """Converts plain-English alert rules into validated SQL queries.

    Uses a two-node LangGraph pipeline:
    clarifier_planner -> sql_agent
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

        execute_sql_tool = asyncio.run(load_execute_sql_tool())

        graph = _build_alert_graph(llm, execute_sql_tool)
        self._app = graph.compile()

        log.info(
            "LLMAlert initialised — endpoint=%s, model=%s",
            endpoint,
            model,
        )

    def create_alert(
        self,
        alert_text: str,
        app_config_id: int,
        classes_info: list[dict] | None = None,
    ) -> str:
        """Run the alert pipeline to convert *alert_text* into a SQL query.

        Returns the generated SQL query string.
        """
        log.info(
            "create_alert called: app_config_id=%d, text=%r",
            app_config_id,
            alert_text,
        )

        token = current_app_config_id.set(app_config_id)
        try:
            inp: AlertState = {
                "alert_text": alert_text,
                "app_config_id": app_config_id,
                "classes_info": classes_info,
                "metric": "",
                "sql_query": "",
            }
            result = asyncio.run(self._app.ainvoke(inp))
            return result["sql_query"]
        except Exception as e:
            log.exception("create_alert failed: %s", e)
            raise
        finally:
            current_app_config_id.reset(token)
