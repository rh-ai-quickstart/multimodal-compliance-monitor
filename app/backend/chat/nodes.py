from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from chat.prompts import (
    build_clarifier_prompt,
    build_context_answer_prompt,
    build_router_prompt,
    build_sql_answer_prompt,
    build_sql_planner_prompt,
    build_sql_agent_prompt,
    build_sql_generator_prompt,
)
from chat.state import ChatState
from logger import get_logger

log = get_logger(__name__)


# --------------- Structured-output schemas ---------------


class RouteDecision(BaseModel):
    route: Literal["context", "sql"] = Field(
        description="'context' for present-tense visual questions, 'sql' for historical DB queries"
    )


class MetricsPlan(BaseModel):
    metrics: list[str] = Field(
        description="Discrete data points to fetch from the database"
    )


# --------------- Node factories ---------------
# Each ``make_*`` function closes over the shared *llm* (and tools where
# needed) and returns an async node callable compatible with StateGraph.


def make_clarifier_node(llm: ChatOpenAI):
    async def clarifier_node(state: ChatState) -> dict:
        history = state.get("messages", [])
        raw_question = state["question"]
        classes_info = state.get("classes_info")

        if not history or len(history) <= 1:
            log.info("Clarifier: no history, passing question through unchanged")
            return {"question": raw_question}

        prompt = build_clarifier_prompt(classes_info)
        messages = [
            SystemMessage(content=prompt),
            *history,
            HumanMessage(content=raw_question),
        ]
        response = await llm.ainvoke(messages)
        clarified = response.content.strip()
        log.info(
            "Clarifier: %r -> %r",
            raw_question,
            clarified,
        )
        return {"question": clarified}

    return clarifier_node


def make_router_node(llm: ChatOpenAI):
    structured_llm = llm.with_structured_output(RouteDecision)

    async def router_node(state: ChatState) -> dict:
        prompt = build_router_prompt(state.get("classes_info"))
        messages = [
            SystemMessage(content=prompt),
            SystemMessage(content=f"Current visual context:\n{state['context']}"),
            HumanMessage(content=state["question"]),
        ]
        decision: RouteDecision = await structured_llm.ainvoke(messages)
        log.info(
            "Router decided: %s for question=%r", decision.route, state["question"]
        )
        return {"route": decision.route}

    return router_node


def make_context_answer_node(llm: ChatOpenAI):
    async def context_answer_node(state: ChatState) -> dict:
        prompt = build_context_answer_prompt(state.get("classes_info"))
        messages = [
            SystemMessage(content=prompt),
            SystemMessage(content=f"The user sees right now:\n{state['context']}"),
            HumanMessage(content=f"User question: {state['question']}"),
        ]
        response = await llm.ainvoke(messages)
        return {"messages": [response]}

    return context_answer_node


def make_sql_planner_node(llm: ChatOpenAI):
    structured_llm = llm.with_structured_output(MetricsPlan)

    async def sql_planner_node(state: ChatState) -> dict:
        prompt = build_sql_planner_prompt(state.get("classes_info"))
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=state["question"]),
        ]
        plan: MetricsPlan = await structured_llm.ainvoke(messages)
        log.info("SQL planner produced %d metrics: %s", len(plan.metrics), plan.metrics)
        return {"metrics": plan.metrics}

    return sql_planner_node


def make_sql_agent_node(llm: ChatOpenAI, execute_sql_tool: StructuredTool):
    async def sql_agent_node(state: ChatState) -> dict:
        app_config_id = state.get("app_config_id")
        classes_info = state.get("classes_info")

        generator_prompt = build_sql_generator_prompt(
            app_config_id=app_config_id,
            classes_info=classes_info,
        )

        @tool
        async def query_metric(metric: str, error_context: str = "") -> str:
            """Generate and execute a SQL query for the given metric.

            Args:
                metric: The metric description to query.
                error_context: Optional error from a previous attempt so the
                    query can be corrected.
            """
            human_content = metric
            if error_context:
                human_content += (
                    f"\n\nPrevious attempt failed with this error:\n"
                    f"{error_context}\nFix the query."
                )
            msgs = [
                SystemMessage(content=generator_prompt),
                HumanMessage(content=human_content),
            ]
            resp = await llm.ainvoke(msgs)
            sql = resp.content.strip()
            log.info("Generated SQL for metric %r: %s", metric, sql)

            result = await execute_sql_tool.ainvoke({"sql": sql})
            return result

        agent_prompt = build_sql_agent_prompt(
            app_config_id=app_config_id,
            classes_info=classes_info,
        )
        agent = create_agent(llm, [query_metric], system_prompt=agent_prompt)

        metrics = state.get("metrics", [])
        metrics_msg = "\n".join(f"{i + 1}. {m}" for i, m in enumerate(metrics))
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=metrics_msg)]},
        )
        last_msg = result["messages"][-1]
        sql_result = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        log.info("SQL agent result length: %d chars", len(sql_result))
        return {"sql_result": sql_result}

    return sql_agent_node


def make_sql_answer_node(llm: ChatOpenAI):
    async def sql_answer_node(state: ChatState) -> dict:
        prompt = build_sql_answer_prompt(state.get("classes_info"))
        metrics = state.get("metrics", [])
        metrics_text = "\n".join(f"{i + 1}. {m}" for i, m in enumerate(metrics))
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(
                content=(
                    f"User question: {state['question']}\n\n"
                    f"Planned metrics:\n{metrics_text}\n\n"
                    f"Raw database results:\n{state['sql_result']}"
                )
            ),
        ]
        response = await llm.ainvoke(messages)
        return {"messages": [response]}

    return sql_answer_node
