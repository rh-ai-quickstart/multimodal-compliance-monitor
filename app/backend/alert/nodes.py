from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool, tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from pydantic import BaseModel, Field

from alert.prompts import (
    build_alert_clarifier_planner_prompt,
    build_alert_sql_agent_prompt,
    build_alert_sql_generator_prompt,
)
from alert.state import AlertState
from logger import get_logger

log = get_logger(__name__)


class AlertMetric(BaseModel):
    metric: str = Field(
        description="A single metric describing the SQL query needed for this alert"
    )


def make_alert_clarifier_planner_node(llm: ChatOpenAI):
    structured_llm = llm.with_structured_output(AlertMetric)

    async def alert_clarifier_planner_node(state: AlertState) -> dict:
        classes_info = state.get("classes_info")
        prompt = build_alert_clarifier_planner_prompt(classes_info)
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=state["alert_text"]),
        ]
        result: AlertMetric = await structured_llm.ainvoke(messages)
        log.info(
            "Alert clarifier+planner: %r -> %r", state["alert_text"], result.metric
        )
        return {"metric": result.metric}

    return alert_clarifier_planner_node


def make_alert_sql_agent_node(llm: ChatOpenAI, execute_sql_tool: StructuredTool):
    async def alert_sql_agent_node(state: AlertState) -> dict:
        app_config_id = state.get("app_config_id")
        classes_info = state.get("classes_info")
        metric = state["metric"]

        generated_sql: str | None = None

        @tool
        async def query_metric(metric_desc: str, error_context: str = "") -> str:
            """Generate and execute a SQL query for the given metric.

            Args:
                metric_desc: The metric description to query.
                error_context: Optional error from a previous attempt so the
                    query can be corrected.
            """
            nonlocal generated_sql
            generator_prompt = build_alert_sql_generator_prompt(
                metric=metric_desc,
                app_config_id=app_config_id,
                classes_info=classes_info,
            )
            human_content = metric_desc
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
            generated_sql = sql
            log.info("Alert SQL generator for metric %r: %s", metric_desc, sql)

            result = await execute_sql_tool.ainvoke({"sql": sql})
            return result

        agent_prompt = build_alert_sql_agent_prompt(
            app_config_id=app_config_id,
            classes_info=classes_info,
        )
        agent = create_agent(llm, [query_metric], system_prompt=agent_prompt)

        await agent.ainvoke(
            {"messages": [HumanMessage(content=f"1. {metric}")]},
        )
        log.info("Alert SQL agent completed, captured SQL: %s", generated_sql)
        return {"sql_query": generated_sql or ""}

    return alert_sql_agent_node
