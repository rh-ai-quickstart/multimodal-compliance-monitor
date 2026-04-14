"""Postgres MCP tool loader.

Connects to a postgres-mcp server (read-only / restricted mode) over SSE
and exposes its tools as LangChain tools for the ReAct agent.

When ``current_app_config_id`` is set (per-request via contextvars), the
``execute_sql`` tool is wrapped to **reject** any query that touches
detection tables without filtering by the active ``app_config_id``.
"""

import contextvars
import os

from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from logger import get_logger

log = get_logger(__name__)

POSTGRES_MCP_URL = os.getenv("POSTGRES_MCP_URL")

_mcp_client = MultiServerMCPClient(
    {
        "postgres": {
            "url": POSTGRES_MCP_URL,
            "transport": "sse",
        }
    }
)

current_app_config_id: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "current_app_config_id", default=None
)

_SCOPED_TABLES = {"detection_classes", "detection_tracks", "detection_observations"}


def _wrap_execute_sql(original_tool):
    """Return a drop-in replacement for *original_tool* that enforces
    ``app_config_id`` scoping at the SQL level."""

    async def _scoped_execute(sql: str) -> str:
        config_id = current_app_config_id.get()
        if config_id is not None:
            sql_lower = sql.lower()
            touches_scoped = any(t in sql_lower for t in _SCOPED_TABLES)
            has_filter = f"app_config_id = {config_id}" in sql_lower
            if touches_scoped and not has_filter:
                return (
                    f"ERROR: Query rejected. You MUST include "
                    f"'detection_classes.app_config_id = {config_id}' "
                    f"when querying detection tables. Rewrite your query and retry."
                )
        return await original_tool.ainvoke({"sql": sql})

    return StructuredTool.from_function(
        coroutine=_scoped_execute,
        name=original_tool.name,
        description=original_tool.description,
    )


async def load_tools():
    """Load all postgres-mcp tools, wrapping ``execute_sql`` with config scoping."""
    tools = await _mcp_client.get_tools()
    log.info(
        "Loaded %d tools from postgres-mcp: %s",
        len(tools),
        [t.name for t in tools],
    )

    wrapped = []
    for tool in tools:
        if tool.name == "execute_sql":
            wrapped.append(_wrap_execute_sql(tool))
            log.info("Wrapped execute_sql with app_config_id scoping guard")
        else:
            wrapped.append(tool)
    return wrapped


async def load_sql_tool_only():
    """Load only the wrapped ``execute_sql`` tool, dropping all others."""
    all_tools = await load_tools()
    sql_tools = [t for t in all_tools if t.name == "execute_sql"]
    if not sql_tools:
        raise RuntimeError("execute_sql tool not found in postgres-mcp tools")
    log.info(
        "Loaded execute_sql tool only (filtered %d other tools)", len(all_tools) - 1
    )
    return sql_tools
