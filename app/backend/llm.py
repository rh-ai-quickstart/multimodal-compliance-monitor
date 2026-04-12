import asyncio
import os
from typing import Generator

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessageChunk, SystemMessage
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

from tools.mcp_tools import load_tools, current_app_config_id
from logger import get_logger

log = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are a terse monitoring assistant for a configurable object-detection system. "
    "You have access to read-only PostgreSQL tools via an MCP server. "
    "Use execute_sql to run SELECT queries and answer questions with real data.\n\n"
    "Database schema:\n"
    "  app_config(id SERIAL PK, model_url VARCHAR, model_name VARCHAR, video_source VARCHAR, created_at TIMESTAMP)\n"
    "  detection_classes(id SERIAL PK, app_config_id INTEGER FK→app_config, "
    "model_class_index INTEGER, name VARCHAR, trackable BOOLEAN, "
    "include_in_counts BOOLEAN)\n"
    "  detection_tracks(track_id INTEGER PK, detection_classes_id INTEGER FK→detection_classes, "
    "first_seen TIMESTAMP, last_seen TIMESTAMP)\n"
    "  detection_observations(id SERIAL PK, track_id INTEGER FK→detection_tracks, "
    "timestamp TIMESTAMP, attributes JSONB)\n"
    "  FK chain: app_config → detection_classes → detection_tracks → detection_observations (all CASCADE).\n"
    "  Each app_config has its own model and video source with its own set of detection_classes.\n"
    "  detection_observations stores ONE ROW PER STATE CHANGE per tracked object (not one per object). "
    "To count unique objects, use COUNT(DISTINCT o.track_id). "
    "COUNT(*) counts observation rows, which inflates numbers.\n"
    "  attributes is JSONB and its keys depend on the config (e.g. hardhat, vest, mask for PPE).\n\n"
    "Scope (reject anything else with a one-line refusal):\n"
    "• Object/detection counts and rates\n"
    "• Compliance or attribute statistics\n"
    "• Detection class breakdowns\n"
    "• Brief summaries and recommendations\n\n"
    "Rules:\n"
    '1. Use tools ONLY if the answer to the question does not exist "on the screen."\n'
    "2. DO NOT use tools if the user asks without specifying a timeframe.\n"
    "3. Prefer numbers and percentages over prose.\n"
    "4. No greetings or filler words.\n"
    "5. Respond in 1-3 short sentences max.\n"
    "6. Never explain methodology—do not mention observation rows, queries, database, or how you arrived at the answer. State only the direct answer."
)


class LLMChat:
    """Conversational LLM backed by a VLLM-served OpenAI-compatible endpoint.

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

        tools = asyncio.run(load_tools())

        self._memory = MemorySaver()
        self._agent = create_agent(
            llm,
            tools,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=self._memory,
        )
        self._session_versions: dict[str, int] = {}

        log.info(
            "LLMChat initialised — endpoint=%s, model=%s, mcp_tools=%d",
            endpoint,
            model,
            len(tools),
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
    ) -> dict:
        messages: list = []
        if app_config_id is not None:
            constraint = (
                f"IMPORTANT: The user is viewing app_config id={app_config_id}. "
                f"ALL SQL queries MUST join or filter through "
                f"detection_classes.app_config_id = {app_config_id}. "
                f"Never query data from other configs.\n"
            )
            if classes_info:
                class_lines = ", ".join(
                    f"{c['name']} (trackable={c['trackable']})" for c in classes_info
                )
                constraint += f"Detection classes for this config: {class_lines}\n"
            messages.append(SystemMessage(content=constraint))
        messages.append(
            SystemMessage(content=f"The user sees on the screen:\n{context}")
        )
        messages.append(HumanMessage(content=f"User question: {question}"))
        return {"messages": messages}

    def chat(
        self,
        question: str,
        context: str,
        session_id: str = "default",
        app_config_id: int | None = None,
        classes_info: list[dict] | None = None,
    ) -> str:
        """Send a question with context through the conversational agent.

        Every prior exchange in *session_id* is automatically included so the
        model can reference earlier questions and answers.

        Uses ainvoke because MCP tools are async-only.
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
            response = asyncio.run(
                self._agent.ainvoke(
                    self._build_input(question, context, app_config_id, classes_info),
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

        Uses astream because MCP tools are async-only.
        """

        async def _astream():
            chunks = []
            async for msg, _metadata in self._agent.astream(
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
