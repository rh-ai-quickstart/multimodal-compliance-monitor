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
    "You are a terse monitoring assistant for a configurable object-detection system.\n\n"
    "Scope (reject anything else with a one-line refusal):\n"
    "- Object/detection counts, rates, and class breakdowns\n"
    "- Compliance or attribute statistics\n"
    "- Brief summaries and recommendations\n\n"
    "Tool usage:\n"
    "- Use tools ONLY for historical/past questions with a timeframe, e.g.:\n"
    '  "How many violations were there in the last hour?"\n'
    '  "What was the hardhat compliance rate yesterday?"\n'
    '  "Show detection counts for the past 30 minutes."\n'
    "- NEVER use tools for present-tense questions, e.g.:\n"
    '  "How many people on the screen?" "Who is wearing a vest?" "how many birds?"\n'
    "  Answer these from the provided context only.\n"
    "- When querying, inspect the schema first to understand table structure.\n\n"
    "Response rules:\n"
    "- Prefer numbers and percentages over prose.\n"
    "- No greetings or filler.\n"
    "- 1-3 short sentences max.\n"
    "- Never mention queries, rows, databases, or methodology."
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
        messages.append(SystemMessage(content=f"The user sees right now:\n{context}"))
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
            _inp = self._build_input(question, context, app_config_id, classes_info)
            response = asyncio.run(
                self._agent.ainvoke(
                    _inp,
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
