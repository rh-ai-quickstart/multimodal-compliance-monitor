import os

from logger import get_logger

log = get_logger(__name__)


def init_tracing() -> None:
    """Activate Phoenix OTEL tracing when PHOENIX_COLLECTOR_ENDPOINT is set.

    Does nothing (zero overhead) when the env var is absent, so normal dev
    runs are unaffected.  When the env var is present, ``phoenix.otel.register``
    auto-discovers the installed ``openinference-instrumentation-langchain``
    package and instruments all LangChain / LangGraph calls.
    """
    endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
    if not endpoint:
        return
    from phoenix.otel import register

    register(
        project_name="ppe-compliance-monitor",
        auto_instrument=True,
    )
    log.info("Phoenix tracing enabled -> %s", endpoint)
