from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel


class AlertState(TypedDict):
    alert_text: str
    app_config_id: int
    classes_info: list[dict] | None
    metric: str
    sql_query: str


class AlertEntry(BaseModel):
    id: str
    app_config_id: int
    rule: str
    severity: Literal["low", "medium", "high"] = "medium"
    status: Literal["pending", "processing", "done", "error"] = "pending"
    sql_query: str | None = None
    error: str | None = None


class AlertsStore(BaseModel):
    configs: dict[int, dict[str, AlertEntry]] = {}
