from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from app_config import ReplyRouteConfig
from llm_client import LlmClient


@dataclass(frozen=True)
class HandlerContext:
    """Data shared by all reply-route handlers."""

    user_text: str
    session_id: str | None
    param: Any | None
    event_id: str
    response: Any
    item_id: str
    history: list[dict[str, str]]


class ReplyHandler(Protocol):
    def handle(self, context: HandlerContext) -> str:
        ...

    def close(self) -> None:
        ...


class BaseHandler:
    def __init__(
        self,
        config: ReplyRouteConfig,
        llm: LlmClient | None = None,
    ) -> None:
        self.config = config
        self.llm = llm or LlmClient.from_config(config)

    def close(self) -> None:
        self.llm.close()
