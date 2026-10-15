from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app_config import load_project_config


@dataclass(frozen=True)
class ConversationTurn:
    user_text: str
    assistant_text: str
    created_at: float


class ConversationManager:
    """Small in-memory conversation store for the single robot process."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        idle_timeout_seconds: float = 40.0,
        max_turns: int = 6,
        max_history_chars: int = 3000,
    ) -> None:
        if idle_timeout_seconds <= 0:
            raise ValueError("conversation.idle_timeout_seconds must be greater than 0")
        if max_turns <= 0:
            raise ValueError("conversation.max_turns must be greater than 0")
        if max_history_chars <= 0:
            raise ValueError("conversation.max_history_chars must be greater than 0")

        self.enabled = enabled
        self.idle_timeout_seconds = idle_timeout_seconds
        self.max_turns = max_turns
        self.max_history_chars = max_history_chars
        self._session_id: str | None = None
        self._updated_at = 0.0
        self._turns: list[ConversationTurn] = []
        self._lock = threading.Lock()

    @classmethod
    def from_config_file(
        cls,
        config_path: str | Path = "config.json",
    ) -> "ConversationManager":
        payload = load_project_config(config_path).get("conversation", {})
        if not isinstance(payload, dict):
            raise ValueError("Config field must be an object: conversation")
        return cls(
            enabled=_get_bool(payload, "enabled", True),
            idle_timeout_seconds=_get_float(payload, "idle_timeout_seconds", 40.0),
            max_turns=_get_int(payload, "max_turns", 6),
            max_history_chars=_get_int(payload, "max_history_chars", 3000),
        )

    def get_history(self, session_id: str | None) -> list[dict[str, str]]:
        if not self.enabled or not session_id:
            return []
        with self._lock:
            self._prepare_session(session_id)
            messages: list[dict[str, str]] = []
            for turn in self._turns:
                messages.append({"role": "user", "content": turn.user_text})
                if turn.assistant_text:
                    messages.append({"role": "assistant", "content": turn.assistant_text})
            return messages

    def add_turn(
        self,
        session_id: str | None,
        *,
        user_text: str,
        assistant_text: str,
    ) -> None:
        if not self.enabled or not session_id:
            return
        user_text = user_text.strip()
        assistant_text = assistant_text.strip()
        if not user_text or not assistant_text:
            return

        with self._lock:
            self._prepare_session(session_id)
            self._turns.append(
                ConversationTurn(
                    user_text=user_text,
                    assistant_text=assistant_text,
                    created_at=time.time(),
                )
            )
            self._trim()
            self._updated_at = time.time()

    def add_failed_turn(
        self,
        session_id: str | None,
        *,
        user_text: str,
        assistant_text: str,
    ) -> None:
        """Keep a failed request and the fallback that was actually spoken."""

        self.add_turn(
            session_id,
            user_text=user_text,
            assistant_text=assistant_text,
        )

    def clear(self) -> None:
        with self._lock:
            self._session_id = None
            self._updated_at = 0.0
            self._turns.clear()

    def _prepare_session(self, session_id: str) -> None:
        now = time.time()
        expired = self._updated_at > 0 and now - self._updated_at > self.idle_timeout_seconds
        if self._session_id != session_id or expired:
            self._session_id = session_id
            self._turns.clear()
            self._updated_at = 0.0

    def _trim(self) -> None:
        while len(self._turns) > self.max_turns or self._history_chars() > self.max_history_chars:
            self._turns.pop(0)

    def _history_chars(self) -> int:
        return sum(len(turn.user_text) + len(turn.assistant_text) for turn in self._turns)


def _get_bool(payload: dict[str, Any], field: str, default: bool) -> bool:
    value = payload.get(field, default)
    if not isinstance(value, bool):
        raise ValueError(f"Config field must be a boolean: conversation.{field}")
    return value


def _get_float(payload: dict[str, Any], field: str, default: float) -> float:
    value = payload.get(field, default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Config field must be a number: conversation.{field}") from exc


def _get_int(payload: dict[str, Any], field: str, default: int) -> int:
    value = payload.get(field, default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Config field must be an integer: conversation.{field}") from exc
