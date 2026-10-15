from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from app_config import (
    CONTROL_ROUTE_NAMES,
    DEFAULT_CONFIG_PATH,
    LanguageConfig,
    LlmModelConfig,
    PromptLayoutConfig,
    ROUTE_NAME_PATTERN,
    load_project_config,
    require_dict,
    require_str,
)


logger = logging.getLogger(__name__)


class ScopeRoute(str, Enum):
    SKILL = "skill"
    STOP = "stop"
    NOISE = "noise"
    TIMEOUT = "timeout"


class TextLlmClient(Protocol):
    """Minimal adapter expected by ScopeRouter.

    Real projects can wrap OpenAI-compatible APIs, private model services,
    customer Agents, or any other classifier behind this callable interface.
    """

    def build_messages(
        self,
        text: str,
        *,
        history: list[dict[str, str]] | None = None,
    ) -> list[dict[str, Any]]:
        ...

    def complete_message(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class ScopeRouteResult:
    route: ScopeRoute | str
    raw_output: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "route": route_name(self.route),
        }
        return result


@dataclass(frozen=True)
class ScopeRouterConfig:
    """Typed schema loaded from config.json."""

    llm: LlmModelConfig
    system_prompt: str
    user_prompt: str
    timeout_reply_text: str
    reply_routes: tuple[tuple[str, str], ...]
    control_routes: tuple[tuple[str, str], ...]
    language: LanguageConfig
    prompt_layout: PromptLayoutConfig

    def __post_init__(self) -> None:
        if not self.system_prompt.strip():
            raise ValueError("router.system_prompt must not be empty")
        if not self.user_prompt.strip():
            raise ValueError("router.user_prompt must not be empty")
        if not self.timeout_reply_text.strip():
            raise ValueError("router.timeout_reply_text must not be empty")
        if not self.reply_routes:
            raise ValueError("router.reply_routes must not be empty")
        if {name for name, _ in self.control_routes} != {
            ScopeRoute.SKILL.value,
            ScopeRoute.STOP.value,
            ScopeRoute.NOISE.value,
        }:
            raise ValueError("router.control_routes must define skill, stop, and noise")

    @classmethod
    def from_file(cls, config_path: str | Path = DEFAULT_CONFIG_PATH) -> "ScopeRouterConfig":
        return cls.from_project_config(load_project_config(config_path))

    @classmethod
    def from_project_config(cls, payload: dict[str, Any]) -> "ScopeRouterConfig":
        reply_routes = _load_reply_route_descriptions(payload)
        language = LanguageConfig.from_project_config(payload)
        prompt_layout = PromptLayoutConfig.from_project_config(payload)
        return cls.from_dict(
            require_dict(payload, "scope_router"),
            language=language,
            prompt_layout=prompt_layout,
            reply_routes=reply_routes,
        )

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        language: LanguageConfig,
        prompt_layout: PromptLayoutConfig,
        reply_routes: tuple[tuple[str, str], ...],
    ) -> "ScopeRouterConfig":
        llm_payload = require_dict(payload, "llm")
        user_prompt = require_str(payload, "user_prompt")
        control_routes_payload = require_dict(payload, "control_routes")
        control_routes = tuple(
            (name, require_str(control_routes_payload, name))
            for name in (
                ScopeRoute.SKILL.value,
                ScopeRoute.STOP.value,
                ScopeRoute.NOISE.value,
            )
        )
        return cls(
            llm=LlmModelConfig.from_dict(llm_payload),
            system_prompt=require_str(payload, "system_prompt"),
            user_prompt=user_prompt,
            timeout_reply_text=require_str(payload, "timeout_reply_text"),
            reply_routes=reply_routes,
            control_routes=control_routes,
            language=language,
            prompt_layout=prompt_layout,
        )


class ScopeRouter:
    """Classifies ASR text into configured reply routes or control routes.

    Developers usually only need to replace ScopeRouterConfig and the llm_client
    adapter. AgentSDK event handling code can keep calling classify(text).
    """

    def __init__(
        self,
        llm_client: TextLlmClient | None = None,
        config: ScopeRouterConfig | None = None,
    ) -> None:
        self._config = config or load_scope_router_config()
        self._llm_client = llm_client or self._create_default_llm_client(self._config)

    @classmethod
    def from_config_file(
        cls,
        llm_client: TextLlmClient | None = None,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ) -> "ScopeRouter":
        return cls(llm_client, ScopeRouterConfig.from_file(config_path))

    @property
    def config(self) -> ScopeRouterConfig:
        return self._config

    def classify(
        self,
        user_text: str,
        *,
        session_id: str | None = None,
        event_id: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> ScopeRouteResult:
        normalized_text = user_text.strip()
        if not normalized_text:
            return ScopeRouteResult(
                route=ScopeRoute.NOISE,
            )

        try:
            messages = self._llm_client.build_messages(
                normalized_text,
                history=history,
            )
            raw_output = self._llm_client.complete_message(
                messages,
                tools=self._route_tools(),
                tool_choice="required",
            )
        except TimeoutError:
            logger.info(
                "scope_router_timeout session_id=%s event_id=%s",
                session_id,
                event_id,
            )
            return ScopeRouteResult(route=ScopeRoute.TIMEOUT)
        except Exception as exc:  # noqa: BLE001 - sample code keeps router isolated.
            logger.exception(
                "scope_router_llm_failed session_id=%s event_id=%s error=%s",
                session_id,
                event_id,
                exc,
            )
            return self._fallback(
                reason="router_llm_failed",
                session_id=session_id,
                event_id=event_id,
            )

        return self._parse_tool_call(
            raw_output,
            session_id=session_id,
            event_id=event_id,
        )

    def build_prompt(
        self,
        user_text: str,
    ) -> str:
        """Returns the rendered user prompt for backward-compatible debugging."""

        build_user_prompt = getattr(self._llm_client, "build_user_prompt", None)
        if callable(build_user_prompt):
            return build_user_prompt(user_text)
        return user_text

    def close(self) -> None:
        close = getattr(self._llm_client, "close", None)
        if callable(close):
            close()

    def _parse_tool_call(
        self,
        raw_output: dict[str, Any],
        *,
        session_id: str | None,
        event_id: str | None,
    ) -> ScopeRouteResult:
        if not isinstance(raw_output, dict):
            return self._fallback(
                reason="invalid_router_message",
                raw_output=raw_output,
                session_id=session_id,
                event_id=event_id,
            )
        tool_calls = raw_output.get("tool_calls")
        if not isinstance(tool_calls, list) or len(tool_calls) != 1:
            return self._fallback(
                reason="router_must_call_exactly_one_tool",
                raw_output=raw_output,
                session_id=session_id,
                event_id=event_id,
            )
        tool_call = tool_calls[0]
        if not isinstance(tool_call, dict):
            return self._fallback(
                reason="invalid_router_tool_call",
                raw_output=raw_output,
                session_id=session_id,
                event_id=event_id,
            )
        function = tool_call.get("function")
        if not isinstance(function, dict):
            return self._fallback(
                reason="router_tool_call_missing_function",
                raw_output=raw_output,
                session_id=session_id,
                event_id=event_id,
            )
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except (TypeError, json.JSONDecodeError):
            return self._fallback(
                reason="invalid_router_tool_arguments",
                raw_output=raw_output,
                session_id=session_id,
                event_id=event_id,
            )
        if not isinstance(arguments, dict) or arguments:
            return self._fallback(
                reason="router_tool_arguments_must_be_empty",
                raw_output=raw_output,
                session_id=session_id,
                event_id=event_id,
            )
        route = parse_scope_route(
            function.get("name"),
            allowed_reply_routes={name for name, _ in self._config.reply_routes},
        )
        if route is None:
            return self._fallback(
                reason="unknown_router_route",
                raw_output=raw_output,
                session_id=session_id,
                event_id=event_id,
            )
        if route is ScopeRoute.TIMEOUT:
            return self._fallback(
                reason="reserved_timeout_route",
                raw_output=raw_output,
                session_id=session_id,
                event_id=event_id,
            )
        return ScopeRouteResult(
            route=route,
            raw_output=raw_output,
        )

    def _route_tools(self) -> list[dict[str, Any]]:
        routes = (*self._config.reply_routes, *self._config.control_routes)
        return [_route_tool(name, description) for name, description in routes]

    def _fallback(
        self,
        *,
        reason: str,
        session_id: str | None,
        event_id: str | None,
        raw_output: Any | None = None,
    ) -> ScopeRouteResult:
        logger.info(
            "scope_router_fallback session_id=%s event_id=%s reason=%s",
            session_id,
            event_id,
            reason,
        )
        return ScopeRouteResult(
            route=ScopeRoute.TIMEOUT,
            raw_output=raw_output,
        )

    @staticmethod
    def _create_default_llm_client(config: ScopeRouterConfig) -> TextLlmClient:
        from llm_client import LlmClient

        return LlmClient.from_config(config)


def load_scope_router_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> ScopeRouterConfig:
    return ScopeRouterConfig.from_file(config_path)


def parse_scope_route(
    value: Any,
    allowed_reply_routes: set[str] | None = None,
) -> ScopeRoute | str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    known_route = None
    for route in ScopeRoute:
        if route.value == normalized:
            known_route = route
            break
    if known_route in {
        ScopeRoute.SKILL,
        ScopeRoute.STOP,
        ScopeRoute.NOISE,
        ScopeRoute.TIMEOUT,
    }:
        return known_route
    if allowed_reply_routes is None:
        return known_route
    if normalized not in allowed_reply_routes:
        return None
    return known_route or normalized


def route_name(route: ScopeRoute | str) -> str:
    if isinstance(route, ScopeRoute):
        return route.value
    return route


def _route_tool(name: str, description: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    }


def _load_reply_route_descriptions(
    payload: dict[str, Any],
) -> tuple[tuple[str, str], ...]:
    configured_routes = require_dict(payload, "reply_routes")
    routes: list[tuple[str, str]] = []
    for name, route_payload in configured_routes.items():
        if not isinstance(name, str) or ROUTE_NAME_PATTERN.fullmatch(name) is None:
            raise ValueError(f"Invalid reply route name: {name}")
        if name in CONTROL_ROUTE_NAMES:
            raise ValueError(f"Control route cannot be configured as a reply route: {name}")
        if not isinstance(route_payload, dict):
            raise ValueError(f"Config field must be an object: reply_routes.{name}")
        routes.append((name, require_str(route_payload, "description")))
    if not routes:
        raise ValueError("Config field reply_routes must not be empty")
    return tuple(routes)
