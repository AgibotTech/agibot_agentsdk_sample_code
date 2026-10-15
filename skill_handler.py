from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from linksoul_agentsdk import AgentParam

from app_config import DEFAULT_CONFIG_PATH, LanguageConfig, load_project_config
from llm_client import LlmClient
from skill_config import (
    SkillActionConfig,
    SkillHandlerConfig,
    load_skill_actions_from_project_config,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SkillHandleResult:
    history_text: str
    assistant_text: str


class SkillHandler:
    """Executes a skill selected by the LLM and checked against local config."""

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG_PATH) -> None:
        project_config = load_project_config(config_path)
        self._actions = load_skill_actions_from_project_config(project_config)
        self._actions_by_id = {action.skill_id: action for action in self._actions}
        self._messages = LanguageConfig.from_project_config(
            project_config
        ).skill_messages
        self._config = SkillHandlerConfig.from_project_config(project_config)
        self._llm = LlmClient.from_config(self._config)

    def close(self) -> None:
        self._llm.close()

    def handle(
        self,
        user_text: str,
        *,
        history: list[dict[str, str]] | None = None,
        session_id: str | None = None,
        param: Any | None = None,
        event_id: str,
        response: Any,
        item_id: str,
    ) -> SkillHandleResult:
        try:
            action, skill_params = self._select_action(user_text, history=history)
        except Exception as exc:  # noqa: BLE001 - skill selection has a spoken fallback.
            logger.exception(
                "skill_selection_failed session_id=%s event_id=%s error=%s",
                session_id,
                event_id,
                exc,
            )
            _send_text(
                response,
                event_id,
                item_id,
                self._config.timeout_reply_text,
            )
            return SkillHandleResult(
                history_text=self._config.timeout_reply_text,
                assistant_text=self._config.timeout_reply_text,
            )

        if action is None:
            logger.info(
                "skill_handler_unmatched session_id=%s event_id=%s text=%s",
                session_id,
                event_id,
                user_text.strip(),
            )
            _send_text(
                response,
                event_id,
                item_id,
                self._messages["unsupported_skill"],
            )
            unsupported_text = self._messages["unsupported_skill"]
            return SkillHandleResult(
                history_text=unsupported_text,
                assistant_text=unsupported_text,
            )

        try:
            skill_param = _build_param(action, skill_params)
        except ValueError as exc:
            logger.warning(
                "skill_handler_invalid_params session_id=%s event_id=%s skill_id=%s error=%s",
                session_id,
                event_id,
                action.skill_id,
                exc,
            )
            _send_text(
                response,
                event_id,
                item_id,
                self._messages["invalid_skill_params"],
            )
            invalid_params_text = self._messages["invalid_skill_params"]
            return SkillHandleResult(
                history_text=invalid_params_text,
                assistant_text=invalid_params_text,
            )

        response.on_skill(
            event_id,
            item_id,
            action.skill_name,
            action.skill_value,
            skill_param,
        )
        if action.reply_text:
            response.on_vlm_item_delta(event_id, item_id, action.reply_text)
        response.on_vlm_item_done(event_id, item_id)
        response.on_vlm_done(event_id)
        return SkillHandleResult(
            history_text=action.reply_text or action.description,
            assistant_text=action.reply_text,
        )

    def _select_action(
        self,
        user_text: str,
        *,
        history: list[dict[str, str]] | None = None,
    ) -> tuple[SkillActionConfig | None, dict[str, Any] | None]:
        if not self._actions:
            return None, None
        message = self._llm.complete_message(
            self._llm.build_messages(user_text, history=history),
            tools=[_action_tool(action) for action in self._actions],
            tool_choice="auto",
        )
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            return None, None
        if len(tool_calls) != 1:
            raise ValueError("skill model must call exactly one tool")
        tool_call = tool_calls[0]
        if not isinstance(tool_call, dict):
            return None, None
        function = tool_call.get("function")
        if not isinstance(function, dict):
            return None, None
        name = function.get("name")
        action = self._actions_by_id.get(name) if isinstance(name, str) else None
        if action is None:
            return None, None
        arguments = json.loads(function.get("arguments") or "{}")
        if not isinstance(arguments, dict):
            raise ValueError("skill tool arguments must be an object")
        return action, arguments or None


def _action_tool(action: SkillActionConfig) -> dict[str, Any]:
    properties: dict[str, dict[str, str]] = {}
    for key, _ in action.string_params:
        properties[key] = {"type": "string"}
    for key, _ in action.integer_params:
        properties[key] = {"type": "integer"}
    return {
        "type": "function",
        "function": {
            "name": action.skill_id,
            "description": action.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "additionalProperties": False,
            },
        },
    }


def _send_text(
    response: Any,
    event_id: str,
    item_id: str,
    text: str,
) -> None:
    response.on_vlm_item_delta(event_id, item_id, text)
    response.on_vlm_item_done(event_id, item_id)
    response.on_vlm_done(event_id)


def _build_param(
    action: SkillActionConfig,
    skill_params: dict[str, Any] | None = None,
) -> AgentParam:
    params = skill_params or {}
    string_defaults = dict(action.string_params)
    integer_defaults = dict(action.integer_params)
    allowed_keys = set(string_defaults) | set(integer_defaults)
    unknown_keys = set(params) - allowed_keys
    if unknown_keys:
        raise ValueError(f"unsupported skill parameters: {sorted(unknown_keys)}")

    param = AgentParam.create().set_string(action.skill_name, action.skill_value)
    for key, value in action.string_params:
        override = params.get(key, value)
        if not isinstance(override, str):
            raise ValueError(f"skill parameter must be a string: {key}")
        param.set_string(key, override)
    for key, value in action.integer_params:
        override = params.get(key, value)
        if isinstance(override, bool) or not isinstance(override, int):
            raise ValueError(f"skill parameter must be an integer: {key}")
        param.set_integer(key, override)
    return param
