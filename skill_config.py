from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app_config import (
    DEFAULT_CONFIG_PATH,
    LanguageConfig,
    LlmModelConfig,
    PromptLayoutConfig,
    load_project_config,
    require_dict,
    require_str,
)


SKILL_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


@dataclass(frozen=True)
class SkillActionConfig:
    skill_id: str
    description: str
    skill_name: str
    skill_value: str
    reply_text: str = ""
    string_params: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    integer_params: tuple[tuple[str, int], ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, payload: dict[str, Any], index: int) -> "SkillActionConfig":
        skill_id = require_str(payload, "skill_id")
        if re.fullmatch(SKILL_ID_PATTERN, skill_id) is None:
            raise ValueError(f"Invalid function name in skill_actions[{index}].skill_id")
        return cls(
            skill_id=skill_id,
            description=require_str(payload, "description"),
            skill_name=require_str(payload, "skill_name"),
            skill_value=require_str(payload, "skill_value"),
            reply_text=_get_optional_str(payload, "reply_text") or "",
            string_params=_get_param_items(payload, "string_params", str),
            integer_params=_get_param_items(payload, "integer_params", int),
        )


@dataclass(frozen=True)
class SkillHandlerConfig:
    llm: LlmModelConfig
    system_prompt: str
    user_prompt: str
    timeout_reply_text: str
    language: LanguageConfig
    prompt_layout: PromptLayoutConfig

    @classmethod
    def from_project_config(cls, payload: dict[str, Any]) -> "SkillHandlerConfig":
        config = require_dict(payload, "skill_handler")
        return cls(
            llm=LlmModelConfig.from_dict(require_dict(config, "llm")),
            system_prompt=require_str(config, "system_prompt"),
            user_prompt=require_str(config, "user_prompt"),
            timeout_reply_text=require_str(config, "timeout_reply_text"),
            language=LanguageConfig.from_project_config(payload),
            prompt_layout=PromptLayoutConfig.from_project_config(payload),
        )


def load_skill_actions(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> tuple[SkillActionConfig, ...]:
    return load_skill_actions_from_project_config(load_project_config(config_path))


def load_skill_actions_from_project_config(
    project_config: dict[str, Any],
) -> tuple[SkillActionConfig, ...]:
    value = project_config.get("skill_actions")
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("Config field must be a list: skill_actions")

    actions = tuple(
        SkillActionConfig.from_dict(item, index)
        for index, item in enumerate(value)
        if isinstance(item, dict)
    )
    if len(actions) != len(value):
        raise ValueError("skill_actions items must be objects")
    skill_ids = [action.skill_id for action in actions]
    if len(skill_ids) != len(set(skill_ids)):
        raise ValueError("skill_actions.skill_id values must be unique")
    return actions


def _get_optional_str(payload: dict[str, Any], field: str) -> str | None:
    value = payload.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Config field must be a string: {field}")
    return value


def _get_param_items(
    payload: dict[str, Any],
    field: str,
    value_type: type,
) -> tuple[tuple[str, Any], ...]:
    value = payload.get(field)
    if value is None:
        return ()
    if not isinstance(value, dict):
        raise ValueError(f"Config field must be an object: {field}")

    items: list[tuple[str, Any]] = []
    for key, item_value in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"Config field has invalid key: {field}")
        if value_type is int:
            try:
                items.append((key, int(item_value)))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Config field must contain integers: {field}") from exc
        elif isinstance(item_value, value_type):
            items.append((key, item_value))
        else:
            raise ValueError(f"Config field has invalid value: {field}.{key}")
    return tuple(items)
