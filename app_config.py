from __future__ import annotations

import json
import os
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")
CONTROL_ROUTE_NAMES = frozenset({"skill", "stop", "noise", "timeout"})
ROUTE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
PROMPT_LAYOUT_FIELDS = {
    "llm": {
        "user_message": {
            "user_prompt",
            "user_input_label",
            "text",
        },
    },
    "knowledge": {
        "qa_item": {
            "index",
            "type",
            "question_label",
            "question",
            "answer_label",
            "answer",
            "source_label",
            "source",
            "score_label",
            "score",
        },
        "chunk_item": {
            "index",
            "type",
            "content_label",
            "content",
            "source_label",
            "source",
            "score_label",
            "score",
        },
    },
}


@dataclass(frozen=True)
class AgentConfig:
    """Shared AgentSDK and knowledge-base credentials."""

    app_id: str
    app_key: str
    app_secret: str
    agent_code: str
    sdk_url: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentConfig":
        return cls(
            app_id=require_secret(payload, "app_id", "app_id_env"),
            app_key=require_secret(payload, "app_key", "app_key_env"),
            app_secret=require_secret(payload, "app_secret", "app_secret_env"),
            agent_code=require_secret(payload, "agent_code", "agent_code_env"),
            sdk_url=require_str(payload, "sdk_url"),
        )


@dataclass(frozen=True)
class LlmModelConfig:
    """OpenAI-compatible model settings loaded from config.json."""

    provider: str
    model: str
    base_url: str
    api_key_env: str | None = None
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int = 256
    timeout_seconds: float = 3.0
    business_timeout_seconds: float = 3.0
    stream: bool = False
    extra_body: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.provider != "openai_compatible":
            raise ValueError("llm.provider only supports openai_compatible in this sample")
        if not self.api_key_env and not self.api_key:
            raise ValueError("llm.api_key_env or llm.api_key is required")
        if self.timeout_seconds <= 0:
            raise ValueError("llm.timeout_seconds must be greater than 0")
        if self.business_timeout_seconds <= 0:
            raise ValueError("llm.business_timeout_seconds must be greater than 0")
        if self.max_tokens <= 0:
            raise ValueError("llm.max_tokens must be greater than 0")
        if not 0 <= self.temperature <= 2:
            raise ValueError("llm.temperature must be between 0 and 2")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LlmModelConfig":
        timeout_seconds = get_float(payload, "timeout_seconds", 3.0)
        return cls(
            provider=require_str(payload, "provider"),
            model=require_str(payload, "model"),
            base_url=require_str(payload, "base_url"),
            api_key_env=get_optional_str(payload, "api_key_env"),
            api_key=get_optional_str(payload, "api_key"),
            temperature=get_float(payload, "temperature", 0.0),
            max_tokens=get_int(payload, "max_tokens", 256),
            timeout_seconds=timeout_seconds,
            business_timeout_seconds=get_float(
                payload,
                "business_timeout_seconds",
                timeout_seconds,
            ),
            stream=get_bool(payload, "stream", False),
            extra_body=get_optional_dict(payload, "extra_body"),
        )


@dataclass(frozen=True)
class KnowledgeBaseConfig:
    """Optional remote knowledge-base retrieval settings."""

    enabled: bool = False
    top_k: int = 10
    rerank: bool = True
    check_qa_first: bool = True
    score_threshold: float = 0.0
    timeout_seconds: float = 5.0
    business_timeout_seconds: float = 5.0
    api_url: str | None = None
    context_prompt: str | None = None
    empty_result: str | None = None
    error_result: str | None = None

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("knowledge_base.top_k must be greater than 0")
        if self.timeout_seconds <= 0:
            raise ValueError("knowledge_base.timeout_seconds must be greater than 0")
        if self.business_timeout_seconds <= 0:
            raise ValueError(
                "knowledge_base.business_timeout_seconds must be greater than 0"
            )
        if self.enabled and not self.api_url:
            raise ValueError("knowledge_base.api_url is required when enabled")
        if self.enabled and not self.context_prompt:
            raise ValueError("knowledge_base.context_prompt is required when enabled")
        if self.enabled and not self.empty_result:
            raise ValueError("knowledge_base.empty_result is required when enabled")
        if self.enabled and not self.error_result:
            raise ValueError("knowledge_base.error_result is required when enabled")

    @classmethod
    def disabled(cls) -> "KnowledgeBaseConfig":
        return cls(enabled=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "KnowledgeBaseConfig":
        timeout_seconds = get_float(payload, "timeout_seconds", 5.0)
        return cls(
            enabled=get_bool(payload, "enabled", False),
            top_k=get_int(payload, "top_k", 10),
            rerank=get_bool(payload, "rerank", True),
            check_qa_first=get_bool(payload, "check_qa_first", True),
            score_threshold=get_float(payload, "score_threshold", 0.0),
            timeout_seconds=timeout_seconds,
            business_timeout_seconds=get_float(
                payload,
                "business_timeout_seconds",
                timeout_seconds,
            ),
            api_url=get_optional_str(payload, "api_url"),
            context_prompt=get_optional_str(payload, "context_prompt"),
            empty_result=get_optional_str(payload, "empty_result"),
            error_result=get_optional_str(payload, "error_result"),
        )


@dataclass(frozen=True)
class LanguageConfig:
    """Language-specific messages and labels."""

    skill_messages: dict[str, str]
    llm_texts: dict[str, str]
    knowledge_texts: dict[str, str]
    knowledge_query_texts: dict[str, str]

    @classmethod
    def from_project_config(cls, payload: dict[str, Any]) -> "LanguageConfig":
        language = require_dict(payload, "language")
        return cls(
            skill_messages=require_str_dict(language, "skill_messages"),
            llm_texts=require_str_dict(
                language,
                "llm_texts",
            ),
            knowledge_texts=require_str_dict(
                language,
                "knowledge_texts",
            ),
            knowledge_query_texts=require_str_dict(
                language,
                "knowledge_query_texts",
            ),
        )


@dataclass(frozen=True)
class PromptLayoutConfig:
    """Developer-controlled layouts used to compose localized prompt text."""

    llm: dict[str, str]
    knowledge: dict[str, str]

    @classmethod
    def from_project_config(cls, payload: dict[str, Any]) -> "PromptLayoutConfig":
        layout = require_dict(payload, "prompt_layout")
        return cls(
            llm=_require_layout_section(layout, "llm"),
            knowledge=_require_layout_section(layout, "knowledge"),
        )


@dataclass(frozen=True)
class ReplyRouteConfig:
    """Configuration and handler registration for one reply intent."""

    name: str
    description: str
    handler: str
    waiting_replies: tuple[str, ...]
    agent: AgentConfig
    language: LanguageConfig
    prompt_layout: PromptLayoutConfig
    llm: LlmModelConfig
    system_prompt: str
    user_prompt: str
    timeout_reply_text: str
    knowledge_base: KnowledgeBaseConfig = field(default_factory=KnowledgeBaseConfig.disabled)

    @classmethod
    def from_dict(
        cls,
        name: str,
        payload: dict[str, Any],
        agent: AgentConfig,
        language: LanguageConfig,
        prompt_layout: PromptLayoutConfig,
    ) -> "ReplyRouteConfig":
        normalized_name = name.strip().lower()
        if not ROUTE_NAME_PATTERN.fullmatch(normalized_name):
            raise ValueError(
                "Reply route names must use lowercase letters, numbers, and underscores: "
                f"{name}"
            )
        if normalized_name in CONTROL_ROUTE_NAMES:
            raise ValueError(f"Control route cannot be configured as a reply route: {name}")

        knowledge_base = get_knowledge_base_config(payload)
        return cls(
            name=normalized_name,
            description=require_str(payload, "description"),
            handler=require_str(payload, "handler"),
            waiting_replies=get_str_tuple(payload, "waiting_replies"),
            agent=agent,
            language=language,
            prompt_layout=prompt_layout,
            llm=LlmModelConfig.from_dict(require_dict(payload, "llm")),
            system_prompt=require_str(payload, "system_prompt"),
            user_prompt=require_str(payload, "user_prompt"),
            timeout_reply_text=require_str(payload, "timeout_reply_text"),
            knowledge_base=knowledge_base,
        )


def load_project_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    path = Path(config_path)
    payload = _load_json_object(path, "Project config")
    locale_path = _resolve_config_file(path, require_str(payload, "locale_file"))
    layout_path = _resolve_config_file(
        path,
        require_str(payload, "prompt_layout_file"),
    )
    locale_payload = _load_json_object(locale_path, "Locale config")
    layout_payload = _load_json_object(layout_path, "Prompt layout config")
    skill_texts = locale_payload.pop("skill_texts", None)
    merged = _deep_merge(payload, locale_payload)
    merged["prompt_layout"] = layout_payload
    _merge_skill_texts(merged, skill_texts)
    return merged


def load_agent_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> AgentConfig:
    return AgentConfig.from_dict(require_dict(load_project_config(config_path), "agent"))


def load_reply_route_configs(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> tuple[ReplyRouteConfig, ...]:
    return parse_reply_route_configs(load_project_config(config_path))


def parse_reply_route_configs(
    project_config: dict[str, Any],
) -> tuple[ReplyRouteConfig, ...]:
    agent = AgentConfig.from_dict(require_dict(project_config, "agent"))
    language = LanguageConfig.from_project_config(project_config)
    prompt_layout = PromptLayoutConfig.from_project_config(project_config)
    routes = require_dict(project_config, "reply_routes")
    if not routes:
        raise ValueError("Config field reply_routes must not be empty")

    return tuple(
        ReplyRouteConfig.from_dict(
            name,
            require_dict(routes, name),
            agent,
            language,
            prompt_layout,
        )
        for name in routes
    )


def require_dict(payload: dict[str, Any], field_path: str) -> dict[str, Any]:
    value = _resolve(payload, field_path)
    if not isinstance(value, dict):
        raise ValueError(f"Config field must be an object: {field_path}")
    return value


def require_str(payload: dict[str, Any], field_path: str) -> str:
    value = _resolve(payload, field_path)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Config field must be a non-empty string: {field_path}")
    return value


def require_secret(
    payload: dict[str, Any],
    direct_field_path: str,
    env_field_path: str,
) -> str:
    env_name = get_optional_str(payload, env_field_path)
    if env_name:
        env_value = os.getenv(env_name)
        if not env_value or not env_value.strip():
            raise ValueError(
                f"Environment variable {env_name} is required for config field "
                f"{direct_field_path}"
            )
        return env_value
    return require_str(payload, direct_field_path)


def get_optional_str(payload: dict[str, Any], field_path: str) -> str | None:
    value = _resolve(payload, field_path, required=False)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Config field must be a string if set: {field_path}")
    return value


def get_optional_dict(
    payload: dict[str, Any],
    field_path: str,
) -> dict[str, Any] | None:
    value = _resolve(payload, field_path, required=False)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"Config field must be an object if set: {field_path}")
    return dict(value)


def get_knowledge_base_config(
    payload: dict[str, Any],
    field_path: str = "knowledge_base",
) -> KnowledgeBaseConfig:
    value = _resolve(payload, field_path, required=False)
    if value is None:
        return KnowledgeBaseConfig.disabled()
    if isinstance(value, dict):
        return KnowledgeBaseConfig.from_dict(value)
    raise ValueError(f"Config field must be an object: {field_path}")


def get_float(payload: dict[str, Any], field_path: str, default: float) -> float:
    value = _resolve(payload, field_path, required=False)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Config field must be a number: {field_path}") from exc


def get_int(payload: dict[str, Any], field_path: str, default: int) -> int:
    value = _resolve(payload, field_path, required=False)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Config field must be an integer: {field_path}") from exc


def get_bool(payload: dict[str, Any], field_path: str, default: bool) -> bool:
    value = _resolve(payload, field_path, required=False)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ValueError(f"Config field must be a boolean: {field_path}")


def get_str_tuple(
    payload: dict[str, Any],
    field_path: str,
    *,
    strip_items: bool = True,
) -> tuple[str, ...]:
    value = _resolve(payload, field_path)
    if not isinstance(value, list):
        raise ValueError(f"Config field must be a string list: {field_path}")
    if any(not isinstance(item, str) for item in value):
        raise ValueError(f"Config field must contain only strings: {field_path}")
    if strip_items:
        return tuple(item.strip() for item in value if item.strip())
    return tuple(item for item in value if item)


def require_str_dict(payload: dict[str, Any], field_path: str) -> dict[str, str]:
    value = require_dict(payload, field_path)
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"Config field has an invalid key: {field_path}")
        if not isinstance(item, str) or not item:
            raise ValueError(f"Config field must contain strings: {field_path}.{key}")
        result[key] = item
    return result


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _resolve_config_file(project_path: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return project_path.parent / path


def _require_layout_section(
    payload: dict[str, Any],
    section: str,
) -> dict[str, str]:
    templates = require_str_dict(payload, section)
    expected_templates = PROMPT_LAYOUT_FIELDS[section]
    configured_names = set(templates)
    expected_names = set(expected_templates)
    if configured_names != expected_names:
        missing = sorted(expected_names - configured_names)
        unknown = sorted(configured_names - expected_names)
        raise ValueError(
            f"Invalid prompt layout section {section}: "
            f"missing={missing}, unknown={unknown}"
        )

    for name, expected_fields in expected_templates.items():
        actual_fields = {
            field_name
            for _, field_name, _, _ in string.Formatter().parse(templates[name])
            if field_name is not None
        }
        if actual_fields != expected_fields:
            missing = sorted(expected_fields - actual_fields)
            unknown = sorted(actual_fields - expected_fields)
            raise ValueError(
                f"Invalid placeholders in prompt_layout.{section}.{name}: "
                f"missing={missing}, unknown={unknown}"
            )
    return templates


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overlay.items():
        current = result.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            result[key] = _deep_merge(current, value)
        else:
            result[key] = value
    return result


def _merge_skill_texts(payload: dict[str, Any], skill_texts: Any) -> None:
    if not isinstance(skill_texts, dict):
        raise ValueError("Locale config field skill_texts must be an object")
    actions = payload.get("skill_actions")
    if not isinstance(actions, list):
        raise ValueError("Config field skill_actions must be a list")

    configured_ids: set[str] = set()
    for action in actions:
        if not isinstance(action, dict):
            raise ValueError("skill_actions items must be objects")
        skill_id = require_str(action, "skill_id")
        localized = skill_texts.get(skill_id)
        if not isinstance(localized, dict):
            raise ValueError(f"Missing locale text for skill: {skill_id}")
        action.update(localized)
        configured_ids.add(skill_id)

    unknown_ids = set(skill_texts) - configured_ids
    if unknown_ids:
        raise ValueError(f"Locale config contains unknown skills: {sorted(unknown_ids)}")


def _resolve(payload: dict[str, Any], field_path: str, *, required: bool = True) -> Any:
    current: Any = payload
    for part in field_path.split("."):
        if not isinstance(current, dict) or part not in current:
            if required:
                raise ValueError(f"Missing config field: {field_path}")
            return None
        current = current[part]
    return current
