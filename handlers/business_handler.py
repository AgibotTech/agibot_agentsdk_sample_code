from __future__ import annotations

import html
import logging
from typing import Any, Callable

from app_config import ReplyRouteConfig
from base_handler import BaseHandler, HandlerContext
from knowledge_retriever import KnowledgeRetriever
from llm_client import LlmClient


logger = logging.getLogger(__name__)


_KNOWLEDGE_QUERY_HISTORY_MESSAGES = 4
_KNOWLEDGE_QUERY_MAX_CHARS = 800


class BusinessHandler(BaseHandler):
    """Hospital guidance reply handler."""

    def __init__(
        self,
        config: ReplyRouteConfig,
        llm: LlmClient | None = None,
        knowledge_retriever: KnowledgeRetriever | None = None,
    ) -> None:
        super().__init__(config, llm)
        self.knowledge_retriever = knowledge_retriever
        if self.knowledge_retriever is None and config.knowledge_base.enabled:
            self.knowledge_retriever = KnowledgeRetriever.from_config(
                config.knowledge_base,
                config.agent,
                config.language.knowledge_texts,
                config.prompt_layout.knowledge,
            )

    def handle(self, context: HandlerContext) -> str:
        text = context.user_text.strip()
        sent_chunks: list[str] = []

        def send_content(content: str) -> None:
            context.response.on_vlm_item_delta(
                context.event_id,
                context.item_id,
                content,
            )
            sent_chunks.append(content)

        try:
            messages = self.llm.build_messages(text, history=context.history)
            if self.knowledge_retriever is not None:
                knowledge_status, knowledge_context = self._retrieve_knowledge_context(
                    text,
                    context.history,
                )
                _attach_knowledge_reference(
                    messages,
                    status=knowledge_status,
                    content=knowledge_context,
                )
            reply_text = self._send_model_reply(
                context,
                messages,
                send_content,
            )

            context.response.on_vlm_item_done(context.event_id, context.item_id)
            context.response.on_vlm_done(context.event_id)
            return reply_text.strip()
        except TimeoutError as exc:
            logger.exception(
                "business_handler_timeout session_id=%s event_id=%s error=%s",
                context.session_id,
                context.event_id,
                exc,
            )
            send_content(self.config.timeout_reply_text)
            context.response.on_vlm_item_done(context.event_id, context.item_id)
            context.response.on_vlm_done(context.event_id)
            return "".join(sent_chunks).strip()
        except Exception as exc:  # noqa: BLE001 - sample handler uses one fallback.
            logger.exception(
                "business_handler_failed session_id=%s event_id=%s error=%s",
                context.session_id,
                context.event_id,
                exc,
            )
            if not sent_chunks:
                send_content(self.config.timeout_reply_text)
            context.response.on_vlm_item_done(context.event_id, context.item_id)
            context.response.on_vlm_done(context.event_id)
            return "".join(sent_chunks).strip()

    def _retrieve_knowledge_context(
        self,
        text: str,
        history: list[dict[str, str]],
    ) -> tuple[str, str]:
        config = self.config.knowledge_base
        try:
            if self.knowledge_retriever is None:
                raise RuntimeError("knowledge retriever is unavailable")
            query = _build_knowledge_query(
                text,
                history,
                self.config.language.knowledge_query_texts,
            )
            retrieved = self.knowledge_retriever.retrieve_context(query)
            if retrieved:
                return "ok", retrieved
            return "empty", config.empty_result or ""
        except Exception as exc:  # noqa: BLE001 - constrain the answer on retrieval failure.
            logger.warning("knowledge_retrieval_failed error=%s", exc)
            return "unavailable", config.error_result or ""

    def _send_model_reply(
        self,
        context: HandlerContext,
        messages: list[dict[str, Any]],
        send_content: Callable[[str], None],
    ) -> str:
        if self.llm.stream_enabled:
            chunks: list[str] = []
            for chunk in self.llm.stream_messages(messages):
                send_content(chunk)
                chunks.append(chunk)
            reply_text = "".join(chunks).strip()
        else:
            reply_text = _message_content(self.llm.complete_message(messages))
            send_content(reply_text)
        if not reply_text:
            raise RuntimeError("business handler returned empty reply")
        return reply_text


def _message_content(message: dict[str, Any]) -> str:
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("business model returned empty content")
    return content.strip()


def _build_knowledge_query(
    text: str,
    history: list[dict[str, str]],
    query_texts: dict[str, str],
) -> str:
    current_line = f"{query_texts['current_question_prefix']}{text.strip()}"
    if len(current_line) >= _KNOWLEDGE_QUERY_MAX_CHARS:
        return current_line[:_KNOWLEDGE_QUERY_MAX_CHARS]

    recent_lines: list[str] = []
    for message in history[-_KNOWLEDGE_QUERY_HISTORY_MESSAGES:]:
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"}:
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        prefix = (
            query_texts["user_prefix"]
            if role == "user"
            else query_texts["assistant_prefix"]
        )
        recent_lines.append(f"{prefix}{content.strip()}")

    if not recent_lines:
        return current_line

    heading = f"\n{query_texts['recent_conversation_heading']}\n"
    remaining = _KNOWLEDGE_QUERY_MAX_CHARS - len(current_line) - len(heading)
    selected_reversed: list[str] = []
    for line in reversed(recent_lines):
        separator_length = 1 if selected_reversed else 0
        available = remaining - separator_length
        if available <= 0:
            break
        if len(line) > available:
            selected_reversed.append(line[:available])
            break
        selected_reversed.append(line)
        remaining = available - len(line)

    if not selected_reversed:
        return current_line
    return current_line + heading + "\n".join(reversed(selected_reversed))


def _attach_knowledge_reference(
    messages: list[dict[str, Any]],
    *,
    status: str,
    content: str,
) -> None:
    if not messages or messages[-1].get("role") != "user":
        raise ValueError("business messages must end with a user message")
    user_content = messages[-1].get("content")
    if not isinstance(user_content, str):
        raise ValueError("business user message content must be text")

    escaped_content = html.escape(content.strip(), quote=False)
    reference = (
        f'<hospital_reference status="{status}">\n'
        f"{escaped_content}\n"
        "</hospital_reference>"
    )
    messages[-1]["content"] = f"{user_content.rstrip()}\n\n{reference}"
