from __future__ import annotations

import logging

from base_handler import BaseHandler, HandlerContext


logger = logging.getLogger(__name__)


class GeneralChatHandler(BaseHandler):
    """General-chat reply handler."""

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
            if self.llm.stream_enabled:
                chunks: list[str] = []
                for chunk in self.llm.stream(text, history=context.history):
                    send_content(chunk)
                    chunks.append(chunk)
                if not sent_chunks:
                    raise RuntimeError("general chat handler returned empty reply")
                reply_text = "".join(chunks)
            else:
                reply_text = self.llm(text, history=context.history)
                if not reply_text.strip():
                    raise RuntimeError("general chat handler returned empty reply")
                send_content(reply_text)

            context.response.on_vlm_item_done(context.event_id, context.item_id)
            context.response.on_vlm_done(context.event_id)
            return reply_text.strip()
        except TimeoutError as exc:
            logger.exception(
                "general_chat_handler_timeout session_id=%s event_id=%s error=%s",
                context.session_id,
                context.event_id,
                exc,
            )
            send_content(self.config.timeout_reply_text)
            context.response.on_vlm_item_done(context.event_id, context.item_id)
            context.response.on_vlm_done(context.event_id)
            return "".join(sent_chunks).strip()
        except Exception as exc:  # noqa: BLE001 - sample handler returns unified fallback.
            logger.exception(
                "general_chat_handler_failed session_id=%s event_id=%s error=%s",
                context.session_id,
                context.event_id,
                exc,
            )
            if not sent_chunks:
                send_content(self.config.timeout_reply_text)
            context.response.on_vlm_item_done(context.event_id, context.item_id)
            context.response.on_vlm_done(context.event_id)
            return "".join(sent_chunks).strip()
