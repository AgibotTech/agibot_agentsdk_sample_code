from __future__ import annotations

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterator


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LlmClient:
    """OpenAI-compatible LLM client with config-based prompt rendering.

    The caller provides only the user text. This object owns:
    - model connection settings from config.llm
    - system_prompt and user_prompt rendering
    - non-streaming, streaming, and tool-capable model calls
    """

    api_key: str
    model: str
    base_url: str
    system_prompt: str
    user_prompt: str
    language_texts: dict[str, str]
    prompt_layouts: dict[str, str]
    temperature: float = 0.0
    max_tokens: int = 256
    timeout_seconds: float = 3.0
    business_timeout_seconds: float = 3.0
    stream_enabled: bool = False
    extra_body: dict[str, Any] | None = None
    _client: Any | None = field(default=None, init=False, repr=False, compare=False)
    _client_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
        compare=False,
    )

    @classmethod
    def from_config(cls, config: Any) -> "LlmClient":
        llm_config = config.llm
        api_key = llm_config.api_key
        if not api_key and llm_config.api_key_env:
            api_key = os.getenv(llm_config.api_key_env)
        if not api_key:
            raise ValueError(
                "Missing API key. Set llm.api_key or configure llm.api_key_env "
                "and provide the matching environment variable."
            )

        return cls(
            api_key=api_key,
            model=llm_config.model,
            base_url=llm_config.base_url,
            system_prompt=config.system_prompt,
            user_prompt=config.user_prompt,
            language_texts=config.language.llm_texts,
            prompt_layouts=config.prompt_layout.llm,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
            timeout_seconds=llm_config.timeout_seconds,
            business_timeout_seconds=llm_config.business_timeout_seconds,
            stream_enabled=llm_config.stream,
            extra_body=llm_config.extra_body,
        )

    def __call__(
        self,
        text: str,
        *,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        message = self.complete_message(self.build_messages(text, history=history))
        return _require_message_content(message)

    def stream(
        self,
        text: str,
        *,
        history: list[dict[str, str]] | None = None,
    ) -> Iterator[str]:
        yield from self.stream_messages(self.build_messages(text, history=history))

    def complete_message(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        started_at = time.perf_counter()
        succeeded = False
        try:
            from openai import APITimeoutError
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required. Install project dependencies first."
            ) from exc

        client = self._get_client()
        try:
            try:
                message = self._run_with_business_timeout(
                    lambda: self._create_completion_message(
                        client,
                        messages,
                        tools=tools,
                        tool_choice=tool_choice,
                    ),
                    operation="chat_completion",
                )
            except APITimeoutError as exc:
                raise TimeoutError("OpenAI-compatible chat API timed out") from exc
            succeeded = True
            return message
        finally:
            self._log_llm_elapsed(started_at, stream=False, succeeded=succeeded)

    def stream_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> Iterator[str]:
        started_at = time.perf_counter()
        succeeded = False
        try:
            from openai import APITimeoutError
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required. Install project dependencies first."
            ) from exc

        client = self._get_client()
        try:
            self._log_request_messages(messages, stream=True)
            request_started_at = time.perf_counter()
            first_content_logged = False
            first_sentence_logged = False
            sentence_buffer: list[str] = []
            stream_events = self._start_stream_worker(client, messages)

            while True:
                try:
                    kind, payload = stream_events.queue.get(
                        timeout=self.business_timeout_seconds,
                    )
                except queue.Empty as exc:
                    stream_events.stop()
                    operation = (
                        "stream_idle"
                        if first_content_logged
                        else "stream_first_content"
                    )
                    self._log_llm_business_timeout(
                        request_started_at,
                        operation=operation,
                    )
                    raise TimeoutError(
                        "OpenAI-compatible chat API stream business timeout"
                    ) from exc

                if kind == "done":
                    break
                if kind == "error":
                    raise payload

                content = payload
                if isinstance(content, str) and content:
                    if not first_content_logged:
                        self._log_llm_first_content_elapsed(
                            request_started_at,
                            stream=True,
                        )
                        first_content_logged = True
                    sentence_buffer.append(content)
                    if (
                        not first_sentence_logged
                        and _contains_sentence_boundary(
                            "".join(sentence_buffer),
                        )
                    ):
                        self._log_llm_first_sentence_elapsed(
                            request_started_at,
                            stream=True,
                            completed_by="sentence_boundary",
                        )
                        first_sentence_logged = True
                    yield content

            if first_content_logged and not first_sentence_logged:
                self._log_llm_first_sentence_elapsed(
                    request_started_at,
                    stream=True,
                    completed_by="stream_end",
                )
            succeeded = True
        except APITimeoutError as exc:
            raise TimeoutError("OpenAI-compatible chat API timed out") from exc
        finally:
            self._log_llm_elapsed(started_at, stream=True, succeeded=succeeded)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required. Install project dependencies first."
            ) from exc

        with self._client_lock:
            if self._client is None:
                object.__setattr__(
                    self,
                    "_client",
                    OpenAI(
                        base_url=self.base_url,
                        api_key=self.api_key,
                        timeout=self.timeout_seconds,
                    ),
                )
            return self._client

    def close(self) -> None:
        client = self._client
        if client is None:
            return
        close = getattr(client, "close", None)
        if callable(close):
            close()
        object.__setattr__(self, "_client", None)

    def _create_completion_message(
        self,
        client: Any,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._log_request_messages(messages, stream=False)
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if self.extra_body is not None:
            request["extra_body"] = self.extra_body
        if tools:
            request["tools"] = tools
        if tool_choice is not None:
            request["tool_choice"] = tool_choice
        completion = client.chat.completions.create(**request)

        if not completion.choices:
            raise RuntimeError("OpenAI-compatible chat API returned no choices")
        return _message_to_dict(completion.choices[0].message)

    def _run_with_business_timeout(
        self,
        operation_func: Any,
        *,
        operation: str,
    ) -> Any:
        result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

        def worker() -> None:
            try:
                result_queue.put_nowait(("result", operation_func()))
            except Exception as exc:  # noqa: BLE001 - propagate from worker thread.
                try:
                    result_queue.put_nowait(("error", exc))
                except queue.Full:
                    pass

        thread = threading.Thread(
            target=worker,
            name=f"llm-{operation}-worker",
            daemon=True,
        )
        started_at = time.perf_counter()
        thread.start()

        try:
            kind, payload = result_queue.get(timeout=self.business_timeout_seconds)
        except queue.Empty as exc:
            self._log_llm_business_timeout(started_at, operation=operation)
            raise TimeoutError("OpenAI-compatible chat API business timeout") from exc

        if kind == "error":
            raise payload
        return payload

    def _start_stream_worker(
        self,
        client: Any,
        messages: list[dict[str, Any]],
    ) -> "_StreamEvents":
        events = _StreamEvents()

        def worker() -> None:
            stream = None
            try:
                request: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "stream": True,
                }
                if self.extra_body is not None:
                    request["extra_body"] = self.extra_body
                stream = client.chat.completions.create(**request)

                for chunk in stream:
                    if events.is_stopped:
                        break
                    if not getattr(chunk, "choices", None):
                        continue
                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", None)
                    if isinstance(content, str) and content:
                        events.put("content", content)
                events.put("done", None)
            except Exception as exc:  # noqa: BLE001 - propagate from worker thread.
                events.put("error", exc)
            finally:
                if stream is not None:
                    close = getattr(stream, "close", None)
                    if callable(close):
                        close()

        thread = threading.Thread(
            target=worker,
            name="llm-stream-worker",
            daemon=True,
        )
        thread.start()
        return events

    def build_messages(
        self,
        text: str,
        *,
        history: list[dict[str, str]] | None = None,
    ) -> list[dict[str, Any]]:
        messages = [{"role": "system", "content": self.system_prompt.strip()}]
        history_messages = [
            message
            for message in history or []
            if isinstance(message, dict)
            if message.get("role") in {"user", "assistant"}
            and isinstance(message.get("content"), str)
            and message["content"].strip()
        ]
        messages.extend(history_messages)

        current_prompt = self.build_user_prompt(text)
        messages.append(
            {
                "role": "user",
                "content": current_prompt,
            }
        )
        return messages

    def build_user_prompt(
        self,
        text: str,
    ) -> str:
        return self.prompt_layouts["user_message"].format(
            user_prompt=self.user_prompt.strip(),
            user_input_label=self.language_texts["user_input_label"],
            text=text.strip(),
        )

    def _log_llm_elapsed(
        self,
        started_at: float,
        *,
        stream: bool,
        succeeded: bool,
    ) -> None:
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        logger.info(
            "llm_call_finished model=%s stream=%s succeeded=%s elapsed_ms=%.2f",
            self.model,
            stream,
            succeeded,
            elapsed_ms,
        )

    def _log_llm_first_content_elapsed(
        self,
        request_started_at: float,
        *,
        stream: bool,
    ) -> None:
        elapsed_ms = (time.perf_counter() - request_started_at) * 1000
        logger.info(
            "llm_first_content_received model=%s stream=%s elapsed_ms=%.2f",
            self.model,
            stream,
            elapsed_ms,
        )

    def _log_llm_first_sentence_elapsed(
        self,
        request_started_at: float,
        *,
        stream: bool,
        completed_by: str,
    ) -> None:
        elapsed_ms = (time.perf_counter() - request_started_at) * 1000
        logger.info(
            "llm_first_sentence_received model=%s stream=%s completed_by=%s elapsed_ms=%.2f",
            self.model,
            stream,
            completed_by,
            elapsed_ms,
        )

    def _log_llm_business_timeout(
        self,
        request_started_at: float,
        *,
        operation: str,
    ) -> None:
        elapsed_ms = (time.perf_counter() - request_started_at) * 1000
        logger.info(
            "llm_business_timeout model=%s operation=%s business_timeout_seconds=%.2f elapsed_ms=%.2f",
            self.model,
            operation,
            self.business_timeout_seconds,
            elapsed_ms,
        )

    def _log_request_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        stream: bool,
    ) -> None:
        logger.info(
            "llm_request_messages model=%s stream=%s messages=%s",
            self.model,
            stream,
            messages,
        )


class _StreamEvents:
    def __init__(self) -> None:
        self.queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._stop_event = threading.Event()

    @property
    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    def stop(self) -> None:
        self._stop_event.set()

    def put(self, kind: str, payload: Any) -> None:
        if self.is_stopped:
            return
        self.queue.put((kind, payload))


def _message_to_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return dict(message)
    model_dump = getattr(message, "model_dump", None)
    if callable(model_dump):
        payload = model_dump(exclude_none=True)
        if isinstance(payload, dict):
            return payload
    raise RuntimeError("OpenAI-compatible chat API returned an invalid message")


def _require_message_content(message: dict[str, Any]) -> str:
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("OpenAI-compatible chat API returned empty content")
    return content


def _contains_sentence_boundary(
    text: str,
) -> bool:
    sentence_boundaries = ("。", "！", "？", "!", "?", "；", ";", "\n")
    return any(boundary in text for boundary in sentence_boundaries)
