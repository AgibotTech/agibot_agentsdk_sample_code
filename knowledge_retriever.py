from __future__ import annotations

import hashlib
import hmac
import json
import logging
import queue
import random
import string
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KnowledgeChunk:
    content: str
    question: str | None = None
    answer: str | None = None
    content_type: str | None = None
    filename: str | None = None
    score: float | None = None

    def to_prompt_text(
        self,
        index: int,
        texts: dict[str, str],
        layouts: dict[str, str],
    ) -> str:
        source = self.filename or ""
        score = f"{self.score:.2f}" if self.score is not None else ""
        if self.is_qa:
            return layouts["qa_item"].format(
                index=index,
                type=texts["qa_type"],
                question_label=texts["question_label"],
                question=self.question or "",
                answer_label=texts["answer_label"],
                answer=self.answer or "",
                source_label=texts["source_label"],
                source=source,
                score_label=texts["score_label"],
                score=score,
            )
        return layouts["chunk_item"].format(
            index=index,
            type=texts["chunk_type"],
            content_label=texts["content_label"],
            content=self.content,
            source_label=texts["source_label"],
            source=source,
            score_label=texts["score_label"],
            score=score,
        )

    @property
    def is_qa(self) -> bool:
        return bool(self.question or self.answer)


@dataclass(frozen=True)
class KnowledgeRetriever:
    app_id: str
    app_key: str
    app_secret: str
    agent_code: str
    api_url: str
    context_prompt: str
    language_texts: dict[str, str]
    prompt_layouts: dict[str, str]
    top_k: int = 10
    rerank: bool = True
    check_qa_first: bool = True
    score_threshold: float = 0.0
    timeout_seconds: float = 5.0
    business_timeout_seconds: float = 5.0

    @classmethod
    def from_config(
        cls,
        config: Any,
        agent: Any,
        language_texts: dict[str, str],
        prompt_layouts: dict[str, str],
    ) -> "KnowledgeRetriever":
        api_url = getattr(config, "api_url", None)
        if not isinstance(api_url, str) or not api_url.strip():
            raise ValueError("knowledge_base.api_url is required when enabled")
        context_prompt = getattr(config, "context_prompt", None)
        if not isinstance(context_prompt, str) or not context_prompt.strip():
            raise ValueError("knowledge_base.context_prompt is required when enabled")
        return cls(
            app_id=agent.app_id,
            app_key=agent.app_key,
            app_secret=agent.app_secret,
            agent_code=agent.agent_code,
            api_url=api_url,
            context_prompt=context_prompt,
            language_texts=language_texts,
            prompt_layouts=prompt_layouts,
            top_k=getattr(config, "top_k", 10),
            rerank=getattr(config, "rerank", True),
            check_qa_first=getattr(config, "check_qa_first", True),
            score_threshold=getattr(config, "score_threshold", 0.0),
            timeout_seconds=getattr(config, "timeout_seconds", 5.0),
            business_timeout_seconds=getattr(
                config,
                "business_timeout_seconds",
                getattr(config, "timeout_seconds", 5.0),
            ),
        )

    def retrieve(self, query: str) -> list[KnowledgeChunk]:
        started_at = time.perf_counter()
        returned_count = 0
        succeeded = False
        try:
            chunks = self._retrieve_with_business_timeout(query)
            returned_count = len(chunks)
            succeeded = True
            return chunks
        finally:
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            logger.info(
                "knowledge_retrieval_finished agent_code=%s top_k=%s returned=%s succeeded=%s elapsed_ms=%.2f",
                self.agent_code,
                self.top_k,
                returned_count,
                succeeded,
                elapsed_ms,
            )

    def _retrieve_with_business_timeout(self, query: str) -> list[KnowledgeChunk]:
        result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

        def worker() -> None:
            try:
                result_queue.put_nowait(("result", self._retrieve_once(query)))
            except Exception as exc:  # noqa: BLE001 - propagate from worker thread.
                try:
                    result_queue.put_nowait(("error", exc))
                except queue.Full:
                    pass

        thread = threading.Thread(
            target=worker,
            name="knowledge-retrieval-worker",
            daemon=True,
        )
        thread.start()

        try:
            kind, payload = result_queue.get(timeout=self.business_timeout_seconds)
        except queue.Empty as exc:
            logger.info(
                "knowledge_retrieval_business_timeout agent_code=%s top_k=%s business_timeout_seconds=%.2f",
                self.agent_code,
                self.top_k,
                self.business_timeout_seconds,
            )
            raise TimeoutError("Knowledge-base retrieval business timeout") from exc

        if kind == "error":
            raise payload
        return payload

    def _retrieve_once(self, query: str) -> list[KnowledgeChunk]:
        body = {
            "agent_code": self.agent_code,
            "query": query,
            "top_k": self.top_k,
            "rerank": self.rerank,
            "check_qa_first": self.check_qa_first,
            "score_threshold": self.score_threshold,
        }
        payload = self._post(body)
        data = payload.get("data")
        if not payload.get("success") or not isinstance(data, list):
            raise RuntimeError(
                f"Knowledge-base search failed: {payload.get('message') or payload}"
            )

        chunks: list[KnowledgeChunk] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            chunk = _parse_chunk(item)
            if chunk is not None:
                chunks.append(chunk)
        return chunks

    def retrieve_context(self, query: str) -> str:
        chunks = _deduplicate_chunks(self.retrieve(query))
        if not chunks:
            logger.info(
                "knowledge_retrieval_empty agent_code=%s query=%s",
                self.agent_code,
                query,
            )
            return ""

        sections = [self.context_prompt.strip()]
        sections.extend(
            chunk.to_prompt_text(
                index,
                self.language_texts,
                self.prompt_layouts,
            )
            for index, chunk in enumerate(chunks, start=1)
        )
        context = "\n\n".join(sections)
        logger.info(
            "knowledge_retrieval_context agent_code=%s query=%s context=%s",
            self.agent_code,
            query,
            context,
        )
        return context

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        path = urllib.parse.urlparse(self.api_url).path
        ts_ms = int(time.time() * 1000)
        nonce = _nonce()
        signature = _sign("POST", path, ts_ms, nonce, self.app_secret)

        data = json.dumps(body).encode()
        request = urllib.request.Request(self.api_url, data=data, method="POST")
        request.add_header("Content-Type", "application/json")
        request.add_header("X-App-Id", self.app_id)
        request.add_header("X-App-Key", self.app_key)
        request.add_header("X-Timestamp", str(ts_ms))
        request.add_header("X-Nonce", nonce)
        request.add_header("X-Signature", signature)

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw_body = response.read().decode()
        except urllib.error.HTTPError as exc:
            raw_body = exc.read().decode()
            raise RuntimeError(
                f"Knowledge-base HTTP {exc.code}: {raw_body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Knowledge-base request failed: {exc}") from exc

        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Knowledge-base returned invalid JSON: {raw_body}"
            ) from exc

        if not isinstance(payload, dict):
            raise RuntimeError(f"Knowledge-base returned non-object JSON: {payload}")
        return payload


def _nonce(length: int = 16) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def _sign(method: str, path: str, ts_ms: int, nonce: str, secret: str) -> str:
    payload = f"{method.upper()}\n{path}\n{ts_ms}\n{nonce}"
    return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


def _parse_chunk(item: dict[str, Any]) -> KnowledgeChunk | None:
    content = _clean_text(item.get("content"))
    question = _clean_text(item.get("question"))
    answer = _normalize_answer(item.get("answer"))
    content_type = _clean_text(item.get("content_type")) or _metadata_text(
        item,
        "content_type",
    )
    filename = _clean_text(item.get("filename")) or _metadata_text(item, "filename")
    score = _safe_float(item.get("score"))

    if not content and not question and not answer:
        return None

    return KnowledgeChunk(
        content=content or question or answer or "",
        question=question,
        answer=answer,
        content_type=content_type,
        filename=filename,
        score=score,
    )


def _deduplicate_chunks(chunks: list[KnowledgeChunk]) -> list[KnowledgeChunk]:
    unique_chunks: list[KnowledgeChunk] = []
    seen_content: set[tuple[str, str]] = set()
    for chunk in chunks:
        if chunk.answer:
            key = ("answer", chunk.answer.strip())
        else:
            key = ("content", chunk.content.strip())
        if key in seen_content:
            continue
        seen_content.add(key)
        unique_chunks.append(chunk)
    return unique_chunks


def _normalize_answer(value: Any) -> str | None:
    text = _clean_text(value)
    if not text:
        return None

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text

    if isinstance(payload, dict):
        behavior_texts = []
        behaviors = payload.get("behaviors")
        if isinstance(behaviors, list):
            for behavior in behaviors:
                if isinstance(behavior, dict):
                    behavior_text = _clean_text(behavior.get("text"))
                    if behavior_text:
                        behavior_texts.append(behavior_text)
        if behavior_texts:
            return "\n".join(behavior_texts)
        return json.dumps(payload, ensure_ascii=False)

    if isinstance(payload, list):
        return json.dumps(payload, ensure_ascii=False)
    return text


def _metadata_text(item: dict[str, Any], key: str) -> str | None:
    metadata = item.get("metadata")
    if not isinstance(metadata, dict):
        return None
    return _clean_text(metadata.get(key))


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
