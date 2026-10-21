"""OpenAI-compatible chat API that runs the RAG pipeline.

Open WebUI speaks this protocol. It never calls Ollama itself: every user
message becomes a `run_answer` query (retrieve → rerank → generate).
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from psycopg.errors import Error as PsycopgError
from pydantic import BaseModel, Field

from app.retrieval.pipeline import prepare_answer, run_answer
from app.retrieval.utils.generate import LLMError, iter_chat_tokens, open_chat_stream

# Single model id advertised to Open WebUI. The name is a label; generation
# still uses the Compose `llm` model inside `run_answer`.
RAG_MODEL_ID = "rag"

router = APIRouter(prefix="/v1", tags=["openai"])


class ChatMessage(BaseModel):
    """One turn in an OpenAI-style messages array."""

    role: str
    content: str | list[Any] | None = None


class ChatCompletionRequest(BaseModel):
    """Subset of the OpenAI chat completions body. Extra fields are ignored."""

    model: str = RAG_MODEL_ID
    messages: list[ChatMessage] = Field(default_factory=list)
    stream: bool = False


def message_text(content: str | list[Any] | None) -> str:
    """Flatten string or multimodal content parts into plain text."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text") or ""))
    return "\n".join(part for part in parts if part.strip())


def last_user_query(messages: list[ChatMessage]) -> str:
    """Question for retrieval: the latest non-empty user turn."""
    for message in reversed(messages):
        if message.role != "user":
            continue
        text = message_text(message.content).strip()
        if text:
            return text
    raise ValueError("messages must include a user question")


def append_sources(answer: str, citations: list[dict[str, Any]]) -> str:
    """Add a short source footer so citations are visible in the chat bubble."""
    lines: list[str] = []
    for citation in citations:
        ref = citation.get("ref")
        title = citation.get("title") or "Document"
        source = citation.get("source_path") or ""
        label = f"[{ref}] {title}"
        if source:
            label = f"{label} — {source}"
        lines.append(label)
    if not lines:
        return answer
    return f"{answer.rstrip()}\n\nSources:\n" + "\n".join(lines)


def completion_payload(*, text: str, model: str) -> dict[str, Any]:
    """Non-streaming chat.completion object."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model or RAG_MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
    }


def completion_token_events(
    upstream: Any,
    *,
    citations: list[dict[str, Any]],
    model: str,
) -> Iterator[str]:
    """SSE stream: one delta per Ollama token, then the sources footer.

    Retrieval already finished. Each yield is a piece of the answer so the
    chat UI can paint tokens as they arrive.
    """
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    model_name = model or RAG_MODEL_ID
    try:
        for piece in iter_chat_tokens(upstream):
            yield _sse(
                _chunk(
                    completion_id=completion_id,
                    created=created,
                    model=model_name,
                    delta={"content": piece},
                    finish_reason=None,
                )
            )
    except LLMError:
        # Headers are already sent. Finish the SSE body instead of raising.
        pass
    finally:
        footer = append_sources("", citations)
        if footer.strip():
            yield _sse(
                _chunk(
                    completion_id=completion_id,
                    created=created,
                    model=model_name,
                    delta={"content": footer},
                    finish_reason=None,
                )
            )
        yield _sse(
            _chunk(
                completion_id=completion_id,
                created=created,
                model=model_name,
                delta={},
                finish_reason="stop",
            )
        )
        yield "data: [DONE]\n\n"


def _chunk(
    *,
    completion_id: str,
    created: int,
    model: str,
    delta: dict[str, Any],
    finish_reason: str | None,
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.get("/models")
def list_models() -> dict[str, Any]:
    """Advertise the RAG pipeline as the only chat model."""
    return {
        "object": "list",
        "data": [
            {
                "id": RAG_MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "software-doc-search",
            }
        ],
    }


@router.post("/chat/completions")
def chat_completions(body: ChatCompletionRequest):
    """Run grounded RAG for the latest user message.

    stream=false waits for the full answer (same as POST /answer/).
    stream=true retrieves first, then forwards Ollama tokens as SSE.
    """
    try:
        query = last_user_query(body.messages)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    model = body.model or RAG_MODEL_ID
    if not body.stream:
        try:
            result = run_answer(query=query)
        except LLMError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except (PsycopgError, OSError) as exc:
            raise HTTPException(status_code=503, detail=f"Upstream error: {exc}") from exc
        text = append_sources(result["answer"], result.get("citations") or [])
        return completion_payload(text=text, model=model)

    try:
        prepared = prepare_answer(query=query)
        upstream = open_chat_stream(user_prompt=prepared["prompt"])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (PsycopgError, OSError) as exc:
        raise HTTPException(status_code=503, detail=f"Upstream error: {exc}") from exc

    return StreamingResponse(
        completion_token_events(
            upstream,
            citations=prepared["citations"],
            model=model,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
