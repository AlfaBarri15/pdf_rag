"""Call the LLM service (Ollama) for grounded chat generation."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL

DEFAULT_SYSTEM_PROMPT = (
    "You are a grounded extractor. Use ONLY the numbered sources in the user message. "
    "Start with the direct answer in one or two short sentences, then cite like [1]. "
    "When the question asks for a figure, find the named line item in the sources "
    "(for example Purchases of property, plant and equipment / PP&E, not a made-up acronym) "
    "and copy that number. Prefer the audited statement row over outlook, guidance, or ranges. "
    "Do not invent arithmetic from other cash-flow or balance-sheet totals. "
    "Use the units the question asked for. If the table is in millions and the question asks "
    "for billions, divide by 1000. Money uses $ and the precision the question used "
    "(for example $1577.00 or $8.70). "
    "If the sources do not contain the answer, say you do not know. Do not guess."
)

FREEFORM_SYSTEM_PROMPT = (
    "You are a helpful assistant . Answer clearly and concisely in a single "
    "paragraph. Do not use line breaks, bullet lists, or markdown headings."
)


class LLMError(RuntimeError):
    """Raised when the LLM service is unreachable or returns an error."""


def chat(
    *,
    user_prompt: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    model: str = OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    timeout_seconds: float = 120.0,
    retries: int = 3,
) -> dict[str, Any]:
    """POST /api/chat on Ollama with stream=false; returns {model, answer, raw}."""
    last_error: LLMError | None = None
    attempts = max(1, retries)
    for attempt in range(1, attempts + 1):
        request = _chat_request(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            model=model,
            base_url=base_url,
            stream=False,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = _llm_http_error(exc)
            if exc.code < 500 or attempt == attempts:
                raise last_error from exc
            time.sleep(min(10 * attempt, 30))
            continue
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last_error = LLMError(f"LLM request failed: {exc}")
            if attempt == attempts:
                raise last_error from exc
            time.sleep(min(10 * attempt, 30))
            continue

        answer = _message_content(body).strip()
        if not answer:
            last_error = LLMError("LLM returned an empty answer")
            if attempt == attempts:
                raise last_error
            time.sleep(min(10 * attempt, 30))
            continue

        return {
            "model": body.get("model") or model,
            "answer": answer,
            "raw": body,
        }

    raise last_error or LLMError("LLM request failed")


def open_chat_stream(
    *,
    user_prompt: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    model: str = OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    timeout_seconds: float = 120.0,
) -> Any:
    """Open Ollama /api/chat with stream=true. Caller reads it via iter_chat_tokens."""
    request = _chat_request(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        model=model,
        base_url=base_url,
        stream=True,
    )
    try:
        return urllib.request.urlopen(request, timeout=timeout_seconds)
    except urllib.error.HTTPError as exc:
        raise _llm_http_error(exc) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LLMError(f"LLM request failed: {exc}") from exc


def iter_chat_tokens(response: Any) -> Iterator[str]:
    """Yield each Ollama content fragment. Closes the response when finished."""
    try:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                body = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LLMError(f"LLM stream line was not JSON: {exc}") from exc
            piece = _message_content(body)
            if piece:
                yield piece
            if body.get("done"):
                break
    finally:
        response.close()


def _chat_request(
    *,
    user_prompt: str,
    system_prompt: str,
    model: str,
    base_url: str,
    stream: bool,
) -> urllib.request.Request:
    """POST body for Ollama `/api/chat` at temperature 0."""
    cleaned = user_prompt.strip()
    if not cleaned:
        raise ValueError("user_prompt must not be empty")

    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "stream": stream,
        "options": {"temperature": 0, "num_predict": 256},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": cleaned},
        ],
    }
    return urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )


def _message_content(body: dict[str, Any]) -> str:
    """Assistant text from one Ollama chat JSON object."""
    message = body.get("message") or {}
    return message.get("content") or ""


def _llm_http_error(exc: urllib.error.HTTPError) -> LLMError:
    """Turn an HTTP error body into LLMError."""
    detail = exc.read().decode("utf-8", errors="replace")
    return LLMError(f"LLM HTTP {exc.code}: {detail}")
