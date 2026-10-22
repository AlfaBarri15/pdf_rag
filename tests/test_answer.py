import io
import json
import urllib.error
import urllib.request

from psycopg import connect
from psycopg.errors import OperationalError

import pytest

from app.config import DATABASE_URL, OLLAMA_BASE_URL
from app.retrieval.pipeline import run_answer
from app.retrieval.utils.generate import LLMError, chat


def _db_available() -> bool:
    try:
        with connect(DATABASE_URL, connect_timeout=2) as conn:
            row = conn.execute(
                "SELECT count(*) FROM chunks WHERE embedding IS NOT NULL"
            ).fetchone()
            return row is not None and row[0] > 0
    except (OperationalError, OSError):
        return False


def _llm_available() -> bool:
    try:
        with urllib.request.urlopen(
            f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags",
            timeout=3,
        ) as response:
            body = json.loads(response.read().decode("utf-8"))
        return bool(body.get("models"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return False


def test_chat_rejects_empty_prompt():
    with pytest.raises(ValueError):
        chat(user_prompt="   ")


@pytest.mark.skipif(not _llm_available(), reason="LLM service not reachable")
def test_chat_returns_answer():
    result = chat(user_prompt="Reply with exactly: ok")
    assert result["answer"]
    assert result["model"]


@pytest.mark.skipif(
    not (_db_available() and _llm_available()),
    reason="Postgres embeddings or LLM not reachable",
)
def test_run_answer_returns_citations():
    payload = run_answer(
        query="what is postgresql",
        top_k=5,
        rerank=False,
        top_n=2,
    )
    assert payload["answer"]
    assert payload["result_count"] >= 1
    assert len(payload["citations"]) == payload["result_count"]
    assert payload["citations"][0]["ref"] == 1
    assert "chunk_id" in payload["citations"][0]


def test_llm_error_type_exists():
    assert issubclass(LLMError, RuntimeError)


def test_chat_retries_http_500(monkeypatch):
    calls = {"n": 0}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {"model": "llama3.2", "message": {"content": "ok"}}
            ).encode()

    def fake_urlopen(request, timeout=0):
        calls["n"] += 1
        if calls["n"] < 2:
            raise urllib.error.HTTPError(
                "http://llm/api/chat",
                500,
                "Internal Server Error",
                hdrs=None,
                fp=io.BytesIO(b'{"error":"model runner has unexpectedly stopped"}'),
            )
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr("app.retrieval.utils.generate.time.sleep", lambda _seconds: None)
    result = chat(user_prompt="hello")
    assert result["answer"] == "ok"
    assert calls["n"] == 2
