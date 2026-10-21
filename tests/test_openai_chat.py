from fastapi.testclient import TestClient

from app.api.utils.openai_chat import (
    RAG_MODEL_ID,
    ChatMessage,
    append_sources,
    last_user_query,
    message_text,
)
from app.main import app

client = TestClient(app)


def test_message_text_flattens_parts():
    assert message_text("hello") == "hello"
    assert message_text([{"type": "text", "text": "connect"}, {"type": "image_url", "image_url": {}}]) == "connect"


def test_last_user_query_uses_latest_user_turn():
    messages = [
        ChatMessage(role="user", content="older question"),
        ChatMessage(role="assistant", content="older answer"),
        ChatMessage(role="user", content="how do I connect"),
    ]
    assert last_user_query(messages) == "how do I connect"


def test_append_sources_adds_footer():
    text = append_sources(
        "Use a connection URI.",
        [{"ref": 1, "title": "Connections", "source_path": "raw/pg.pdf"}],
    )
    assert text.startswith("Use a connection URI.")
    assert "Sources:" in text
    assert "[1] Connections — raw/pg.pdf" in text


def test_list_models_advertises_rag():
    response = client.get("/v1/models")
    assert response.status_code == 200
    ids = [item["id"] for item in response.json()["data"]]
    assert ids == [RAG_MODEL_ID]


def test_chat_completions_runs_rag(monkeypatch):
    def fake_run_answer(*, query: str):
        assert query == "how do I connect"
        return {
            "answer": "Use a connection URI.",
            "citations": [
                {"ref": 1, "title": "Connections", "source_path": "raw/pg.pdf"}
            ],
        }

    monkeypatch.setattr("app.api.utils.openai_chat.run_answer", fake_run_answer)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": RAG_MODEL_ID,
            "messages": [
                {"role": "system", "content": "ignored"},
                {"role": "user", "content": "how do I connect"},
            ],
        },
    )
    assert response.status_code == 200
    body = response.json()
    content = body["choices"][0]["message"]["content"]
    assert "Use a connection URI." in content
    assert "[1] Connections — raw/pg.pdf" in content
    assert body["model"] == RAG_MODEL_ID


def test_chat_completions_streams_ollama_tokens(monkeypatch):
    monkeypatch.setattr(
        "app.api.utils.openai_chat.prepare_answer",
        lambda *, query: {
            "prompt": "grounded prompt",
            "citations": [
                {"ref": 1, "title": "Connections", "source_path": "raw/pg.pdf"}
            ],
        },
    )

    class _Upstream:
        def __iter__(self):
            return iter(
                [
                    b'{"message":{"content":"Use"},"done":false}\n',
                    b'{"message":{"content":" a URI"},"done":true}\n',
                ]
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        "app.api.utils.openai_chat.open_chat_stream",
        lambda **kwargs: _Upstream(),
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": RAG_MODEL_ID,
            "stream": True,
            "messages": [{"role": "user", "content": "question"}],
        },
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert '"content": "Use"' in response.text
    assert '"content": " a URI"' in response.text
    assert "Sources:" in response.text
    assert "[1] Connections — raw/pg.pdf" in response.text
    assert response.text.strip().endswith("data: [DONE]")


def test_chat_completions_requires_a_user_message():
    response = client.post(
        "/v1/chat/completions",
        json={"model": RAG_MODEL_ID, "messages": [{"role": "system", "content": "hi"}]},
    )
    assert response.status_code == 400
