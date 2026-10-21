from pydantic import BaseModel, Field

from fastapi import APIRouter, HTTPException

from app.retrieval.utils.generate import FREEFORM_SYSTEM_PROMPT, LLMError, chat

router = APIRouter(prefix="/chat", tags=["chat"])


def _single_line(text: str) -> str:
    """Collapse newlines/whitespace so JSON answers stay one line."""
    return " ".join(text.split())


class ChatRequest(BaseModel):
    """Simple chat body: send a message, get an answer back."""

    message: str = Field(..., min_length=1, description="User message to the LLM")
    system: str | None = Field(
        None,
        description="Optional system prompt (defaults to a freeform assistant)",
    )
    single_line: bool = Field(
        True,
        description="If true, collapse newlines/whitespace in the answer",
    )


@router.post("/")
def chat_with_llm(body: ChatRequest):
    """Proxy a single-turn chat to the Compose `llm` service (no retrieval)."""
    try:
        result = chat(
            user_prompt=body.message,
            system_prompt=(body.system or FREEFORM_SYSTEM_PROMPT).strip()
            or FREEFORM_SYSTEM_PROMPT,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    answer = result["answer"]
    if body.single_line:
        answer = _single_line(answer)

    return {
        "message": body.message.strip(),
        "answer": answer,
        "model": result["model"],
    }
