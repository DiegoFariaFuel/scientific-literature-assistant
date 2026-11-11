from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .answer import answer_question
from .retrieval import HybridRetriever

app = FastAPI(title="SciLit Evidence Assistant", version="0.1.0")


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)


@lru_cache(maxsize=1)
def _retriever() -> HybridRetriever:
    return HybridRetriever(os.environ.get("SCILIT_INDEX_DIR", "index"), os.environ.get("SCILIT_RERANK_MODEL"))


@app.get("/health")
def health():
    return {"status": "ready" if os.path.exists(os.path.join(os.environ.get("SCILIT_INDEX_DIR", "index"), "manifest.json"))
            else "index_missing"}


@app.post("/ask")
def ask(payload: Question):
    try:
        return answer_question(_retriever(), payload.question, payload.top_k)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        if exc.__class__.__module__.startswith("requests"):
            raise HTTPException(status_code=502, detail="could not reach the local LLM server") from exc
        raise


def main() -> None:
    import uvicorn
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8080")))

