"""API REST.  Lancement :  uvicorn ragassist.api:app --reload
Configuration choisie via la variable d'environnement RAG_CONFIG (défaut : configs/default.yaml).
"""
from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import load_config
from .pipeline import RAGPipeline

app = FastAPI(title="Assistant RAG documentaire", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class Source(BaseModel):
    rank: int
    source: str | None
    page: int
    chunk_id: str | None
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    latency_s: float
    input_tokens: int
    output_tokens: int
    cost_usd: float | None


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline(load_config(os.getenv("RAG_CONFIG", "configs/default.yaml")))


@app.get("/health")
def health() -> dict:
    cfg = get_pipeline().cfg
    return {"status": "ok", "config": cfg.name, "llm": f"{cfg.llm.provider}:{cfg.llm.name}",
            "chunks": get_pipeline().vs._collection.count()}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        res = get_pipeline().ask(req.question)
    except Exception as e:  # erreur fournisseur (quota, clé invalide…)
        raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}") from e
    return AskResponse(answer=res.answer, sources=[Source(**s) for s in res.sources],
                       latency_s=res.latency_s, input_tokens=res.input_tokens,
                       output_tokens=res.output_tokens, cost_usd=res.cost_usd)
