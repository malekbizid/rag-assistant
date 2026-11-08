"""Pipeline RAG : recherche des extraits, génération d'une réponse citée, mesure latence / tokens / coût.

Usage :  python -m ragassist.pipeline --config configs/default.yaml "Ma question ?"
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .config import ModelSpec, RAGConfig, load_config
from .ingest import build_index
from .llm import make_chat, message_text

REFUSAL = "Je ne trouve pas cette information dans les documents."

SYSTEM_PROMPT = f"""Tu es un assistant qui répond à des questions sur un corpus de documents.
Règles :
- Réponds UNIQUEMENT à partir des extraits fournis, jamais de tes connaissances générales.
- Après chaque affirmation, cite l'extrait qui la justifie par son numéro entre crochets, ex. [1] ou [2][3].
- Si les extraits ne permettent pas de répondre, réponds exactement : « {REFUSAL} »
- Réponds dans la langue de la question, de façon concise et précise (chiffres, dates, conditions)."""


def format_context(docs: list[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, start=1):
        page = d.metadata.get("page", 0)
        loc = f", p. {page}" if page else ""
        parts.append(f"[{i}] (source : {d.metadata.get('source', '?')}{loc})\n{d.page_content}")
    return "\n\n".join(parts)


def tracing_callbacks() -> list[Any]:
    """Callback Langfuse si les variables LANGFUSE_* sont définies, sinon aucun."""
    if not os.getenv("LANGFUSE_PUBLIC_KEY"):
        return []
    try:
        from langfuse.langchain import CallbackHandler  # langfuse >= 3
    except ImportError:
        try:
            from langfuse.callback import CallbackHandler  # langfuse 2.x
        except ImportError:
            return []
    return [CallbackHandler()]


def cost_usd(spec: ModelSpec, tokens_in: int, tokens_out: int) -> float | None:
    if spec.price_in_per_1m is None or spec.price_out_per_1m is None:
        return None
    return (tokens_in * spec.price_in_per_1m + tokens_out * spec.price_out_per_1m) / 1_000_000


@dataclass
class RAGAnswer:
    question: str
    answer: str
    sources: list[dict] = field(default_factory=list)
    retrieval_s: float = 0.0
    generation_s: float = 0.0
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_estimated: bool = False
    cost_usd: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class RAGPipeline:
    def __init__(self, cfg: RAGConfig, vectorstore: Any = None, llm: BaseChatModel | None = None):
        self.cfg = cfg
        self.vs = vectorstore if vectorstore is not None else build_index(cfg, verbose=False)
        self.llm = llm if llm is not None else make_chat(cfg.llm)
        self.callbacks = tracing_callbacks()

    def retrieve(self, question: str) -> list[Document]:
        k = self.cfg.top_k
        if self.cfg.search_type == "mmr":
            return self.vs.max_marginal_relevance_search(question, k=k, fetch_k=max(20, 4 * k))
        return self.vs.similarity_search(question, k=k)

    def ask(self, question: str) -> RAGAnswer:
        t0 = time.perf_counter()
        docs = self.retrieve(question)
        t1 = time.perf_counter()

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Extraits :\n\n{format_context(docs)}\n\nQuestion : {question}"),
        ]
        msg = self.llm.invoke(messages, config={"callbacks": self.callbacks, "run_name": self.cfg.name})
        t2 = time.perf_counter()

        answer = message_text(msg).strip()
        usage = getattr(msg, "usage_metadata", None) or {}
        estimated = not usage
        t_in = usage.get("input_tokens") or sum(len(message_text(m)) for m in messages) // 4
        t_out = usage.get("output_tokens") or len(answer) // 4

        sources = [
            {"rank": i, "source": d.metadata.get("source"), "page": d.metadata.get("page", 0),
             "chunk_id": d.metadata.get("chunk_id"), "text": d.page_content}
            for i, d in enumerate(docs, start=1)
        ]
        return RAGAnswer(
            question=question, answer=answer, sources=sources,
            retrieval_s=t1 - t0, generation_s=t2 - t1, latency_s=t2 - t0,
            input_tokens=int(t_in), output_tokens=int(t_out), tokens_estimated=estimated,
            cost_usd=cost_usd(self.cfg.llm, int(t_in), int(t_out)),
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Pose une question au corpus.")
    ap.add_argument("question")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    res = RAGPipeline(load_config(args.config)).ask(args.question)
    print(res.answer, "\n")
    for s in res.sources:
        loc = f" p.{s['page']}" if s["page"] else ""
        print(f"  [{s['rank']}] {s['source']}{loc}")
    cost = f" · {res.cost_usd:.5f} $" if res.cost_usd is not None else ""
    print(f"\n{res.latency_s:.2f} s · {res.input_tokens}+{res.output_tokens} tokens{cost}")


if __name__ == "__main__":
    main()
