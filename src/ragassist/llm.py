"""Fabrique de modèles : un seul point d'entrée pour Gemini, OpenAI, Ollama et les modèles hors-ligne.

Les fournisseurs « fake » (chat) et « hash » (embeddings) tournent sans clé d'API :
- `hash` est un sac de mots haché — une vraie baseline lexicale, utile pour comparer aux embeddings denses ;
- `fake` est un LLM factice qui sert uniquement aux tests et à la démo hors-ligne.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import unicodedata
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from .config import ModelSpec

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def make_chat(spec: ModelSpec) -> BaseChatModel:
    p = spec.provider
    if p == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=spec.name, temperature=spec.temperature)
    if p == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=spec.name, temperature=spec.temperature)
    if p == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=spec.name, temperature=spec.temperature, base_url=OLLAMA_URL)
    if p == "fake":
        return FakeExtractiveChat()
    raise ValueError(f"Fournisseur de chat non supporté : {p}")


def make_embeddings(spec: ModelSpec) -> Embeddings:
    p = spec.provider
    if p == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=spec.name)
    if p == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=spec.name)
    if p == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=spec.name, base_url=OLLAMA_URL)
    if p == "hash":
        return HashingEmbeddings()
    raise ValueError(f"Fournisseur d'embeddings non supporté : {p}")


def message_text(msg: BaseMessage) -> str:
    """Texte d'un message, que `content` soit une chaîne ou une liste de blocs (cas de Gemini)."""
    content = msg.content
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


# --------------------------------------------------------------------------- hors-ligne

_STOPWORDS = set(
    "le la les un une des de du d l et ou a au aux en dans par pour sur avec sans ce cet cette ces "
    "est sont etre qui que quoi dont ou il elle ils elles on nous vous je tu se sa son ses leur leurs "
    "ne pas plus tout tous toute toutes y the of and to in is are for on with".split()
)


def _tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return [t for t in re.findall(r"[a-z0-9]+", text) if t not in _STOPWORDS and len(t) > 1]


class HashingEmbeddings(Embeddings):
    """Sac de mots haché (dimension fixe), normalisé L2. Déterministe, sans réseau."""

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _tokens(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class FakeExtractiveChat(BaseChatModel):
    """LLM factice : recopie la 1re phrase de l'extrait [1]. Répond aussi aux prompts juge/générateur."""

    @property
    def _llm_type(self) -> str:
        return "fake-extractive"

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any) -> ChatResult:
        prompt = message_text(messages[-1])  # le message utilisateur, sans le prompt système
        if "ÉVALUATEUR" in prompt:
            text = json.dumps({"faithfulness": 1, "correctness": 0.5, "justification": "évaluation factice"})
        elif "GÉNÉRATEUR" in prompt:
            extract = prompt.split("<<<", 1)[-1].split(">>>", 1)[0].strip()
            sentence = re.split(r"(?<=[.!?])\s+", extract)[0]
            text = json.dumps(
                {"question": f"Que dit le document à propos de : {sentence[:60]}… ?",
                 "reference_answer": sentence, "evidence": sentence},
                ensure_ascii=False,
            )
        else:
            m = re.search(r"\[1\][^\n]*\n(.+?)(?:\n\[2\]|\Z)", prompt, re.S)
            body = " ".join(l for l in (m.group(1) if m else "").splitlines() if l and not l.startswith("#"))
            first = next((s for s in re.split(r"(?<=[.!?])\s+", body) if len(s) > 20), "")
            text = f"{first} [1]" if first else "Je ne trouve pas cette information dans les documents."
        usage = {"input_tokens": len(prompt) // 4, "output_tokens": len(text) // 4,
                 "total_tokens": (len(prompt) + len(text)) // 4}
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text, usage_metadata=usage))])
