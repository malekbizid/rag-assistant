"""Configuration du pipeline RAG (chargée depuis un fichier YAML)."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel

try:  # charge les clés d'API depuis .env si présent
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

Provider = Literal["gemini", "openai", "ollama", "fake", "hash"]


class ModelSpec(BaseModel):
    """Un modèle (LLM ou embeddings) et, optionnellement, son tarif."""

    provider: Provider
    name: str = ""
    temperature: float = 0.0
    # Tarifs en $ par million de tokens — à renseigner depuis la page tarifs du fournisseur.
    price_in_per_1m: float | None = None
    price_out_per_1m: float | None = None


class RAGConfig(BaseModel):
    name: str = "default"
    data_dir: str = "data/raw"
    index_dir: str = "data/index"
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4
    search_type: Literal["similarity", "mmr"] = "similarity"
    embedding: ModelSpec
    llm: ModelSpec

    def index_key(self) -> str:
        """Identifiant d'index : ne dépend que du corpus, des embeddings et du découpage.

        Deux expériences qui ne diffèrent que par le LLM ou top_k réutilisent donc le même index.
        """
        payload = {
            "data": str(Path(self.data_dir).resolve()),
            "emb": [self.embedding.provider, self.embedding.name],
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]
        return f"idx_{digest}"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(path: str | Path) -> RAGConfig:
    return RAGConfig(**_read_yaml(path))


def load_experiments(path: str | Path) -> tuple[list[RAGConfig], ModelSpec | None]:
    """Lit un fichier d'expériences : `base` + liste d'`experiments` (surcharges) + `judge`."""
    data = _read_yaml(path)
    base = data.get("base", {})
    configs = [RAGConfig(**deep_merge(base, exp)) for exp in data.get("experiments", [])]
    names = [c.name for c in configs]
    if len(names) != len(set(names)):
        raise ValueError(f"Noms d'expériences en double : {names}")
    judge = ModelSpec(**data["judge"]) if data.get("judge") else None
    return configs, judge
