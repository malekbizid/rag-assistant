"""Métriques déterministes : retrieval (hit@k, MRR, rappel des preuves), citations, refus.

Pourquoi des « preuves » textuelles plutôt que des identifiants de chunks ?
Les chunks changent dès qu'on modifie chunk_size : un identifiant annoté pour une configuration
ne vaut plus rien pour la suivante. On annote donc chaque question avec la (ou les) phrase(s)
du document qui justifient la réponse, et un chunk est pertinent s'il contient cette phrase.
Le même jeu de questions sert ainsi à comparer toutes les stratégies de découpage.
"""
from __future__ import annotations

import re
import unicodedata

from .pipeline import REFUSAL

CITATION_RE = re.compile(r"\[(\d+)\]")
REFUSAL_MARKERS = ["je ne trouve pas cette information", "i cannot find this information"]


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = text.replace("’", "'").replace("«", '"').replace("»", '"')
    return re.sub(r"\s+", " ", text).strip()


def evidence_in_text(evidence: str, text: str, probe: int = 60) -> bool:
    """La preuve est dans le chunk si elle y figure en entier, ou (preuve longue coupée entre deux
    chunks) si son début ou sa fin y figure."""
    e, t = norm(evidence), norm(text)
    if not e:
        return False
    if e in t:
        return True
    return len(e) > probe and (e[:probe] in t or e[-probe:] in t)


def chunk_is_relevant(source: dict, item: dict) -> bool:
    evidence = item.get("evidence") or []
    if evidence:
        return any(evidence_in_text(e, source["text"]) for e in evidence)
    # Repli si aucune preuve textuelle n'est annotée : fichier + page.
    pages = item.get("pages") or []
    return bool(item.get("source")) and source.get("source") == item["source"] and source.get("page") in pages


def retrieval_metrics(sources: list[dict], item: dict) -> dict:
    """Seulement pour les questions qui ont une réponse dans le corpus."""
    if not item.get("answerable", True):
        return {}
    relevant = [chunk_is_relevant(s, item) for s in sources]
    first = next((i for i, r in enumerate(relevant) if r), None)
    evidence = item.get("evidence") or []
    if evidence:
        recall = sum(any(evidence_in_text(e, s["text"]) for s in sources) for e in evidence) / len(evidence)
    else:
        recall = float(first is not None)
    return {
        "hit": float(first is not None),
        "mrr": 0.0 if first is None else 1.0 / (first + 1),
        "evidence_recall": recall,
    }


def is_refusal(answer: str) -> bool:
    a = norm(answer)
    return norm(REFUSAL) in a or any(m in a for m in REFUSAL_MARKERS)


def citation_metrics(answer: str, n_sources: int) -> dict:
    nums = [int(n) for n in CITATION_RE.findall(answer)]
    return {
        "has_citation": bool(nums),
        "citations_valid": bool(nums) and all(1 <= n <= n_sources for n in nums),
    }


def percentile(values: list[float], q: float) -> float | None:
    """Percentile par rang le plus proche (q entre 0 et 100)."""
    if not values:
        return None
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(round(q / 100 * len(s) + 0.5)) - 1))
    return s[idx]


def mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None
