"""Génère un BROUILLON de questions à partir de chunks tirés au hasard.

Usage :  python -m ragassist.generate_questions --config configs/default.yaml --n 40

⚠️ Ces questions doivent être relues à la main avant de servir d'évaluation :
corrige les formulations, supprime les questions triviales ou ambiguës, puis copie les
bonnes lignes dans data/eval/questions.jsonl. Ajoute aussi à la main :
  - des questions « multi-passages » (réponse répartie sur 2 extraits ou plus) ;
  - des questions « hors-corpus » (reference_answer: null, answerable: false).
Un jeu 100 % synthétique favorise le retrieval (la question reprend les mots de l'extrait) :
c'est pour ça qu'on le reformule et qu'on le complète.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from langchain_core.messages import HumanMessage

from .config import load_config
from .ingest import build_index
from .llm import make_chat, message_text
from .metrics import evidence_in_text

GEN_PROMPT = """Tu es un GÉNÉRATEUR de questions d'évaluation pour un assistant documentaire.
À partir de l'extrait ci-dessous, rédige UNE question qu'un utilisateur réel pourrait poser,
dont la réponse se trouve dans l'extrait (de préférence un fait précis : chiffre, date, condition, règle).
Évite de reprendre mot pour mot les termes de l'extrait dans la question.

Extrait :
<<<
{chunk}
>>>

Réponds uniquement avec un objet JSON :
{{"question": "...", "reference_answer": "réponse courte et complète", "evidence": "phrase copiée MOT POUR MOT de l'extrait qui justifie la réponse"}}"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="data/eval/questions.draft.jsonl")
    ap.add_argument("--min-chars", type=int, default=300, help="ignore les chunks trop courts")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.config)
    vs = build_index(cfg, verbose=False)
    data = vs.get(include=["documents", "metadatas"])
    pool = [(t, m) for t, m in zip(data["documents"], data["metadatas"]) if len(t) >= args.min_chars]
    random.Random(args.seed).shuffle(pool)
    llm = make_chat(cfg.llm)

    kept, rejected = [], 0
    for text, meta in pool:
        if len(kept) >= args.n:
            break
        try:
            raw = message_text(llm.invoke([HumanMessage(content=GEN_PROMPT.format(chunk=text))]))
            start, end = raw.find("{"), raw.rfind("}")
            q = json.loads(raw[start : end + 1])
        except Exception:
            rejected += 1
            continue
        # La preuve doit réellement figurer dans le chunk : sinon le LLM l'a inventée.
        if not q.get("question") or not evidence_in_text(q.get("evidence", ""), text):
            rejected += 1
            continue
        kept.append({
            "id": f"gen-{len(kept) + 1:03d}",
            "question": q["question"],
            "reference_answer": q.get("reference_answer"),
            "evidence": [q["evidence"]],
            "source": meta.get("source"),
            "pages": [meta.get("page", 0)],
            "answerable": True,
            "category": "factuelle",
            "origin": "synthetique",
            "reviewed": False,
        })
        print(f"  {len(kept)}/{args.n}", end="\r")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for item in kept:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\n{len(kept)} questions écrites dans {args.out} ({rejected} rejetées : preuve introuvable ou JSON invalide).")
    print("→ Relis-les, puis copie les bonnes dans data/eval/questions.jsonl.")


if __name__ == "__main__":
    main()
