"""LLM-juge : note la fidélité aux extraits et l'exactitude par rapport à la réponse de référence.

Bonnes pratiques appliquées :
- le même juge (modèle et prompt fixes, température 0) note toutes les expériences ;
- grille discrète (0 / 0,5 / 1) plutôt qu'une note libre, plus stable d'un appel à l'autre ;
- une justification en une phrase pour pouvoir auditer les notes à la main ;
- de préférence un juge d'une autre famille que les modèles évalués (biais d'auto-préférence).
"""
from __future__ import annotations

import json
import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from .llm import message_text

JUDGE_PROMPT = """Tu es un ÉVALUATEUR strict d'un assistant de question-réponse sur documents.

Question : {question}

Extraits fournis à l'assistant :
{context}

Réponse de référence : {reference}

Réponse de l'assistant : {answer}

Évalue deux critères :
1. faithfulness (fidélité) : chaque affirmation de la réponse est-elle soutenue par les extraits ?
   1 = entièrement ; 0.5 = au moins une affirmation non soutenue ; 0 = majoritairement non soutenue.
   Un refus de répondre obtient 1.
2. correctness (exactitude) : la réponse contient-elle l'information essentielle de la référence, sans contradiction ?
   1 = oui ; 0.5 = partiellement ; 0 = non ou refus injustifié.
   Si la référence indique que le corpus ne contient pas la réponse : 1 si l'assistant refuse de répondre, 0 sinon.

Réponds uniquement avec un objet JSON, sans texte autour :
{{"faithfulness": <0|0.5|1>, "correctness": <0|0.5|1>, "justification": "<une phrase>"}}"""

NO_ANSWER = "AUCUNE — le corpus ne contient pas la réponse à cette question."


def _clamp(v: object) -> float | None:
    try:
        x = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return min([0.0, 0.5, 1.0], key=lambda g: abs(g - x))


def parse_verdict(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("pas de JSON dans la réponse du juge")
    data = json.loads(m.group(0))
    return {
        "faithfulness": _clamp(data.get("faithfulness")),
        "correctness": _clamp(data.get("correctness")),
        "justification": str(data.get("justification", ""))[:300],
    }


def judge_answer(llm: BaseChatModel, question: str, answer: str, sources: list[dict],
                 reference: str | None, retries: int = 1) -> dict:
    context = "\n\n".join(f"[{s['rank']}] {s['text']}" for s in sources) or "(aucun)"
    prompt = JUDGE_PROMPT.format(question=question, context=context,
                                 reference=reference if reference else NO_ANSWER, answer=answer)
    error = ""
    for _ in range(retries + 1):
        try:
            return parse_verdict(message_text(llm.invoke([HumanMessage(content=prompt)])))
        except Exception as e:  # JSON invalide, erreur réseau…
            error = f"{type(e).__name__}: {e}"
    return {"faithfulness": None, "correctness": None, "justification": f"échec du juge ({error})"}
