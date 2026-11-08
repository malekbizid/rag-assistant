"""Banc d'évaluation : lance chaque configuration sur le jeu de questions et produit un rapport comparatif.

Usage :
    python -m ragassist.evaluate --experiments configs/experiments.yaml --questions data/eval/questions.jsonl
    options : --only exp1,exp2   --limit 10   --no-judge   --out results
Sorties (dans results/<horodatage>/) : <expérience>.jsonl (détail par question), summary.csv, report.md
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from .config import RAGConfig, load_experiments
from .judge import judge_answer
from .llm import make_chat
from .metrics import citation_metrics, is_refusal, mean, percentile, retrieval_metrics
from .pipeline import RAGPipeline

COLUMNS = [
    ("experiment", "Expérience", None),
    ("hit_at_k", "Hit@k", "pct"),
    ("mrr", "MRR", "f2"),
    ("evidence_recall", "Rappel preuves", "pct"),
    ("faithfulness", "Fidélité", "pct"),
    ("correctness", "Exactitude", "pct"),
    ("abstention", "Refus hors-corpus", "pct"),
    ("false_refusal", "Faux refus", "pct"),
    ("citation_rate", "Réponses citées", "pct"),
    ("latency_p50_s", "Latence p50 (s)", "f2"),
    ("latency_p95_s", "Latence p95 (s)", "f2"),
    ("tokens_in", "Tokens in", "int"),
    ("tokens_out", "Tokens out", "int"),
    ("cost_per_1k_usd", "Coût / 1000 req ($)", "f2"),
    ("errors", "Erreurs", "int"),
]


def load_questions(path: str | Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            item = json.loads(line)
            if "question" not in item:
                raise ValueError(f"{path}:{n} — champ 'question' manquant")
            item.setdefault("id", f"q{n:03d}")
            item.setdefault("answerable", item.get("reference_answer") is not None)
            item.setdefault("category", "non classée")
            items.append(item)
    return items


def run_experiment(cfg: RAGConfig, questions: list[dict], judge_llm, out_dir: Path) -> list[dict]:
    pipe = RAGPipeline(cfg)
    rows = []
    for i, item in enumerate(questions, start=1):
        print(f"  [{cfg.name}] {i}/{len(questions)} {item['id']}", end="\r")
        row = {"experiment": cfg.name, "id": item["id"], "category": item["category"],
               "answerable": item["answerable"], "question": item["question"],
               "reference_answer": item.get("reference_answer")}
        try:
            ans = pipe.ask(item["question"])
        except Exception as e:
            rows.append({**row, "error": f"{type(e).__name__}: {e}"})
            continue
        refused = is_refusal(ans.answer)
        row.update(
            answer=ans.answer, refused=refused,
            latency_s=ans.latency_s, retrieval_s=ans.retrieval_s,
            input_tokens=ans.input_tokens, output_tokens=ans.output_tokens, cost_usd=ans.cost_usd,
            retrieved=[{"rank": s["rank"], "source": s["source"], "page": s["page"]} for s in ans.sources],
            **retrieval_metrics(ans.sources, item),
            **citation_metrics(ans.answer, len(ans.sources)),
        )
        if judge_llm is not None:
            row.update(judge_answer(judge_llm, item["question"], ans.answer, ans.sources,
                                    item.get("reference_answer") if item["answerable"] else None))
        rows.append(row)
    print()
    with open(out_dir / f"{cfg.name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


def summarize(cfg: RAGConfig, rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    ans = [r for r in ok if r["answerable"]]
    unans = [r for r in ok if not r["answerable"]]
    answered = [r for r in ok if not r["refused"]]
    lat = [r["latency_s"] for r in ok]
    costs = [r["cost_usd"] for r in ok]
    return {
        "experiment": cfg.name,
        "hit_at_k": mean([r.get("hit") for r in ans]),
        "mrr": mean([r.get("mrr") for r in ans]),
        "evidence_recall": mean([r.get("evidence_recall") for r in ans]),
        "faithfulness": mean([r.get("faithfulness") for r in ok]),
        "correctness": mean([r.get("correctness") for r in ans]),
        "abstention": mean([float(r["refused"]) for r in unans]),
        "false_refusal": mean([float(r["refused"]) for r in ans]),
        "citation_rate": mean([float(r["has_citation"]) for r in answered]),
        "latency_p50_s": percentile(lat, 50),
        "latency_p95_s": percentile(lat, 95),
        "tokens_in": mean([r["input_tokens"] for r in ok]),
        "tokens_out": mean([r["output_tokens"] for r in ok]),
        "cost_per_1k_usd": None if (not costs or None in costs) else 1000 * sum(costs) / len(costs),
        "errors": len(rows) - len(ok),
        "n": len(rows),
        "config": f"llm={cfg.llm.provider}:{cfg.llm.name} · emb={cfg.embedding.provider}:{cfg.embedding.name} · "
                  f"chunks={cfg.chunk_size}/{cfg.chunk_overlap} · k={cfg.top_k} · {cfg.search_type}",
    }


def _fmt(v, kind) -> str:
    if v is None:
        return "—"
    if kind == "pct":
        return f"{100 * v:.0f} %"
    if kind == "f2":
        return f"{v:.2f}"
    if kind == "int":
        return f"{v:.0f}"
    return str(v)


def by_category(rows: list[dict]) -> dict[str, float | None]:
    groups = defaultdict(list)
    for r in rows:
        if "error" not in r:
            groups[r["category"]].append(r.get("correctness"))
    return {cat: mean(vals) for cat, vals in sorted(groups.items())}


def write_report(summaries: list[dict], all_rows: dict[str, list[dict]], out_dir: Path,
                 judge_name: str | None, n_questions: int) -> None:
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)

    L = [f"# Rapport d'évaluation — {datetime.now():%Y-%m-%d %H:%M}", "",
         f"{n_questions} questions · juge : {judge_name or 'désactivé'}", "",
         "| " + " | ".join(h for _, h, _ in COLUMNS) + " |",
         "|" + "---|" * len(COLUMNS)]
    for s in summaries:
        L.append("| " + " | ".join(_fmt(s[k], kind) for k, _, kind in COLUMNS) + " |")

    L += ["", "## Configurations", ""] + [f"- **{s['experiment']}** : {s['config']}" for s in summaries]

    cats = sorted({r["category"] for rows in all_rows.values() for r in rows})
    if any(s["correctness"] is not None for s in summaries):
        L += ["", "## Exactitude par catégorie de question", "",
              "| Expérience | " + " | ".join(cats) + " |", "|" + "---|" * (len(cats) + 1)]
        for name, rows in all_rows.items():
            bc = by_category(rows)
            L.append(f"| {name} | " + " | ".join(_fmt(bc.get(c), "pct") for c in cats) + " |")

    L += ["", "## Échecs à analyser (3 pires réponses par expérience)", ""]
    for name, rows in all_rows.items():
        scored = [r for r in rows if r.get("correctness") is not None]
        worst = sorted(scored, key=lambda r: (r["correctness"], r.get("faithfulness") or 0))[:3]
        errs = [r for r in rows if "error" in r][:3]
        if not worst and not errs:
            continue
        L.append(f"### {name}")
        for r in worst:
            L.append(f"- `{r['id']}` exactitude {r['correctness']}, fidélité {r.get('faithfulness')} — "
                     f"*{r['question']}* → {r.get('justification', '')}")
        for r in errs:
            L.append(f"- `{r['id']}` ERREUR : {r['error']}")
        L.append("")

    L += ["## Définitions", "",
          "- **Hit@k** : part des questions (avec réponse) dont au moins un extrait récupéré contient la preuve annotée.",
          "- **MRR** : moyenne de 1/rang du premier extrait pertinent.",
          "- **Rappel preuves** : part des preuves annotées retrouvées dans les k extraits (questions multi-passages).",
          "- **Fidélité** (LLM-juge) : la réponse est-elle entièrement soutenue par les extraits ? Mesure les hallucinations.",
          "- **Exactitude** (LLM-juge) : la réponse correspond-elle à la référence ?",
          "- **Refus hors-corpus** : part des questions sans réponse dans le corpus où l'assistant refuse bien (plus haut = mieux).",
          "- **Faux refus** : part des questions avec réponse où l'assistant refuse à tort (plus bas = mieux).",
          "- **Coût / 1000 req** : calculé avec les tarifs renseignés dans la config (— si non renseignés)."]
    (out_dir / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description="Compare des configurations RAG sur un jeu de questions annotées.")
    ap.add_argument("--experiments", default="configs/experiments.yaml")
    ap.add_argument("--questions", default="data/eval/questions.jsonl")
    ap.add_argument("--out", default="results")
    ap.add_argument("--only", help="noms d'expériences séparés par des virgules")
    ap.add_argument("--limit", type=int, help="ne prendre que les N premières questions (test rapide)")
    ap.add_argument("--no-judge", action="store_true", help="métriques déterministes seulement (gratuit)")
    args = ap.parse_args(argv)

    configs, judge_spec = load_experiments(args.experiments)
    if args.only:
        wanted = set(args.only.split(","))
        configs = [c for c in configs if c.name in wanted]
        if missing := wanted - {c.name for c in configs}:
            raise SystemExit(f"Expériences inconnues : {sorted(missing)}")
    questions = load_questions(args.questions)[: args.limit]
    judge_llm = None if args.no_judge or judge_spec is None else make_chat(judge_spec)
    judge_name = None if judge_llm is None else f"{judge_spec.provider}:{judge_spec.name}"

    out_dir = Path(args.out) / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(questions)} questions × {len(configs)} expériences → {out_dir}")

    summaries, all_rows = [], {}
    for cfg in configs:
        t = time.perf_counter()
        rows = run_experiment(cfg, questions, judge_llm, out_dir)
        all_rows[cfg.name] = rows
        summaries.append(summarize(cfg, rows))
        print(f"  [{cfg.name}] terminé en {time.perf_counter() - t:.0f} s")

    write_report(summaries, all_rows, out_dir, judge_name, len(questions))
    print((out_dir / "report.md").read_text(encoding="utf-8").split("## Configurations")[0])
    return out_dir


if __name__ == "__main__":
    main()
