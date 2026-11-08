"""Démo Streamlit.  Lancement :  streamlit run app/streamlit_app.py
Onglet 1 : poser des questions et voir les sources citées. Onglet 2 : comparer les résultats d'évaluation.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from ragassist.config import load_config
from ragassist.pipeline import RAGPipeline

st.set_page_config(page_title="Assistant documentaire", page_icon="📄", layout="wide")

CONFIG_DIR = Path("configs")
configs = sorted(p.name for p in CONFIG_DIR.glob("*.yaml") if not p.name.startswith("experiments"))
default = os.getenv("RAG_CONFIG", "configs/default.yaml")
config_name = st.sidebar.selectbox("Configuration", configs,
                                   index=configs.index(Path(default).name) if Path(default).name in configs else 0)


@st.cache_resource(show_spinner="Chargement de l'index…")
def get_pipeline(path: str) -> RAGPipeline:
    return RAGPipeline(load_config(path))


try:
    pipe = get_pipeline(str(CONFIG_DIR / config_name))
except Exception as e:
    st.error(f"Impossible de charger la configuration : {e}")
    st.stop()

cfg = pipe.cfg
st.sidebar.markdown(
    f"**LLM** : `{cfg.llm.provider}:{cfg.llm.name}`  \n**Embeddings** : `{cfg.embedding.provider}:{cfg.embedding.name}`  \n"
    f"**Chunks** : {cfg.chunk_size} car. · **k** = {cfg.top_k} · {cfg.search_type}  \n"
    f"**Index** : {pipe.vs._collection.count()} chunks"
)

tab_chat, tab_eval = st.tabs(["💬 Assistant", "📊 Évaluation"])

with tab_chat:
    st.title("Assistant documentaire")
    st.caption("Les réponses ne s'appuient que sur les documents indexés ; chaque affirmation cite sa source.")
    if "history" not in st.session_state:
        st.session_state.history = []

    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            st.caption(turn["meta"])
            cited = {int(n) for n in re.findall(r"\[(\d+)\]", turn["answer"])}
            for s in turn["sources"]:
                loc = f", p. {s['page']}" if s["page"] else ""
                mark = "✅ " if s["rank"] in cited else ""
                with st.expander(f"{mark}[{s['rank']}] {s['source']}{loc}"):
                    st.write(s["text"])

    if question := st.chat_input("Pose une question sur les documents…"):
        with st.spinner("Recherche et rédaction…"):
            try:
                res = pipe.ask(question)
            except Exception as e:
                st.error(f"Erreur du fournisseur : {e}")
                st.stop()
        cost = f" · {res.cost_usd * 1000:.2f} $ / 1000 req" if res.cost_usd is not None else ""
        st.session_state.history.append({
            "question": question, "answer": res.answer, "sources": res.sources,
            "meta": f"{res.latency_s:.2f} s (recherche {res.retrieval_s * 1000:.0f} ms) · "
                    f"{res.input_tokens} + {res.output_tokens} tokens{cost}",
        })
        st.rerun()

with tab_eval:
    st.title("Comparaison des configurations")
    runs = sorted(Path("results").glob("*/summary.csv"), reverse=True)
    if not runs:
        st.info("Aucun résultat. Lance : `python -m ragassist.evaluate --experiments configs/experiments.yaml`")
    else:
        run = st.selectbox("Campagne", runs, format_func=lambda p: p.parent.name)
        df = pd.read_csv(run)
        pct = ["hit_at_k", "evidence_recall", "faithfulness", "correctness", "abstention", "false_refusal", "citation_rate"]
        show = df[["experiment", *pct, "latency_p50_s", "cost_per_1k_usd"]].set_index("experiment")
        st.dataframe(show.style.format({c: "{:.0%}" for c in pct} | {"latency_p50_s": "{:.2f}", "cost_per_1k_usd": "{:.2f}"},
                                       na_rep="—"), width="stretch")
        metric = st.selectbox("Métrique", pct, index=pct.index("correctness"))
        st.bar_chart(df.set_index("experiment")[metric])
        report = run.parent / "report.md"
        if report.exists():
            with st.expander("Rapport complet"):
                st.markdown(report.read_text(encoding="utf-8"))
