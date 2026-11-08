"""Tests hors-ligne (aucune clé d'API) : embeddings lexicaux + LLM factice sur le corpus d'exemple."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ragassist import evaluate
from ragassist.config import RAGConfig, deep_merge, load_experiments
from ragassist.ingest import build_index, load_documents, split_documents
from ragassist.judge import parse_verdict
from ragassist.metrics import (citation_metrics, evidence_in_text, is_refusal, percentile,
                               retrieval_metrics)
from ragassist.pipeline import RAGPipeline

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "examples" / "corpus"
QUESTIONS = ROOT / "examples" / "questions.jsonl"


@pytest.fixture
def cfg(tmp_path) -> RAGConfig:
    return RAGConfig(name="test", data_dir=str(CORPUS), index_dir=str(tmp_path / "index"),
                     chunk_size=400, chunk_overlap=60, top_k=3,
                     embedding={"provider": "hash"}, llm={"provider": "fake", "price_in_per_1m": 1, "price_out_per_1m": 2})


def test_ingest_and_chunk_ids(cfg):
    docs = load_documents(cfg.data_dir)
    assert {d.metadata["source"] for d in docs} == {"notes-de-frais.md", "politique-teletravail.md"}
    chunks = split_documents(docs, 400, 60)
    assert all(len(c.page_content) <= 400 for c in chunks)
    assert len({c.metadata["chunk_id"] for c in chunks}) == len(chunks)


def test_index_is_reused(cfg):
    n = build_index(cfg, verbose=False)._collection.count()
    assert n > 0
    assert build_index(cfg, verbose=False)._collection.count() == n  # pas de doublons


def test_index_key_ignores_llm_and_k(cfg):
    other = cfg.model_copy(update={"top_k": 8, "llm": cfg.llm.model_copy(update={"name": "x"})})
    assert other.index_key() == cfg.index_key()
    assert cfg.model_copy(update={"chunk_size": 200}).index_key() != cfg.index_key()


def test_retrieval_finds_hotel_ceiling(cfg):
    docs = RAGPipeline(cfg).retrieve("plafond nuit hôtel Paris")
    assert any("170 euros" in d.page_content for d in docs)


def test_pipeline_answer_has_citation_and_cost(cfg):
    res = RAGPipeline(cfg).ask("Quel est le plafond pour une nuit d'hôtel à Paris ?")
    assert citation_metrics(res.answer, len(res.sources))["citations_valid"]
    assert len(res.sources) == 3 and res.cost_usd is not None and res.cost_usd > 0


def test_evidence_matching_is_robust():
    chunk = "Le plafond par nuit d'hôtel,\npetit-déjeuner inclus, est de 120 euros en province."
    assert evidence_in_text("plafond par nuit d’hôtel, petit-déjeuner inclus", chunk)
    assert not evidence_in_text("", chunk)
    long_ev = "Le plafond par nuit d'hôtel, petit-déjeuner inclus, est de 120 euros en province. " + "x" * 100
    assert evidence_in_text(long_ev, chunk)  # preuve coupée entre deux chunks


def test_retrieval_metrics():
    item = {"answerable": True, "evidence": ["alpha beta", "gamma"]}
    sources = [{"text": "rien"}, {"text": "... alpha beta ..."}, {"text": "autre"}]
    m = retrieval_metrics(sources, item)
    assert m == {"hit": 1.0, "mrr": 0.5, "evidence_recall": 0.5}
    assert retrieval_metrics(sources, {"answerable": False}) == {}


def test_refusal_and_citations():
    assert is_refusal("Je ne trouve pas cette information dans les documents.")
    assert not is_refusal("Le plafond est de 170 euros [1].")
    assert citation_metrics("Oui [1][3].", 3) == {"has_citation": True, "citations_valid": True}
    assert citation_metrics("Oui [5].", 3)["citations_valid"] is False


def test_judge_parsing():
    v = parse_verdict('Voici : {"faithfulness": 0.9, "correctness": "0", "justification": "ok"}')
    assert v["faithfulness"] == 1.0 and v["correctness"] == 0.0


def test_percentile():
    assert percentile([1, 2, 3, 4], 50) == 2
    assert percentile([1, 2, 3, 4], 95) == 4
    assert percentile([], 50) is None


def test_experiments_merge():
    configs, judge = load_experiments(ROOT / "configs" / "experiments.yaml")
    by_name = {c.name: c for c in configs}
    assert by_name["chunks-400"].chunk_size == 400 and by_name["chunks-400"].top_k == 4
    assert by_name["emb-lexical"].embedding.provider == "hash"
    assert judge is not None
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}) == {"a": {"b": 3, "c": 2}}


def test_end_to_end_evaluation(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    out = evaluate.main(["--experiments", "configs/experiments-offline.yaml", "--questions", str(QUESTIONS),
                         "--out", str(tmp_path)])
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "offline-k3" in report and "Exactitude par catégorie" in report
    assert (out / "summary.csv").exists() and (out / "offline-k1.jsonl").exists()


def test_api(cfg, monkeypatch, tmp_path):
    from ragassist import api

    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(f"name: t\ndata_dir: {CORPUS}\nindex_dir: {tmp_path / 'idx'}\n"
                        "embedding: {provider: hash}\nllm: {provider: fake}\n", encoding="utf-8")
    monkeypatch.setenv("RAG_CONFIG", str(cfg_file))
    api.get_pipeline.cache_clear()
    client = TestClient(api.app)
    assert client.get("/health").json()["status"] == "ok"
    r = client.post("/ask", json={"question": "Quel jour suis-je au bureau ?"})
    assert r.status_code == 200 and r.json()["sources"]
    assert client.post("/ask", json={"question": ""}).status_code == 422
