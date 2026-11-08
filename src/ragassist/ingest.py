"""Ingestion : lecture des documents, découpage en chunks, indexation dans Chroma.

Usage :  python -m ragassist.ingest --config configs/default.yaml [--rebuild]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .config import RAGConfig, load_config
from .llm import make_embeddings

SUPPORTED = {".pdf", ".md", ".txt"}
COLLECTION = "chunks"


def clean_text(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # mots coupés en fin de ligne (PDF)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_documents(data_dir: str | Path) -> list[Document]:
    """Un Document par page de PDF, un par fichier .md/.txt (page = 0)."""
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dossier de documents introuvable : {root}")
    docs: list[Document] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        rel = path.relative_to(root).as_posix()
        if path.suffix.lower() == ".pdf":
            for i, page in enumerate(PdfReader(str(path)).pages, start=1):
                text = clean_text(page.extract_text() or "")
                if text:
                    docs.append(Document(page_content=text, metadata={"source": rel, "page": i}))
        else:
            text = clean_text(path.read_text(encoding="utf-8", errors="replace"))
            if text:
                docs.append(Document(page_content=text, metadata={"source": rel, "page": 0}))
    return docs


def split_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for c in chunks:
        m = c.metadata
        m["chunk_id"] = f"{m['source']}#p{m['page']}@{m['start_index']}"
    return chunks


def index_path(cfg: RAGConfig) -> Path:
    return Path(cfg.index_dir) / cfg.index_key()


def open_vectorstore(cfg: RAGConfig) -> Chroma:
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=make_embeddings(cfg.embedding),
        persist_directory=str(index_path(cfg)),
        collection_metadata={"hnsw:space": "cosine"},
    )


def build_index(cfg: RAGConfig, rebuild: bool = False, batch_size: int = 64, verbose: bool = True) -> Chroma:
    """Construit l'index s'il n'existe pas encore (ou si rebuild=True) et le renvoie."""
    path = index_path(cfg)
    if rebuild and path.exists():
        shutil.rmtree(path)
    vs = open_vectorstore(cfg)
    if vs._collection.count() > 0:
        if verbose:
            print(f"[index] {path} déjà construit ({vs._collection.count()} chunks)")
        return vs

    docs = load_documents(cfg.data_dir)
    chunks = split_documents(docs, cfg.chunk_size, cfg.chunk_overlap)
    if not chunks:
        raise ValueError(f"Aucun texte extrait de {cfg.data_dir} (formats acceptés : {sorted(SUPPORTED)})")
    unique = {c.metadata["chunk_id"]: c for c in chunks}
    chunks = list(unique.values())

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vs.add_documents(batch, ids=[c.metadata["chunk_id"] for c in batch])
        if verbose:
            print(f"[index] {min(i + batch_size, len(chunks))}/{len(chunks)} chunks indexés", end="\r")

    manifest = {
        "data_dir": cfg.data_dir,
        "embedding": cfg.embedding.model_dump(),
        "chunk_size": cfg.chunk_size,
        "chunk_overlap": cfg.chunk_overlap,
        "n_pages": len(docs),
        "n_chunks": len(chunks),
        "sources": sorted({d.metadata["source"] for d in docs}),
    }
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if verbose:
        print(f"\n[index] {len(docs)} pages → {len(chunks)} chunks → {path}")
    return vs


def main() -> None:
    ap = argparse.ArgumentParser(description="Indexe les documents du corpus.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--rebuild", action="store_true", help="Supprime et reconstruit l'index")
    args = ap.parse_args()
    build_index(load_config(args.config), rebuild=args.rebuild)


if __name__ == "__main__":
    main()
