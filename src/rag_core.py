# src/rag_core.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings


@dataclass
class RAGConfig:
    persist_dir: str
    embedding_model: str
    top_k: int = 4


def build_retriever(cfg: RAGConfig):
    """
    Load existing Chroma vectordb (persisted) and return a retriever.
    IMPORTANT: embedding_model MUST match the one used during ingest.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=cfg.embedding_model,
        encode_kwargs={"normalize_embeddings": True},  # match your ingest
    )

    vectordb = Chroma(
        persist_directory=cfg.persist_dir,
        embedding_function=embeddings,
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": cfg.top_k})
    return retriever


def format_docs(docs: List[Document], max_chars: int = 6000) -> Tuple[str, List[str]]:
    chunks = []
    citations = []
    total = 0

    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", d.metadata.get("page_number", "NA"))
        cite = f"[{i}] ({src}, page {page})"
        text = (d.page_content or "").strip()

        block = f"{cite}\n{text}\n"
        if total + len(block) > max_chars:
            break

        chunks.append(block)
        citations.append(cite)
        total += len(block)

    return "\n".join(chunks).strip(), citations