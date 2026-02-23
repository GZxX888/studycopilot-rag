# ===== 标准库 =====
from typing import List
import re

# ===== LangChain =====
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ===== 项目内 =====
from config import RAGConfig
from loaders import load_documents_from_dir


# ============================================================
# 1) 低信息过滤
# ============================================================
def is_low_information_chunk(text: str) -> bool:
    if not text:
        return True

    t = text.strip().lower()

    if len(t) < 30:
        return True

    blacklist = [
        "thanks",
        "thank you",
        "follow the slope",
        "questions?",
        "any questions",
        "end of lecture",
    ]
    for b in blacklist:
        if b in t:
            return True

    letters = sum(c.isalpha() for c in t)
    if letters / max(len(t), 1) < 0.3:
        return True

    # PPT 渐进残留页常见模式
    if t.count("what is") > 3:
        return True

    return False


# ============================================================
# 2) 移除渐进式幻灯片页（保留“最后一页”）
#    逻辑：如果当前页是上一页的“子串”，说明它更短、更早，是过渡页 → 删
# ============================================================
def remove_progressive_slides(docs: List[Document]) -> List[Document]:
    cleaned: List[Document] = []
    prev_text = ""

    for doc in docs:
        text = doc.page_content.strip()
        if len(text) < 20:
            continue

        # 当前页是上一页的子串 => 当前更短，是过渡页，丢弃
        if prev_text and text in prev_text:
            continue

        cleaned.append(doc)
        prev_text = text

    return cleaned


# ============================================================
# 3) 去重（完全相同文本）
# ============================================================
def deduplicate_docs(docs: List[Document]) -> List[Document]:
    unique = []
    seen = set()
    for d in docs:
        t = d.page_content.strip()
        if t in seen:
            continue
        seen.add(t)
        unique.append(d)
    return unique


# ============================================================
# 4) semantic chunking（页内语义分块）
#    - PPT 通常“一页就是一个语义单元”，但有些页很长（例题、推导）会太大。
#    - 策略：只对“超长页”做页内拆分；短页保持整页。
# ============================================================
def semantic_chunk_pages(
    docs: List[Document],
    max_page_chars: int,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "。", " ", ""],
    )

    out: List[Document] = []
    for d in docs:
        text = d.page_content.strip()

        # 短页直接保留，不拆
        if len(text) <= max_page_chars:
            out.append(d)
            continue

        # 长页才拆分（页内语义拆）
        sub_docs = splitter.split_documents([d])
        out.extend(sub_docs)

    return out


def ingest_documents(cfg: RAGConfig) -> None:
    # ---------- 1) load ----------
    docs: List[Document] = load_documents_from_dir(cfg.data_dir)
    if not docs:
        raise RuntimeError("No documents found in data directory.")

    print(f"Original pages: {len(docs)}")

    # ---------- 2) remove progressive ----------
    docs = remove_progressive_slides(docs)
    print(f"After removing progressive slides: {len(docs)}")

    # ---------- 3) low-info filter (page-level) ----------
    docs = [d for d in docs if not is_low_information_chunk(d.page_content)]
    print(f"After low-information filtering: {len(docs)}")

    # ---------- 4) dedup ----------
    docs = deduplicate_docs(docs)
    print(f"After deduplication: {len(docs)}")

    # ---------- 5) semantic chunking (page-internal) ----------
    # 你可以根据课件实际长度调：
    # - max_page_chars：超过这个长度才拆
    # - chunk_size / overlap：拆分粒度
    max_page_chars = 1800
    chunk_size = getattr(cfg, "chunk_size", 800)
    chunk_overlap = getattr(cfg, "chunk_overlap", 120)

    chunks = semantic_chunk_pages(
        docs,
        max_page_chars=max_page_chars,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    print(f"After semantic chunking: {len(chunks)}")

    # ---------- 6) Embeddings (normalize) ----------
    embeddings = HuggingFaceEmbeddings(
        model_name=cfg.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )

    # ---------- 7) VectorDB ----------
    cfg.vectordb_dir.mkdir(parents=True, exist_ok=True)
    _ = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(cfg.vectordb_dir),
        collection_name="studycopilot",
    )

    print("\n✅ Ingest finished")
    print(f"- Final Chunks: {len(chunks)}")
    print(f"- VectorDB path: {cfg.vectordb_dir}")


if __name__ == "__main__":
    cfg = RAGConfig()
    ingest_documents(cfg)
