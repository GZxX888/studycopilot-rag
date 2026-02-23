from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RAGConfig:
    """
    RAG 项目的统一配置中心
    """

    # ---------- 路径 ----------
    project_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = project_root / "data"
    vectordb_dir: Path = project_root / "vectordb"

    # ---------- 文本切块（给 ingest.py 用）----------
    chunk_size: int = 800
    chunk_overlap: int = 120

    # ---------- 检索 ----------
    top_k: int = 4            # 最终喂给 LLM 的 chunk 数
    candidate_k: int = 30     # 每个 query 粗召回的候选数

    # ---------- Embedding ----------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ---------- LLM ----------
    llm_model: str = "llama3"

    # ---------- Query Router ----------
    enable_query_router: bool = True
    force_level: int | None = None         # 设为 0/1/2 可强制策略；None 表示自动判断
    n_rewrites: int = 2
    max_sub_questions: int = 3

    # ---------- Hybrid Retrieval ----------
    enable_hybrid: bool = True
    bm25_k: int = 20


    # ---------- Evidence Gate ----------
    min_evidence_chars: int = 80

    # ---------- Aspect Split（definition/application 拆分）----------
    enable_aspect_split: bool = True

    # ---------- Debug ----------
    show_debug: bool = False