# ===== 标准库 =====
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

    # ---------- 文本切块 ----------
    chunk_size: int = 800
    chunk_overlap: int = 120

    # ---------- 检索 ----------
    top_k: int = 4

    # ---------- Embedding ----------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
