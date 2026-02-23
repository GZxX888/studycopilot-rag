# ===== 标准库 =====
from pathlib import Path
from typing import List

# ===== LangChain =====
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document


def load_documents_from_dir(data_dir: Path) -> List[Document]:
    """
    从 data_dir 读取 pdf / md / txt 文件
    返回 LangChain Document 列表
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    documents: List[Document] = []

    for file_path in sorted(data_dir.rglob("*")):
        if file_path.is_dir():
            continue

        suffix = file_path.suffix.lower()

        # ---- PDF ----
        if suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
            documents.extend(loader.load())

        # ---- Markdown / Text ----
        elif suffix in [".md", ".txt"]:
            loader = TextLoader(str(file_path), encoding="utf-8")
            documents.extend(loader.load())

        # 其他格式直接忽略

    return documents
