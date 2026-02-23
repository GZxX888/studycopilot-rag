# ===== 标准库 =====
from typing import List, Tuple
from difflib import SequenceMatcher

# ===== LangChain =====
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# ===== LLM =====
from langchain_community.chat_models import ChatOllama

# ===== Rerank =====
from flashrank import Ranker, RerankRequest


# ===== 项目内 =====
from config import RAGConfig


def remove_similar_docs(docs: List[Document], threshold: float = 0.9) -> List[Document]:
    filtered = []

    for d in docs:
        text = d.page_content.strip()
        is_duplicate = False

        for kept in filtered:
            ratio = SequenceMatcher(None, text, kept.page_content.strip()).ratio()
            if ratio > threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            filtered.append(d)

    return filtered


def rerank_docs(query: str, docs: List[Document], top_k: int) -> List[Document]:
    ranker = Ranker()

    passages = [
        {"id": str(i), "text": d.page_content}
        for i, d in enumerate(docs)
    ]

    request = RerankRequest(query=query, passages=passages)
    ranked = ranker.rerank(request)

    # 取前 top_k
    top_ids = [int(r["id"]) for r in ranked[:top_k]]
    return [docs[i] for i in top_ids]


def is_chinese(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def load_vectorstore(cfg: RAGConfig) -> Chroma:
    if not cfg.vectordb_dir.exists():
        raise RuntimeError("VectorDB not found. Please run ingest.py first.")

    embeddings = HuggingFaceEmbeddings(
        model_name=cfg.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )

    return Chroma(
        persist_directory=str(cfg.vectordb_dir),
        embedding_function=embeddings,
        collection_name="studycopilot",
    )


def build_base_retriever(vectordb: Chroma, candidate_k: int):
    return vectordb.as_retriever(search_kwargs={"k": candidate_k})



def format_docs(docs: List[Document]) -> str:
    lines = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", None)
        ref = f"{source}, page {page}" if page is not None else source
        content = doc.page_content.strip().replace("\n", " ")
        lines.append(f"[{i}] ({ref}) {content}")
    return "\n".join(lines)


def has_meaningful_evidence(docs: List[Document], min_chars: int = 80) -> bool:
    # 简单 gate：至少有一条不是太短
    return any(len(d.page_content.strip()) >= min_chars for d in docs)


def translate_to_english(llm: ChatOllama, question: str) -> str:
    # 强制只输出译文，减少乱输出
    prompt = (
        "Translate the user question into English. "
        "Output ONLY the English translation, no extra words.\n\n"
        f"User question:\n{question}"
    )
    return llm.invoke(prompt).content.strip()


def generate_rewrites(llm: ChatOllama, en_question: str, n: int = 2) -> List[str]:
    # 生成多个英文改写，用于 multi-query（不要太多，成本高）
    prompt = (
        f"Rewrite the following question into {n} alternative English queries "
        "that would help retrieving relevant lecture notes. "
        "Output each query on a new line. No numbering.\n\n"
        f"Question:\n{en_question}"
    )
    out = llm.invoke(prompt).content.strip()
    lines = [x.strip() for x in out.splitlines() if x.strip()]
    # 防止模型输出太多
    return lines[:n]


RAG_PROMPT_EN = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strictly constrained study assistant.\n"
            "You MUST answer ONLY using the provided excerpts.\n"
            "Prefer excerpts that directly define the concept.\n"
            "If the excerpts do NOT directly contain the answer, reply exactly:\n"
            "\"Insufficient information to answer.\"\n"
            "Forbidden:\n"
            "- Using general knowledge\n"
            "- Guessing or extrapolating beyond excerpts\n"
            "Do NOT include any meta notes like “Note: …” or “The above answer only …”.\n"
        ),
        (
            "human",
            "Question: {question}\n\n"
            "Excerpts:\n{context}\n\n"
            "Requirements:\n"
            "1) Simple conclusion first\n"
            "2) Then explain using ONLY the excerpts\n"
            "3) Cite excerpt numbers like [1], [2]"
        ),
    ]
)

RAG_PROMPT_ZH = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个严格受限的学习助手。\n"
            "你【只能】使用“资料片段”中的信息回答问题。\n"
            "如果资料片段中【没有直接相关的信息】，你【必须】回答：\n"
            "“资料不足，无法回答该问题。”\n"
            "禁止：\n"
            "- 使用常识或背景知识补充答案\n"
            "- 推测资料中未明确给出的内容\n"
            "禁止输出任何“Note/说明：我只用了资料”等自我声明。\n"
        ),
        (
            "human",
            "问题：{question}\n\n"
            "资料片段：\n{context}\n\n"
            "要求：\n"
            "1) 先给简要结论\n"
            "2) 再结合资料片段解释（只能引用资料）\n"
            "3) 引用资料编号（如 [1], [2]）"
        ),
    ]
)


def build_rag_chain(cfg: RAGConfig):
    vectordb = load_vectorstore(cfg)

    llm = ChatOllama(model="llama3")

    # 你可以调这两个：
    candidate_k = getattr(cfg, "candidate_k", 30)  # 粗召回
    final_k = getattr(cfg, "top_k", 4)             # 最终喂给LLM

    base_retriever = build_base_retriever(vectordb, candidate_k)

    def rag_invoke(user_question: str) -> str:
        user_wants_zh = is_chinese(user_question)

        # 1) 统一成英文检索 query（语言对齐）
        en_query = user_question
        if user_wants_zh:
            en_query = translate_to_english(llm, user_question)

        # 2) multi-query（可选但推荐）
        rewrites = generate_rewrites(llm, en_query, n=2)
        queries = [en_query] + rewrites

        # 3) 多路召回 + 合并去重
        merged_docs: List[Document] = []
        seen = set()
        for q in queries:
            docs = base_retriever.invoke(q)
            for d in docs:
                key = (d.metadata.get("source"), d.metadata.get("page"), d.page_content[:120])
                if key in seen:
                    continue
                seen.add(key)
                merged_docs.append(d)

        merged_docs = remove_similar_docs(merged_docs, threshold=0.9)

        # 4) rerank（对合并后的候选做精排）
        # ContextualCompressionRetriever 是“先检索再压缩”，这里我们复用它：
        # 直接对英文主 query rerank，输出 top_k
        candidate_docs = merged_docs  # multi-query 合并后的候选
        docs_for_llm = rerank_docs(en_query, candidate_docs, final_k)

        # 5) evidence gate
        if not has_meaningful_evidence(docs_for_llm):
            return "资料不足，无法回答该问题。" if user_wants_zh else "Insufficient information to answer."

        context = format_docs(docs_for_llm)

        # 6) 输出语言：按用户输入语言
        prompt = (RAG_PROMPT_ZH if user_wants_zh else RAG_PROMPT_EN).format_messages(
            question=user_question,
            context=context
        )
        resp = llm.invoke(prompt).content.strip()

        # ✅ 把证据也输出，确保 [2][4] 可验证
        evidence_block = f"资料片段：\n{context}\n\n"
        return evidence_block + resp

    return rag_invoke


if __name__ == "__main__":
    cfg = RAGConfig()
    rag = build_rag_chain(cfg)

    print("Type your question (type 'exit' to quit):")

    while True:
        question = input(">> ")
        if question.lower() in ["exit", "quit"]:
            break

        answer = rag(question)
        print("\n=== ANSWER ===")
        print(answer)
        print("\n")
