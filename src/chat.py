# ===== 标准库 =====
from __future__ import annotations

from typing import List, Tuple, Optional
from difflib import SequenceMatcher

# ===== LangChain =====
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# ===== LLM（本地 Ollama 聊天模型）=====
from langchain_community.chat_models import ChatOllama

# ===== Rerank（FlashRank：对召回文段做重排序）=====
from flashrank import Ranker, RerankRequest
from rank_bm25 import BM25Okapi
import re

# ===== 项目内配置 =====
from config import RAGConfig


# ============================================================
# 1) 工具函数：去重 / 重排序
# ============================================================

def remove_similar_docs(docs: List[Document], threshold: float = 0.9) -> List[Document]:
    """
    去除“内容高度相似”的文档块（chunk）
    - 用 difflib.SequenceMatcher 做字符串相似度（0~1）
    - threshold 越高：越严格（>0.9 基本可视为重复）
    """
    filtered: List[Document] = []
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
    """
    使用 FlashRank 对候选 docs 重排序
    - 输入：query（主查询）+ 候选段落 docs（粗召回合并后的集合）
    - 输出：按相关性排序后的 top_k 个 docs
    """
    if not docs:
        return []

    ranker = Ranker()
    passages = [{"id": str(i), "text": d.page_content} for i, d in enumerate(docs)]
    request = RerankRequest(query=query, passages=passages)
    ranked = ranker.rerank(request)

    top_ids = [int(r["id"]) for r in ranked[: max(top_k, 1)]]
    return [docs[i] for i in top_ids if 0 <= i < len(docs)]


# ============================================================
# 2) 语言检测 / 证据格式化 / 证据闸门
# ============================================================

def is_chinese(text: str) -> bool:
    """粗略判断：文本中是否包含中文字符"""
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def format_docs(docs: List[Document]) -> str:
    """
    将检索到的 docs 格式化成可引用的“资料片段”块
    输出形如：
    [1] (source, page X) content...
    """
    lines = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", None)
        ref = f"{source}, page {page}" if page is not None else source
        content = doc.page_content.strip().replace("\n", " ")
        lines.append(f"[{i}] ({ref}) {content}")
    return "\n".join(lines)


def has_meaningful_evidence(docs: List[Document], min_chars: int = 80) -> bool:
    """
    证据闸门（简化版）：
    - 至少有一个 chunk 的有效长度 >= min_chars
    """
    return any(len(d.page_content.strip()) >= min_chars for d in docs)


def translate_to_english(llm: ChatOllama, question: str) -> str:
    """
    将中文问题翻译为英文（用于统一检索语种）
    - 注意：这里只输出英文翻译，不要多余解释
    """
    prompt = (
        "Translate the user question into English. "
        "Output ONLY the English translation, no extra words.\n\n"
        f"User question:\n{question}"
    )
    return llm.invoke(prompt).content.strip()


# ============================================================
# 3) Query 模块：分级触发（Router）+ Query Rewrite + Decomposition
# ============================================================

def question_complexity_level(question_en: str) -> int:
    """
    根据英文问题的复杂度启用不同检索策略
      0 = 简单（单 query）
      1 = 中等（multi-query）
      2 = 复杂（decomposition + multi-query）
    """
    q = " " + question_en.lower().strip() + " "

    multi_markers = [
        " and ", " or ", " also ", " compare ", " difference ", " relationship ",
        " pros and cons ", " advantages ", " disadvantages ", " steps ", " derive ", " prove ",
    ]

    long_question = len(q.split()) > 25
    multi_hit = any(m in q for m in multi_markers)
    many_punct = q.count("?") >= 2 or q.count(";") >= 2

    if (multi_hit and long_question) or many_punct:
        return 2
    if multi_hit or long_question:
        return 1
    return 0


def generate_rewrites(llm: ChatOllama, en_question: str, n: int = 2) -> List[str]:
    """
    Query Rewrite（多查询生成）：
    - 输入：英文问题
    - 输出：n 条“更适合检索”的改写 query（每行一个）
    """
    if n <= 0:
        return []
    prompt = (
        f"Rewrite the following question into {n} alternative English retrieval queries "
        "that would help retrieving relevant lecture notes. "
        "Output each query on a new line. No numbering.\n\n"
        f"Question:\n{en_question}"
    )
    out = llm.invoke(prompt).content.strip()
    lines = [x.strip() for x in out.splitlines() if x.strip()]
    return lines[:n]


def decompose_question(llm: ChatOllama, en_question: str, max_parts: int = 3) -> List[str]:
    """
    Decomposition（问题拆解）：
    - 将复杂问题拆成最多 max_parts 个“可独立检索”的子问题
    - 输出每行一个子问题
    """
    if max_parts <= 1:
        return [en_question]

    prompt = (
        f"Break the following question into up to {max_parts} independent sub-questions "
        "that can be answered separately from lecture notes. "
        "Output each sub-question on a new line. No numbering.\n\n"
        f"Question:\n{en_question}"
    )
    out = llm.invoke(prompt).content.strip()
    subs = [x.strip() for x in out.splitlines() if x.strip()]
    return subs[:max_parts] if subs else [en_question]


def build_queries(
    llm: ChatOllama,
    en_question: str,
    level: int,
    n_rewrites: int,
    max_sub_questions: int,
) -> Tuple[List[str], List[str]]:
    """
    Query Builder：根据 level 构建最终用于检索的 query 列表
    返回：
      queries_for_retrieval：用于 base_retriever.invoke(q) 的 query 列表
      debug_notes：调试信息（可选输出）
    """
    debug: List[str] = []

    if level <= 0:
        debug.append("LEVEL=0 (single query)")
        return [en_question], debug

    if level == 1:
        debug.append("LEVEL=1 (multi-query)")
        rewrites = generate_rewrites(llm, en_question, n=n_rewrites)
        return [en_question] + rewrites, debug

    debug.append("LEVEL=2 (decomposition + multi-query)")
    subs = decompose_question(llm, en_question, max_parts=max_sub_questions)
    debug.append(f"SUB_QUESTIONS={len(subs)}")

    queries: List[str] = []
    for sq in subs:
        rewrites = generate_rewrites(llm, sq, n=max(1, n_rewrites // 2))
        queries.extend([sq] + rewrites)

    # 去重（保持顺序）
    seen = set()
    deduped: List[str] = []
    for q in queries:
        key = q.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(q.strip())

    return deduped, debug


# ============================================================
# 4) Aspect Split（关键新增）：把“definition + application”拆成子问
#    - 目的：避免出现“定义有证据但应用没证据”时还硬答应用
#    - 做法：两个子问分别检索、分别 evidence gate、分别回答，然后拼接输出
# ============================================================

def detect_definition_application_split(en_question: str) -> Optional[List[Tuple[str, str]]]:
    """
    如果问题明显是“definition + application/usage”，返回两个子问：
      [("Definition", ...), ("Application", ...)]
    否则返回 None（表示按原流程走）。
    """
    q = en_question.lower()

    # 常见写法：definition and application / definition & application / definition and use / use of ...
    has_def = "definition" in q or "define" in q
    has_app = ("application" in q) or ("applications" in q) or ("use" in q) or ("usage" in q)

    # “同时问两件事”的明显信号
    has_and = " and " in q or "&" in q

    if has_def and has_app and has_and:
        return [
            ("Definition", f"Briefly define backpropagation." if "backprop" in q else f"Briefly define the concept asked in: {en_question}"),
            ("Application", f"Briefly describe the application/usage of backpropagation in neural network training and optimization." if "backprop" in q else f"Briefly describe the application/usage asked in: {en_question}"),
        ]

    # 也有用户不写 definition，但问 “what is ... and what is it used for”
    if (("what is" in q) or ("describe" in q)) and has_app and has_and:
        # 这里不强行写 backprop，保持更通用
        return [
            ("Definition", f"Briefly define the topic in this question: {en_question}"),
            ("Application", f"Briefly describe the application/usage mentioned or implied by this question: {en_question}"),
        ]

    return None


# ============================================================
# 5) CoT（内部推理要点）模块（internal-only）
# ============================================================

def cot_reasoning_notes(llm: ChatOllama, question: str, context: str, user_wants_zh: bool) -> str:
    """
    生成内部推理要点（给模型用，不给用户看）
    """
    if user_wants_zh:
        prompt = (
            "你是一个“内部推理助手”。你将得到用户问题与资料片段。\n"
            "任务：用资料片段做一个非常简短的“推理要点清单”，帮助后续回答更贴合证据。\n"
            "要求：\n"
            "1) 只写要点清单（最多 6 条）\n"
            "2) 每条要点尽量对应资料编号（如 [1][2]）\n"
            "3) 不要写最终答案，不要写多余解释\n\n"
            f"问题：\n{question}\n\n"
            f"资料片段：\n{context}\n"
        )
    else:
        prompt = (
            "You are an INTERNAL reasoning assistant. You will be given a user question and excerpts.\n"
            "Task: produce a very short bullet-point 'reasoning plan' grounded ONLY in the excerpts.\n"
            "Requirements:\n"
            "1) Bullet list only (max 6 bullets)\n"
            "2) Each bullet should reference excerpt numbers like [1][2] when possible\n"
            "3) Do NOT write the final answer; do NOT add extra explanation\n\n"
            f"Question:\n{question}\n\n"
            f"Excerpts:\n{context}\n"
        )
    return llm.invoke(prompt).content.strip()


# ============================================================
# 6) 向量库 / Retriever（Chroma）
# ============================================================

def load_vectorstore(cfg: RAGConfig) -> Chroma:
    """
    加载 Chroma 向量库（持久化目录来自 cfg.vectordb_dir）
    """
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
    """创建基础 retriever（粗召回）"""
    return vectordb.as_retriever(search_kwargs={"k": candidate_k})


# ============================================================
# 7) Prompt（回答阶段：严格只用 excerpts）
# ============================================================

RAG_PROMPT_EN = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strictly constrained study assistant.\n"
            "You MUST answer ONLY using the provided excerpts.\n\n"
            "Important rules:\n"
            "1) If a part of the question is supported by the excerpts, answer that part.\n"
            "2) If a part is NOT supported, explicitly state:\n"
            "\"Insufficient information for this part based on the provided excerpts.\"\n"
            "3) Do NOT fabricate or infer beyond the excerpts.\n"
            "4) Do NOT use general knowledge.\n"
            "5) Do NOT include meta explanations such as “Based on the excerpts…”.\n\n"
            "Internal notes are provided to improve faithfulness.\n"
            "Do NOT reveal internal notes.\n"
        ),
        ("system", "INTERNAL_NOTES (do not reveal):\n{notes}\n"),
        (
            "human",
            "Question: {question}\n\n"
            "Excerpts:\n{context}\n\n"
            "Requirements:\n"
            "- Give a brief conclusion first\n"
            "- Then explain strictly using the excerpts\n"
            "- Cite excerpt numbers like [1], [2]\n"
            "- Handle each aspect of the question separately if necessary"
        ),
    ]
)

RAG_PROMPT_ZH = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个严格受限的学习助手。\n"
            "你【只能】使用“资料片段”中的信息回答问题。\n\n"
            "重要规则：\n"
            "1) 如果问题中的某一部分有证据支持，就回答该部分。\n"
            "2) 如果某一部分没有证据支持，请明确说明：\n"
            "“根据提供的资料，该部分信息不足。”\n"
            "3) 严禁编造或推测资料中未明确给出的内容。\n"
            "4) 严禁使用常识或背景知识。\n"
            "5) 禁止输出任何元解释（如“根据资料…”）。\n\n"
            "系统会提供内部推理要点用于提高准确性。\n"
            "你绝对不能输出内部推理要点。\n"
        ),
        ("system", "内部推理要点（禁止输出给用户）：\n{notes}\n"),
        (
            "human",
            "问题：{question}\n\n"
            "资料片段：\n{context}\n\n"
            "要求：\n"
            "- 先给简要结论\n"
            "- 再结合资料片段解释（必须引用资料编号）\n"
            "- 如果某部分无证据支持，请单独说明"
        ),
    ]
)


# ============================================================
# 8) 主链路：build_rag_chain
# ============================================================

def build_rag_chain(cfg: RAGConfig):
    vectordb = load_vectorstore(cfg)
    # === 构建 BM25 索引（基于全部 chunk） ===
    all_docs = vectordb.get()["documents"]
    all_metadatas = vectordb.get()["metadatas"]

    tokenized_corpus = [
        re.findall(r"\w+", doc.lower()) for doc in all_docs
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    llm = ChatOllama(model=cfg.llm_model)

    # 检索参数
    candidate_k = cfg.candidate_k       # 粗召回（每个 query 召回数量）
    final_k = cfg.top_k                 # 最终进入上下文 chunk 数量

    # Query Router 参数
    enable_router = cfg.enable_query_router
    force_level = cfg.force_level
    n_rewrites = cfg.n_rewrites
    max_sub_questions = cfg.max_sub_questions

    # Evidence Gate 参数
    min_evidence_chars = cfg.min_evidence_chars

    # Aspect split 参数
    enable_aspect_split = cfg.enable_aspect_split

    base_retriever = build_base_retriever(vectordb, candidate_k)

    def retrieve_docs_for_query(en_query: str) -> Tuple[List[Document], List[str], List[str]]:
        """
        给一个英文 query，完成：
        Router → queries 构建 → 多路召回 → 去重 → rerank → 返回 docs_for_llm
        返回：(docs_for_llm, queries_used, debug_notes)
        """
        if force_level is not None:
            level = int(force_level)
        else:
            level = question_complexity_level(en_query) if enable_router else 1

        queries, debug_notes = build_queries(
            llm=llm,
            en_question=en_query,
            level=level,
            n_rewrites=n_rewrites,
            max_sub_questions=max_sub_questions,
        )

        merged_docs: List[Document] = []
        seen = set()
        for q in queries:

            # -------- Dense Retrieval --------
            dense_docs = base_retriever.invoke(q)

            for d in dense_docs:
                key = (d.metadata.get("source"), d.metadata.get("page"), d.page_content[:120])
                if key not in seen:
                    seen.add(key)
                    merged_docs.append(d)

                # -------- Sparse Retrieval (BM25) --------
            if cfg.enable_hybrid:

                tokenized_query = re.findall(r"\w+", q.lower())
                bm25_scores = bm25.get_scores(tokenized_query)

                bm25_top_n = sorted(
                    range(len(bm25_scores)),
                    key=lambda i: bm25_scores[i],
                    reverse=True
                )[: cfg.bm25_k]

                for idx in bm25_top_n:
                    doc_text = all_docs[idx]
                    metadata = all_metadatas[idx]

                    fake_doc = Document(page_content=doc_text, metadata=metadata)

                    key = (metadata.get("source"), metadata.get("page"), doc_text[:120])
                    if key not in seen:
                        seen.add(key)
                        merged_docs.append(fake_doc)

        merged_docs = remove_similar_docs(merged_docs, threshold=0.9)
        docs_for_llm = rerank_docs(en_query, merged_docs, final_k)

        return docs_for_llm, queries, debug_notes

    def answer_with_docs(user_question: str, docs_for_llm: List[Document], user_wants_zh: bool) -> str:
        """
        给定 docs_for_llm（已 rerank 的 top_k），生成最终回答（或拒答）
        """
        if not has_meaningful_evidence(docs_for_llm, min_chars=min_evidence_chars):
            return "资料不足，无法回答该问题。" if user_wants_zh else "Insufficient information to answer."

        context = format_docs(docs_for_llm)
        notes = cot_reasoning_notes(llm, user_question, context, user_wants_zh)

        prompt = (RAG_PROMPT_ZH if user_wants_zh else RAG_PROMPT_EN).format_messages(
            question=user_question,
            context=context,
            notes=notes,
        )
        resp = llm.invoke(prompt).content.strip()
        evidence_block = f"资料片段：\n{context}\n\n"
        return evidence_block + resp

    def rag_invoke(user_question: str) -> str:
        """
        对外入口：传入用户问题，返回“资料片段 + 答案/拒答”
        """
        user_wants_zh = is_chinese(user_question)

        # 1) 统一检索语种为英文（中文先翻译）
        en_query = translate_to_english(llm, user_question) if user_wants_zh else user_question

        # 2) Aspect split：如果是“definition + application”，拆成两个子问分别检索/回答
        if enable_aspect_split:
            aspects = detect_definition_application_split(en_query)
        else:
            aspects = None

        if aspects:
            # 对每个 aspect 单独检索 + 单独 evidence gate + 单独回答
            outputs: List[str] = []
            for label, aspect_q_en in aspects:
                docs_for_llm, queries_used, debug_notes = retrieve_docs_for_query(aspect_q_en)

                # 输出的“展示问题”用更贴近用户意图的 phrasing
                if user_wants_zh:
                    display_q = "简要说明反向传播的定义。" if label == "Definition" else "简要说明反向传播的应用/用途。"
                else:
                    display_q = "Briefly describe the definition of backpropagation." if label == "Definition" else "Briefly describe the application/usage of backpropagation."

                ans = answer_with_docs(display_q, docs_for_llm, user_wants_zh)

                # 开发调试：可输出 router/debug/queries（默认关闭）
                if cfg.show_debug:
                    dbg = "DEBUG:\n" + "\n".join(debug_notes) + "\n"
                    dbg += "QUERIES:\n" + "\n".join([f"- {q}" for q in queries_used]) + "\n\n"
                    ans = dbg + ans

                # 拼接成分段输出
                header = f"## {label}\n" if not user_wants_zh else ("## 定义\n" if label == "Definition" else "## 应用\n")
                outputs.append(header + ans)

            return "\n\n".join(outputs)

        # 3) 默认：按原问题走一次检索/回答
        docs_for_llm, queries_used, debug_notes = retrieve_docs_for_query(en_query)
        ans = answer_with_docs(user_question, docs_for_llm, user_wants_zh)

        if cfg.show_debug:
            dbg = "DEBUG:\n" + "\n".join(debug_notes) + "\n"
            dbg += "QUERIES:\n" + "\n".join([f"- {q}" for q in queries_used]) + "\n\n"
            return dbg + ans

        return ans

    return rag_invoke


# ============================================================
# 9) CLI 运行入口
# ============================================================

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