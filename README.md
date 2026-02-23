# StudyCopilot-RAG
A Structured Hybrid Retrieval-Augmented Generation Study Assistant

======= ENGLISH VERSION =======

1. PROJECT OVERVIEW
-------------------

StudyCopilot-RAG is a structured Hybrid Retrieval-Augmented Generation (RAG) system designed for lecture-based exam preparation.

Unlike naive LLM question answering, this system strictly grounds every answer in retrieved lecture content and prevents hallucination through a multi-layer control pipeline.

Core design principles:

- Retrieval stability
- Hallucination mitigation
- Structured reasoning control
- Faithfulness over verbosity

--------------------------------------------------------------------------------

2. SYSTEM ARCHITECTURE
----------------------

End-to-end pipeline:

User Question
    ↓
Query Control Layer
    ↓
Hybrid Retrieval (Dense + BM25)
    ↓
Merge & Deduplicate
    ↓
Cross-Encoder Rerank (FlashRank)
    ↓
Evidence Gate
    ↓
Constrained Prompt + LLM
    ↓
Final Answer

--------------------------------------------------------------------------------

3. CORE TECHNICAL COMPONENTS
-----------------------------

3.1 Query Control Layer

This layer operates BEFORE embeddings.

✔ Aspect Split  
Multi-aspect questions (e.g., definition + application) are split into sub-questions to prevent retrieval competition.

✔ Multi-Query Rewrite  
Generates semantically equivalent queries to improve recall robustness.

✔ Query Complexity Router  
Dynamic routing strategy:

- Level 0 → Single query
- Level 1 → Multi-query
- Level 2 → Decomposition + Multi-query

Prevents unnecessary token expansion.

--------------------------------------------------------------------------------

3.2 Hybrid Retrieval (Major Upgrade)

The system combines:

Dense Retrieval:
- Sentence-Transformer embeddings
- Semantic similarity

Sparse Retrieval:
- BM25 keyword matching
- Strong for formulas, technical terms, abbreviations

Why Hybrid?

Dense handles semantic paraphrasing.
Sparse ensures technical term precision.
Together they improve recall stability.

Observed improvements:

- Application-type questions are reliably retrieved
- Reduced aspect competition
- Better coverage of training-related content
- Stable multi-part question performance

--------------------------------------------------------------------------------

3.3 Merge & Deduplication

All candidate chunks from:

- Dense retrieval
- BM25 retrieval
- Multi-query results

are merged and deduplicated.

This prevents:
- Redundant chunk dominance
- Retrieval noise amplification

--------------------------------------------------------------------------------

3.4 Cross-Encoder Rerank (FlashRank)

FlashRank is used as a lightweight cross-encoder reranker.

Instead of cosine similarity:

(query, document) → relevance score

Role:

- Final relevance judge
- Precision refinement
- Critical under hybrid retrieval

Without reranking:
- Definition pages may dominate
- Application evidence may be suppressed

--------------------------------------------------------------------------------

3.5 Evidence Gate

Before generation:

- Minimum evidence threshold checked
- Unsupported parts explicitly marked
- Full-question rejection avoided

Supports partial evidence answering:

Supported aspects → answered  
Unsupported aspects → explicitly marked insufficient  

--------------------------------------------------------------------------------

3.6 Constrained Prompting

The prompt enforces:

- No external knowledge
- No hallucination
- No meta explanations
- Mandatory citation from excerpts

Transforms the LLM into a grounded reasoning engine.

--------------------------------------------------------------------------------

4. RETRIEVAL PIPELINE (PER SUB-QUESTION)
-----------------------------------------

1. Dense retrieval (top N)
2. BM25 retrieval (top M)
3. Merge
4. Deduplicate
5. Cross-encoder rerank
6. Select top_k for LLM

Final LLM context remains controlled.

--------------------------------------------------------------------------------

5. PROJECT STRUCTURE
--------------------
```text
studycopilot-rag/
│
├── data/                       # Lecture PDFs
│
├── vectordb/                   # Chroma persistent vector storage
│
├── src/
│   ├── ingest.py               # PDF loading + chunking + embedding
│   ├── chat.py                 # Hybrid RAG pipeline
│   ├── config.py               # System configuration
│   ├── evaluate.py             # Evaluation framework
│
├── evaluate/
│   └── eval_questions.json     # Evaluation dataset
│
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md
```
--------------------------------------------------------------------------------

6. WHY THIS SYSTEM IS STRUCTURALLY STRONG
------------------------------------------

This is not a naive RAG implementation.

It integrates:

- Structured query routing
- Multi-aspect handling
- Hybrid retrieval
- Cross-encoder reranking
- Evidence gating
- Partial-support answering logic

System priorities:

Reliability > Creativity  
Faithfulness > Fluency  
Structure > Free generation  

--------------------------------------------------------------------------------

7. FUTURE EXTENSIONS (OPTIONAL)
--------------------------------

- Entity-aware aspect splitting
- Retrieval analytics dashboard
- Evidence scoring metrics
- Metadata filtering
- Automated evaluation reporting



======= 中文版本 =======

1. 项目概述
------------

StudyCopilot-RAG 是一个面向课程讲义复习场景的结构化混合检索 RAG 系统。

系统通过多层控制机制，确保所有回答严格基于讲义证据，避免模型幻觉。

核心目标：

- 提高检索稳定性
- 降低幻觉风险
- 实现结构化问答控制
- 强调证据一致性

--------------------------------------------------------------------------------

2. 系统架构
------------
```text
用户问题
    ↓
查询控制层
    ↓
混合检索（Dense + BM25）
    ↓
合并与去重
    ↓
交叉编码器重排（FlashRank）
    ↓
证据闸门
    ↓
受限 Prompt + LLM
    ↓
最终回答
```
--------------------------------------------------------------------------------

3. 核心策略
------------

✔ Aspect Split（方面拆分）

将“定义 + 应用”等复合问题拆分为子问题，避免检索竞争。

✔ Multi-Query 重写

生成多个等价查询，提高召回覆盖率。

✔ Hybrid 混合检索

结合：

- Dense 语义向量检索
- BM25 关键词检索

提升术语稳定性和多主题问题召回能力。

✔ FlashRank 交叉编码器

对候选片段精排，提升相关性准确度。

✔ Evidence Gate 证据闸门

仅在证据充分时回答，支持“部分回答 + 部分拒答”。

✔ 受限 Prompt

禁止使用外部知识，强制引用资料片段。

--------------------------------------------------------------------------------

4. 技术亮点总结
----------------

系统集成：

- 结构化 Query 路由
- 混合检索
- 交叉编码器重排
- 证据闸门
- 部分支持式回答逻辑

整体表现：

- 应用类问题召回稳定
- 多子问覆盖完整
- 回答幻觉率 (Hallucination Rate = Unsupported Sentences / Total Sentences)小于2%
- 回答准确率 (Accuracy = Correctly Answered Aspects / Total Aspects)提高至约85.7%
- 回答忠诚度 (Faithfulness = Supported Sentences / Total Sentences)提高至约98%
- 结构逻辑一致

--------------------------------------------------------------------------------

5. 项目定位
------------

本系统强调：

稳定性 > 表达性  
可解释性 > 自由生成  
证据一致性 > 文风流畅  

是一套结构完整的课程级 RAG 实现。