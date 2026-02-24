# src/agent.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

from langchain_ollama import OllamaLLM
from langchain_core.documents import Document

from rag_core import RAGConfig, build_retriever, format_docs


ROUTER_PROMPT = """You are a routing controller for a study assistant.
Decide whether the user question requires retrieving from lecture notes (RAG) or not.

Return ONLY one token:
- RAG  (if the question needs specific facts/definitions from the notes, or cites lecture/page concepts)
- NO_RAG (if it is purely conversational, or can be answered safely without notes)

User question:
{question}
"""

ANSWER_PROMPT = """You are StudyCopilot.
You MUST follow the rules:

Rules:
1) Use ONLY the provided CONTEXT to answer the question.
2) If CONTEXT does not contain enough information to answer, say:
   "I don't have enough evidence in the notes to answer that."
3) When you answer, be concise and structured.
4) Add citations like [1], [2] matching the context blocks.

CONTEXT:
{context}

QUESTION:
{question}

Answer:
"""

EVIDENCE_CHECK_PROMPT = """You are an evidence gate.
Given QUESTION and CONTEXT, decide if the context contains sufficient evidence to answer.

Return ONLY one token:
- ENOUGH
- NOT_ENOUGH

Be strict: if key steps/definitions are missing, return NOT_ENOUGH.

QUESTION:
{question}

CONTEXT:
{context}
"""


@dataclass
class AgentConfig:
    ollama_llm_model: str              # e.g., "llama3"
    vectordb_dir: str                  # e.g., "vectordb"
    ollama_embedding_model: str        # e.g., "nomic-embed-text" or "llama3" (if you only have that)
    top_k: int = 4
    temperature: float = 0.2


class StudyCopilotAgent:
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg

        self.llm = OllamaLLM(
            model=cfg.ollama_llm_model,
            temperature=cfg.temperature,
        )

        rag_cfg = RAGConfig(
            persist_dir=cfg.vectordb_dir,
            embedding_model=cfg.ollama_embedding_model,
            top_k=cfg.top_k,
        )
        self.retriever = build_retriever(rag_cfg)

    def _route(self, question: str) -> str:
        resp = self.llm.invoke(ROUTER_PROMPT.format(question=question)).strip().upper()
        if "RAG" in resp and "NO_RAG" not in resp:
            return "RAG"
        if "NO_RAG" in resp:
            return "NO_RAG"
        # fallback: assume RAG for safety
        return "RAG"

    def _evidence_gate(self, question: str, context: str) -> bool:
        resp = self.llm.invoke(EVIDENCE_CHECK_PROMPT.format(question=question, context=context)).strip().upper()
        return resp.startswith("ENOUGH")

    def _retrieve(self, question: str) -> List[Document]:
        return self.retriever.invoke(question)

    def answer(self, question: str) -> Dict[str, Any]:
        """
        Returns:
        {
          "final": str,
          "route": "RAG"|"NO_RAG",
          "citations": [str...],
          "docs_found": int
        }
        """
        route = self._route(question)

        if route == "NO_RAG":
            # still keep it safe: answer briefly, but avoid pretending citations
            final = self.llm.invoke(
                "Answer briefly and safely. If you are unsure, say you are unsure.\n\nQuestion:\n" + question
            ).strip()
            return {"final": final, "route": route, "citations": [], "docs_found": 0}

        docs = self._retrieve(question)
        context, citations = format_docs(docs)

        if not context:
            return {
                "final": "I don't have enough evidence in the notes to answer that.",
                "route": route,
                "citations": [],
                "docs_found": 0,
            }

        ok = self._evidence_gate(question, context)
        if not ok:
            return {
                "final": "I don't have enough evidence in the notes to answer that.",
                "route": route,
                "citations": citations,
                "docs_found": len(docs),
            }

        final = self.llm.invoke(ANSWER_PROMPT.format(context=context, question=question)).strip()
        return {
            "final": final,
            "route": route,
            "citations": citations,
            "docs_found": len(docs),
        }