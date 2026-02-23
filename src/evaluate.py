import json
import re
from pathlib import Path
from datetime import datetime

from config import RAGConfig
from chat import build_rag_chain
from langchain_ollama import ChatOllama


REFUSAL_MARKERS = [
    "资料不足，无法回答该问题。",
    "资料不足",
    "无法回答",
    "证据不足",
    "Insufficient information to answer.",
    "Insufficient information",
]

EVIDENCE_HEADER_ZH = "资料片段："


# -------------------------
# 解析 rag_invoke 输出：contexts + answer
# -------------------------
def _is_refused(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True
    return any(m.lower() in t for m in REFUSAL_MARKERS)


def _strip_debug_prefix(s: str) -> str:
    if not s:
        return ""
    idx = s.find(EVIDENCE_HEADER_ZH)
    return s[idx:] if idx != -1 else s


def _extract_all_evidence_blocks(s: str):
    """
    Parse chat output of the form:

    资料片段：
    [1] (...)
    [2] (...)
    ...
    <answer starts here>

    Supports aspect split: multiple blocks can appear.
    Returns (contexts, answer_text)
    """
    if not s:
        return [], ""

    s = _strip_debug_prefix(s)

    if EVIDENCE_HEADER_ZH not in s:
        return [], s.strip()

    contexts = []
    answer_parts = []

    # split by each evidence header occurrence
    parts = s.split(EVIDENCE_HEADER_ZH)
    # parts[0] is prefix before first evidence header
    prefix = parts[0].strip()
    if prefix:
        answer_parts.append(prefix)

    for block in parts[1:]:
        block = block.lstrip("\n\r\t ").rstrip()
        if not block:
            continue

        lines = block.splitlines()
        ctx_lines = []
        ans_lines = []

        in_ctx = True
        for line in lines:
            # context lines look like: [1] (source, page x) ...
            if in_ctx and re.match(r"^\[\d+\]\s", line.strip()):
                ctx_lines.append(line.rstrip())
            else:
                in_ctx = False
                ans_lines.append(line.rstrip())

        ctx = "\n".join([l for l in ctx_lines if l.strip()]).strip()
        ans = "\n".join([l for l in ans_lines if l.strip()]).strip()

        if ctx:
            contexts.append(ctx)
        if ans:
            answer_parts.append(ans)

    merged_answer = "\n\n".join([p for p in answer_parts if p.strip()]).strip()
    return contexts, merged_answer

def _call_rag(rag_chain, question: str) -> str:
    """
    rag_chain is your rag_invoke (callable), returns str
    """
    return rag_chain(question)


# -------------------------
# Judge (pure LLM) helpers
# -------------------------
def _safe_parse_json(text: str):
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _judge(judge_llm: ChatOllama, task: str, question: str, answer: str, context: str):
    """
    task in {"faithfulness","accuracy"}
    Output strictly JSON
    """
    prompt = f"""
You are a strict evaluator for a Retrieval-Augmented Generation (RAG) system.

Rules:
- Use ONLY the provided CONTEXT as evidence.
- If the answer includes any claim not supported by CONTEXT, mark it NOT faithful.
- If the answer does not correctly address the QUESTION based on CONTEXT, mark it NOT accurate.
- Output ONLY valid JSON. No markdown. No extra text.

TASK: {task}

QUESTION:
{question}

ANSWER:
{answer}

CONTEXT:
{context}

JSON schema:
- If TASK=faithfulness: {{"faithfulness": 0 or 1, "reason": "short"}}
- If TASK=accuracy:     {{"accuracy": 0 or 1, "reason": "short"}}
""".strip()

    # ChatOllama expects messages or string depending on your setup; .invoke(str) works in most cases
    resp = judge_llm.invoke(prompt)
    text = resp.content if hasattr(resp, "content") else str(resp)

    obj = _safe_parse_json(text)
    if isinstance(obj, dict):
        return obj

    # parse fail -> conservative 0
    if task == "faithfulness":
        return {"faithfulness": 0, "reason": "judge_json_parse_failed"}
    return {"accuracy": 0, "reason": "judge_json_parse_failed"}


def compute_faithfulness(answer: str, context: str, judge_llm: ChatOllama, question: str):
    if not answer or _is_refused(answer) or not (context or "").strip():
        return None
    v = _judge(judge_llm, "faithfulness", question, answer, context).get("faithfulness")
    return v if v in (0, 1) else None


def compute_aspect_accuracy(answer: str, context: str, judge_llm: ChatOllama, question: str, should_answer: bool):
    if not should_answer:
        return None
    if not answer:
        return 0
    if _is_refused(answer):
        return 0
    if not (context or "").strip():
        return None
    v = _judge(judge_llm, "accuracy", question, answer, context).get("accuracy")
    return v if v in (0, 1) else None


def compute_hallucination(answer: str, context: str, judge_llm: ChatOllama, question: str):
    f = compute_faithfulness(answer, context, judge_llm, question)
    if f is None:
        return None
    return 0 if f == 1 else 1


# -------------------------
# 1) 运行问答
# -------------------------
def run_evaluation():
    cfg = RAGConfig()

    rag_chain = build_rag_chain(cfg)                 # ✅ RAG answering
    judge_llm = ChatOllama(model=cfg.llm_model)      # ✅ pure LLM judge

    base_dir = Path(__file__).resolve().parent.parent
    out_dir = base_dir / "evaluate"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 找 eval_questions.json（项目内优先，其次 /mnt/data）
    candidates = [
        out_dir / "eval_questions.json",
        Path("/mnt/data/eval_questions.json"),
    ]
    eval_file = next((p for p in candidates if p.exists()), None)
    if eval_file is None:
        raise FileNotFoundError("Cannot find eval_questions.json (tried project evaluate/ and /mnt/data).")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"eval_results_{ts}.jsonl"

    with open(eval_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    records = []
    with open(out_path, "w", encoding="utf-8") as fout:
        for item in questions:
            qid = item.get("id", "")
            q = item["question"]
            should_answer = bool(item["should_answer"])

            print(f"\n=== {qid} ===")
            print("Question:", q)

            raw_text = _call_rag(rag_chain, q)  # ✅ should return str

            # 解析
            ctx_list, answer_text = _extract_all_evidence_blocks(raw_text)
            merged_context = "\n\n".join(ctx_list).strip()

            refused = _is_refused(raw_text) or _is_refused(answer_text)

            print("Model output:", answer_text if answer_text else raw_text)

            # 自动评分
            acc = compute_aspect_accuracy(answer_text, merged_context, judge_llm, q, should_answer)
            faithful = compute_faithfulness(answer_text, merged_context, judge_llm, q)
            hallu = compute_hallucination(answer_text, merged_context, judge_llm, q)

            rec = {
                "id": qid,
                "question": q,
                "should_answer": should_answer,
                "raw_output": raw_text,
                "answer": answer_text,
                "context": merged_context,
                "refused": refused,
                "aspect_accuracy": acc,
                "faithfulness": faithful,
                "hallucination": hallu,
            }

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            records.append(rec)

    print(f"\nSaved per-item results to: {out_path}")
    print(f"Loaded questions from: {eval_file}")
    return records


# -------------------------
# 3) 汇总
# -------------------------
def main():
    records = run_evaluation()

    total_should_answer = sum(1 for r in records if r["should_answer"] is True)
    total_should_refuse = sum(1 for r in records if r["should_answer"] is False)

    correct_refusals = sum(1 for r in records if (r["should_answer"] is False and r["refused"] is True))
    refusal_correctness = correct_refusals / max(total_should_refuse, 1)

    scored_acc = [r["aspect_accuracy"] for r in records if r["should_answer"] is True and r["aspect_accuracy"] in (0, 1)]
    strict_correct = sum(1 for r in records if r["should_answer"] is True and r["aspect_accuracy"] == 1)
    accuracy_strict = strict_correct / max(total_should_answer, 1)
    accuracy_scored = (sum(scored_acc) / max(len(scored_acc), 1)) if scored_acc else 0.0

    scored_f = [r["faithfulness"] for r in records if r["faithfulness"] in (0, 1)]
    faithfulness = (sum(scored_f) / max(len(scored_f), 1)) if scored_f else 0.0

    scored_h = [r["hallucination"] for r in records if r["hallucination"] in (0, 1)]
    hallucination_rate = (sum(scored_h) / max(len(scored_h), 1)) if scored_h else 0.0

    print("\n========== METRICS ==========")
    print(f"Answer Accuracy (strict):       {accuracy_strict:.3f}")
    print(f"Answer Accuracy (scored-only):  {accuracy_scored:.3f}")
    print(f"Faithfulness:                   {faithfulness:.3f}")
    print(f"Refusal Correctness:            {refusal_correctness:.3f}")
    print(f"Hallucination Rate (optional):  {hallucination_rate:.3f}")


if __name__ == "__main__":
    main()