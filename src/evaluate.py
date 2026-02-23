import json
from pathlib import Path

from config import RAGConfig
from chat import build_rag_chain


def evaluate():

    cfg = RAGConfig()
    rag_chain = build_rag_chain(cfg)

    BASE_DIR = Path(__file__).resolve().parent.parent
    eval_file = BASE_DIR / "evaluate" / "eval_questions.json"

    with open(eval_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    total_should_answer = 0
    total_should_refuse = 0

    correct_answers = 0
    faithful_answers = 0
    correct_refusals = 0
    total_answers = 0

    for item in questions:
        q = item["question"]
        should_answer = item["should_answer"]

        print(f"\n=== {item['id']} ===")
        print("Question:", q)

        response = rag_chain(q)


        print("Model output:", response)

        refused = "资料不足" in response or "无法回答" in response

        # 分类
        if should_answer:
            total_should_answer += 1

            if not refused:
                total_answers += 1

                # 手动标注 correct / faithful（现在先人工判断）
                user_correct = input("Answer correct? (y/n): ")
                user_faithful = input("Faithful to retrieved docs? (y/n): ")

                if user_correct.lower() == "y":
                    correct_answers += 1
                if user_faithful.lower() == "y":
                    faithful_answers += 1
            else:
                print("❌ Should answer but refused")

        else:
            total_should_refuse += 1

            if refused:
                correct_refusals += 1
            else:
                print("❌ Should refuse but answered")

    print("\n========== METRICS ==========")

    accuracy = correct_answers / max(total_should_answer, 1)
    faithfulness = faithful_answers / max(total_answers, 1)
    refusal = correct_refusals / max(total_should_refuse, 1)

    print(f"Answer Accuracy: {accuracy:.3f}")
    print(f"Faithfulness: {faithfulness:.3f}")
    print(f"Refusal Correctness: {refusal:.3f}")


if __name__ == "__main__":
    evaluate()
