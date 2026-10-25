"""Run every question in eval/questions.json and score retrieval and answers separately.

Run from the project root:
    python -m eval.run_eval            # saves eval/results/baseline.json
    python -m eval.run_eval chroma     # saves eval/results/chroma.json, to compare runs
"""
import json
import sys

from src import config
from src.generate import ask_llm
from src.pipeline import answer, build_index

QUESTIONS = config.ROOT / "eval" / "questions.json"
RESULTS_DIR = config.ROOT / "eval" / "results"
NO_ANSWER = "don't know"

JUDGE_PROMPT = """You are grading an answer against the expected answer.
Reply with exactly one word: CORRECT if the answer contains the key facts of the expected answer, otherwise INCORRECT.

Question: {question}
Expected answer: {expected}
Answer to grade: {actual}
Grade:"""


def retrieval_hit(q: dict, results: list[dict]) -> bool:
    """Did the top-k chunks include every document the answer needs?"""
    retrieved = {r["source"] for r in results}
    return all(doc in retrieved for doc in q["source_docs"])


def answer_correct(q: dict, actual: str) -> bool:
    if q["type"] == "unanswerable":
        return NO_ANSWER in actual.lower()  # it should refuse
    # LLM-as-judge: the model grades whether the answer matches the expected one
    grade = ask_llm(JUDGE_PROMPT.format(question=q["question"], expected=q["expected_answer"], actual=actual))
    return grade.strip().upper().startswith("CORRECT")


def main(run_name: str) -> None:
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    store = build_index()

    rows = []
    for q in questions:
        out = answer(q["question"], store)
        row = {
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "expected": q["expected_answer"],
            "answer": out["answer"],
            "retrieved": [r["id"] for r in out["results"]],
            "retrieval_hit": retrieval_hit(q, out["results"]) if q["source_docs"] else None,
            "correct": answer_correct(q, out["answer"]),
        }
        rows.append(row)
        print(f"{row['id']:4} {row['type']:12} retrieval={str(row['retrieval_hit']):5} correct={row['correct']}")

    answerable = [r for r in rows if r["retrieval_hit"] is not None]
    hits = sum(r["retrieval_hit"] for r in answerable)
    correct = sum(r["correct"] for r in rows)
    print(f"\nRetrieval hit@{config.TOP_K}: {hits}/{len(answerable)}")
    print(f"Answers correct:  {correct}/{len(rows)}")

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{run_name}.json"
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved details to {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "baseline")
