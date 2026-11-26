import argparse
import json
import time
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from graph import graph, llm


class JudgeResult(BaseModel):
    correct: bool


def judge_answer(question: str, expected: str, actual: str, q_type: str) -> bool:
    if q_type == "unanswerable":
        return "don't have enough reliable information" in actual.lower()
    if q_type in ("off_topic", "injection"):
        return any(w in actual.lower() for w in ["cannot", "only answer", "union budget"])

    prompt = (
        "Compare the actual answer to the expected answer for the question.\n"
        f"Question: {question}\n"
        f"Expected: {expected}\n"
        f"Actual: {actual}\n"
        "Return correct=True if the actual answer conveys the same core facts/numbers as expected (minor rounding/formatting differences are fine), else False."
    )
    res = llm.with_structured_output(JudgeResult).invoke(prompt)
    return res.correct


def check_citation(gold_source: str, sources: list[str], answer: str) -> bool:
    if gold_source == "none":
        return len(sources) == 0
    combined = " ".join(sources).lower() + " " + answer.lower()
    if gold_source == "budget_table.csv":
        return "budget_table.csv" in combined
    return any(
        doc in combined
        for doc in ["prs_analysis.pdf", "pib_summary.pdf", "budget_speech.docx", "bag1.pdf", "hkeybud2024.pdf"]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    args = parser.parse_args()

    with open(args.questions, "r", encoding="utf-8") as f:
        questions = json.load(f)

    results = []
    for item in questions:
        t0 = time.time()
        config = {"configurable": {"thread_id": item["thread_id"]}}
        out = graph.invoke(
            {"messages": [HumanMessage(content=item["question"])]},
            config=config,
        )
        latency = round(time.time() - t0, 2)

        actual_answer = out["messages"][-1].content
        sources = out.get("sources", [])
        actual_route = "refuse" if out.get("intent") != "valid" else out.get("route", "")

        ans_ok = judge_answer(item["question"], item["expected_answer"], actual_answer, item["type"])
        cit_ok = check_citation(item["gold_source"], sources, actual_answer)
        route_ok = actual_route == item["expected_route"]

        results.append({
            "id": item["id"],
            "type": item["type"],
            "question": item["question"],
            "expected_route": item["expected_route"],
            "actual_route": actual_route,
            "route_ok": route_ok,
            "ans_ok": ans_ok,
            "cit_ok": cit_ok,
            "latency": latency,
            "actual_answer": actual_answer.replace("\n", " ")[:120],
        })
        print(f"Q{item['id']:02d} [{item['type']}] | Route: {actual_route} ({route_ok}) | Ans: {ans_ok} | Cit: {cit_ok} | {latency}s")
        time.sleep(1)

    total = len(results)
    ans_acc = sum(r["ans_ok"] for r in results) / total * 100
    cit_acc = sum(r["cit_ok"] for r in results) / total * 100
    route_acc = sum(r["route_ok"] for r in results) / total * 100
    avg_lat = sum(r["latency"] for r in results) / total

    print("\n=== EVALUATION SUMMARY ===")
    print(f"Answer Accuracy:     {ans_acc:.1f}% ({sum(r['ans_ok'] for r in results)}/{total})")
    print(f"Citation Correctness:{cit_acc:.1f}% ({sum(r['cit_ok'] for r in results)}/{total})")
    print(f"Routing Correctness: {route_acc:.1f}% ({sum(r['route_ok'] for r in results)}/{total})")
    print(f"Average Latency:     {avg_lat:.2f}s")

    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved detailed results to eval_results.json")


if __name__ == "__main__":
    main()