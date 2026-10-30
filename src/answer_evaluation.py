
import csv
import json
import re
from pathlib import Path

import ollama

from ask_rag import (
    LLM_MODEL,
    load_resources,
    retrieve_passages,
    build_context,
    generate_answer,
)


# ------------------------------------------------------------
# FILE PATHS
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

QUESTIONS_FILE = PROJECT_DIR / "data" / "evaluation_questions.json"
OUTPUT_FILE = PROJECT_DIR / "experiments" / "answer_evaluation.csv"


# ------------------------------------------------------------
# CHECK CITATION LABELS
# ------------------------------------------------------------

def check_citations(answer, sources):
    # Recognize [S1], [S1, Page 4], and similar formats.
    used_labels = set(
        re.findall(r"\[(S\d+)(?:[^\]]*)\]", answer)
    )

    context = build_context(sources)

    # Source labels available in the retrieved context.
    valid_labels = set(
        re.findall(r"\[(S\d+)\]", context)
    )

    if not used_labels:
        return "NO_CITATIONS"

    if used_labels.issubset(valid_labels):
        return "VALID_LABELS"

    return "INVALID_LABELS"


# ------------------------------------------------------------
# EVALUATE ONE ANSWER
# ------------------------------------------------------------


def evaluate_answer(question, reference_answer, answer, context):

    prompt = f"""
You are an evidence-based evaluator for an aquaculture RAG system.

Your most important task is to compare the MEANING of the
generated answer with the reference answer.

QUESTION:
{question}

REFERENCE ANSWER:
{reference_answer}

GENERATED ANSWER:
{answer}

SUPPLIED EVIDENCE:
{context}

Follow these steps carefully:

1. Identify the essential facts in the reference answer.
2. For each essential fact, check whether the generated answer
   states that fact directly or expresses the same meaning.
3. Do NOT say a fact is missing if the generated answer actually
   includes it, even if it uses different wording.
4. Extra relevant details are acceptable if supported by evidence.
5. Penalize unsupported claims, factual contradictions, and
   important missing facts.
6. Check that the answer addresses the actual question.
7. Use the supplied evidence to assess factual support.
8. Do not require direct quotations from the evidence.

IMPORTANT EXAMPLES:
- If the reference mentions biomass and daily feed intake,
  and the answer mentions both, do not report either as missing.
- If the answer says "depending on the severity of the situation"
  and the reference says the same, do not report severity as missing.
- If an answer mentions the correct number but then contradicts
  that number, identify the contradiction.

Choose exactly one grade:

"correct": All essential facts are present, with no important
contradictions or unsupported central claims.

"partial": Some essential facts are missing, or there are minor
errors or unsupported details.

"incorrect": The answer fails to answer the question, omits most
essential facts, or contains a serious contradiction.

Return JSON with these keys:
{{
  "grade": "correct",
  "reason": "One concise sentence explaining the decision.",
  "matched_facts": ["Essential facts found in the answer"],
  "missing_facts": ["Essential facts genuinely missing"],
  "contradictions": ["Important contradictions, if any"]
}}

Use empty lists when there are no missing facts or contradictions.
Return JSON only. Do not use Markdown fences.
"""

    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful evaluator. Compare meaning, "
                        "check the answer before identifying missing facts, "
                        "and return valid JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            format="json",
            options={
                "temperature": 0,
                "num_predict": 400,
            },
        )

        raw_text = response["message"]["content"].strip()
        result = json.loads(raw_text)

        if not isinstance(result, dict):
            raise ValueError("Evaluator response is not a JSON object.")

        grade = str(result.get("grade", "")).strip().lower()

        grade_mapping = {
            "correct": (5, "PASS"),
            "partial": (3, "REVIEW"),
            "incorrect": (1, "FAIL"),
        }

        if grade not in grade_mapping:
            raise ValueError(
                f"Unexpected evaluator grade: {grade!r}"
            )

        score, verdict = grade_mapping[grade]

        reason = str(
            result.get("reason", "No explanation provided.")
        )

        matched = result.get("matched_facts", [])
        missing = result.get("missing_facts", [])
        contradictions = result.get("contradictions", [])

        if not all(
            isinstance(value, list)
            for value in (matched, missing, contradictions)
        ):
            raise ValueError(
                "matched_facts, missing_facts, and contradictions "
                "must all be JSON lists."
            )

        # Include the judge's fact checklist in the saved explanation.
        reason += (
            " | Matched facts: "
            + ("; ".join(map(str, matched)) if matched else "None listed")
        )
        reason += (
            " | Missing facts: "
            + ("; ".join(map(str, missing)) if missing else "None listed")
        )
        reason += (
            " | Contradictions: "
            + (
                "; ".join(map(str, contradictions))
                if contradictions else "None listed"
            )
        )

        return {
            "score": score,
            "verdict": verdict,
            "reason": reason,
        }

    except Exception as error:
        print("\nEvaluator error:", str(error))

        return {
            "score": 0,
            "verdict": "EVALUATION_ERROR",
            "reason": str(error),
        }

# ------------------------------------------------------------
# MAIN EVALUATION
# ------------------------------------------------------------

def main():

    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation questions not found: {QUESTIONS_FILE}"
        )

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
        questions = json.load(file)

    if not isinstance(questions, list):
        raise ValueError(
            "evaluation_questions.json must contain a list of questions."
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("Loading aquaculture knowledge base...")

    embedding_model, collection = load_resources()

    results = []

    for index, item in enumerate(questions, start=1):

        question = item["question"]
        reference_answer = item["reference_answer"]

        print("\n" + "=" * 60)
        print(f"Evaluating question {index}/{len(questions)}")
        print(question)

        try:
            sources = retrieve_passages(
                question,
                embedding_model,
                collection,
            )

            context = build_context(sources)

            answer = generate_answer(question, context)

            citation_status = check_citations(answer, sources)

            evaluation = evaluate_answer(
                question,
                reference_answer,
                answer,
                context,
            )

            row = {
                "question_id": item.get("id", index),
                "question": question,
                "reference_answer": reference_answer,
                "generated_answer": answer,
                "score": evaluation["score"],
                "verdict": evaluation["verdict"],
                "evaluation_reason": evaluation["reason"],
                "citation_status": citation_status,
            }

            print("\nGenerated answer:")
            print(answer)
            print("\nScore:", evaluation["score"], "/ 5")
            print("Verdict:", evaluation["verdict"])
            print("Citation check:", citation_status)
            print("Reason:", evaluation["reason"])

        except Exception as error:
            row = {
                "question_id": item.get("id", index),
                "question": question,
                "reference_answer": reference_answer,
                "generated_answer": "",
                "score": 0,
                "verdict": "EVALUATION_ERROR",
                "evaluation_reason": str(error),
                "citation_status": "NOT_CHECKED",
            }

            print("ERROR:", error)

        results.append(row)

    # Save all results to CSV.
    fieldnames = [
        "question_id",
        "question",
        "reference_answer",
        "generated_answer",
        "score",
        "verdict",
        "evaluation_reason",
        "citation_status",
    ]

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(results)

    # Calculate summary using successfully evaluated answers.
    valid_results = [
        row for row in results
        if row["score"] in range(1, 6)
    ]

    passed = sum(
        row["verdict"] == "PASS" for row in results
    )

    review = sum(
        row["verdict"] == "REVIEW" for row in results
    )

    failed = sum(
        row["verdict"] == "FAIL" for row in results
    )

    errors = sum(
        row["verdict"] == "EVALUATION_ERROR"
        for row in results
    )

    if valid_results:
        average_score = sum(
            row["score"] for row in valid_results
        ) / len(valid_results)
    else:
        average_score = 0

    print("\n" + "=" * 60)
    print("ANSWER EVALUATION SUMMARY")
    print("=" * 60)
    print("Total questions:", len(results))
    print("Successfully scored:", len(valid_results))
    print(f"Average score: {average_score:.2f}/5")
    print("PASS:", passed)
    print("REVIEW:", review)
    print("FAIL:", failed)
    print("Evaluation errors:", errors)
    print("Results saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
