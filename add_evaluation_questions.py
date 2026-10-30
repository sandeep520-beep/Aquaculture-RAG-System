import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
QUESTIONS_FILE = DATA_DIR / "evaluation_questions.json"
UNANSWERABLE_FILE = DATA_DIR / "unanswerable_candidates.json"

# Each paraphrase inherits source/page/reference answer from its original.
PARAPHRASES = [
    (1, "How much of the operating costs in finfish and prawn farming may be spent on feed?"),
    (2, "When deciding how much feed to give, what should a farmer take into account?"),
    (3, "What are the consequences of giving cultured fish or prawns more feed than they eat?"),
    (4, "In what ways may low-grade aquaculture feed harm pond water quality?"),
    (5, "How should feeding be adjusted if dissolved oxygen falls below a suitable level?"),
    (6, "What four groups of factors influence the cost and efficiency of aquaculture feed?"),
    (7, "How does the Nile tilapia guide describe aquaculture using the FAO definition?"),
    (8, "Which management actions are examples of human intervention in tilapia farming?"),
    (9, "Which changes in water conditions can place stress on aquatic animals, according to the manual?"),
    (10, "What roles do fish and other aquatic foods play in African nutrition and livelihoods?"),
    (11, "What can happen to water quality when feed remains uneaten in a pond?"),
    (12, "In the described Nile tilapia case, why might feeding fish at half satiation be useful?"),
    (13, "What happens to dissolved oxygen as water gets warmer, saltier, or deeper?"),
    (14, "Which pond-management measures can help raise low dissolved oxygen in Nile tilapia culture?"),
    (15, "Why should ammonia accumulation be controlled in tilapia ponds?"),
    (16, "What is the relationship between dissolved oxygen, feeding activity, and Nile tilapia growth?"),
    (17, "Which water-quality measurements are recommended for monitoring a Nile tilapia pond?"),
    (18, "At what temperature do Nile tilapia feed and grow best, and what range can they tolerate?"),
    (19, "How does replacing some pond water help manage water quality for Nile tilapia?"),
    (20, "What effects can unsuitable pond water have on Nile tilapia health and growth?"),
]

# Candidates only; verify against all four PDFs before calling them unanswerable.
UNANSWERABLE_CANDIDATES = [
    "What was the exact retail price of Nile tilapia feed in Varanasi on 1 January 2026?",
    "What is the current market price per kilogram of Nile tilapia in Kharagpur today?",
    "Which individual farmer in Varanasi had the highest tilapia yield in 2025?",
    "What will the dissolved oxygen level in a specific pond be at 8:00 tomorrow morning?",
    "What is the exact electricity bill for running an aerator in a particular farm last month?",
    "Which brand of smartphone was used by the author to take photographs for the Nile tilapia guide?",
    "What is the name and personal telephone number of the owner of the largest tilapia farm in West Bengal?",
    "How many Nile tilapia are currently alive in a specific farm pond that is not described in the documents?",
    "What was the exact rainfall measured at a particular farm in Kharagpur yesterday?",
    "Which aquaculture company will offer the highest internship stipend next summer?",
]

def main():
    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(f"Could not find {QUESTIONS_FILE}")
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    if not isinstance(questions, list):
        raise ValueError("evaluation_questions.json must contain a JSON list.")

    existing_ids = {item.get("id") for item in questions}
    existing_text = {item.get("question", "").strip().casefold() for item in questions}
    by_id = {item.get("id"): item for item in questions}
    added = 0

    for original_id, paraphrase in PARAPHRASES:
        new_id = 20 + original_id
        if new_id in existing_ids or paraphrase.casefold() in existing_text:
            continue
        original = by_id.get(original_id)
        if original is None:
            raise ValueError(f"Original question ID {original_id} is missing.")
        questions.append({
            "id": new_id,
            "question": paraphrase,
            "relevant_sources": list(original["relevant_sources"]),
            "relevant_pages": list(original["relevant_pages"]),
            "reference_answer": original["reference_answer"],
        })
        added += 1

    questions.sort(key=lambda item: item["id"])
    QUESTIONS_FILE.write_text(
        json.dumps(questions, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    candidates = [
        {
            "id": index,
            "question": question,
            "status": "candidate_not_yet_verified",
            "note": "Search all four PDFs before marking this question unanswerable.",
        }
        for index, question in enumerate(UNANSWERABLE_CANDIDATES, start=1)
    ]
    UNANSWERABLE_FILE.write_text(
        json.dumps(candidates, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Added {added} paraphrased questions.")
    print(f"Total answerable retrieval questions: {len(questions)}")
    print(f"Updated dataset: {QUESTIONS_FILE}")
    print(f"Created {len(candidates)} unverified unanswerable-question candidates:")
    print(UNANSWERABLE_FILE)
    print("\nImportant: candidates are NOT yet verified as unanswerable.")
    print("Next, run: python src\\evaluation.py")

if __name__ == "__main__":
    main()
