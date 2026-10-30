
from pathlib import Path
import json

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "chroma_db"
QUESTIONS_FILE = BASE_DIR / "data" / "evaluation_questions.json"
OUTPUT_FILE = BASE_DIR / "experiments" / "retrieval_results.csv"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "aquaculture_documents"
TOP_K = 5


def main():
    # 1. Check the evaluation dataset
    if not QUESTIONS_FILE.exists():
        print(f"Evaluation questions file not found: {QUESTIONS_FILE}")
        return

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
        questions = json.load(file)

    if not questions:
        print("The evaluation dataset is empty.")
        return

    # 2. Connect to the existing ChromaDB index
    client = chromadb.PersistentClient(path=str(DB_DIR))

    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        print("Index not found. Check your ChromaDB collection.")
        return

    total_chunks = collection.count()

    if total_chunks == 0:
        print("The collection is empty.")
        return

    # 3. Load the same embedding model used for retrieval
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    results_rows = []

    # 4. Evaluate each question
    for item in questions:
        question_id = item["id"]
        question = item["question"]

        relevant_sources = set(item.get("relevant_sources", []))
        relevant_pages = set(item.get("relevant_pages", []))

        if not relevant_sources or not relevant_pages:
            print(
                f"Skipping question {question_id}: "
                "relevant_sources or relevant_pages is missing."
            )
            continue

        query_embedding = model.encode(
            [question],
            normalize_embeddings=True
        )

        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=min(TOP_K, total_chunks),
            include=["metadatas", "distances", "documents"]
        )

        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        documents = results["documents"][0]

        relevant_rank = None
        relevant_source_found = None
        relevant_page_found = None

        # A result counts as relevant only when both its
        # source filename and page match the expected labels.
        for rank, metadata in enumerate(metadatas, start=1):
            source = metadata.get("source", "")
            page = metadata.get("page")

            if source in relevant_sources and page in relevant_pages:
                relevant_rank = rank
                relevant_source_found = source
                relevant_page_found = page
                break

                # Exact source-and-page match
        hit_at_5 = int(relevant_rank is not None)

        # Source-only match: did any expected PDF appear in the top 5?
        source_hit_at_5 = int(
            any(
                metadata.get("source", "") in relevant_sources
                for metadata in metadatas
            )
        )

        reciprocal_rank = (
            1 / relevant_rank if relevant_rank is not None else 0.0
        )

        retrieved_sources = [
            m.get("source", "Unknown") for m in metadatas
        ]
        retrieved_pages = [
            m.get("page", "Unknown") for m in metadatas
        ]

        row = {
            "question_id": question_id,
            "question": question,
            "expected_sources": " | ".join(sorted(relevant_sources)),
            "expected_pages": ",".join(
                map(str, sorted(relevant_pages))
            ),
            "retrieved_sources": " | ".join(retrieved_sources),
            "retrieved_pages": ",".join(map(str, retrieved_pages)),
            "retrieved_distances": ",".join(
                f"{distance:.4f}" for distance in distances
            ),
             "hit_at_5": hit_at_5,
            "source_hit_at_5": source_hit_at_5,
            "first_relevant_rank": relevant_rank,
            "reciprocal_rank": reciprocal_rank,
            "matched_source": relevant_source_found or "",
            "matched_page": (
                relevant_page_found
                if relevant_page_found is not None
                else ""
            ),
        }

        results_rows.append(row)

        print("\n" + "-" * 60)
        print(f"Question {question_id}: {question}")
        print("Retrieved sources and pages:")

        for rank, (metadata, distance) in enumerate(
            zip(metadatas, distances), start=1
        ):
            print(
                f"  {rank}. {metadata.get('source', 'Unknown')} "
                f"| page {metadata.get('page', 'Unknown')} "
                f"| distance {distance:.4f}"
            )

        print(f"Hit@5: {hit_at_5}")
        print(f"First relevant rank: {relevant_rank}")

        if not hit_at_5:
            print("No passage matched the expected source and page.")

    # 5. Save detailed results
    if not results_rows:
        print("\nNo questions could be evaluated.")
        return

    df = pd.DataFrame(results_rows)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    # 6. Calculate summary metrics
    hit_at_5_score = df["hit_at_5"].mean()
    source_hit_at_5_score = df["source_hit_at_5"].mean()
    mrr_score = df["reciprocal_rank"].mean()
    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Questions evaluated: {len(df)}")
    print(f"Top K: {TOP_K}")
    print(f"Exact source-and-page Hit@5: {hit_at_5_score:.1%}")
    print(f"Source-only Hit@5: {source_hit_at_5_score:.1%}")
    print(f"MRR: {mrr_score:.4f}")
    print(f"Results saved to: {OUTPUT_FILE}")

    print("\nMetric meaning:")
    print("Hit@5 = fraction of questions with a matching passage in top 5.")
    print("MRR = average reciprocal rank of the first matching passage.")
    print(
        "These metrics measure source/page matching, not whether "
        "the passage fully supports the answer."
    )


if __name__ == "__main__":
    main()
