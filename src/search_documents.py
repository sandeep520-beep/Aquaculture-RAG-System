
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "chroma_db"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "aquaculture_documents"


def main():
    client = chromadb.PersistentClient(path=str(DB_DIR))

    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        print("Index not found. Run build_index.py first.")
        return

    if collection.count() == 0:
        print("The index is empty.")
        return

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    question = input("\nAsk an aquaculture question: ").strip()

    if not question:
        print("Please enter a question.")
        return

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True
    )

    results = collection.query(
        query_embeddings=question_embedding.tolist(),
        n_results=min(3, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    print("\nMOST RELEVANT PASSAGES")
    print("=" * 50)

    for i, (text, metadata, distance) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ),
        start=1
    ):
        print(f"\nResult {i}")
        print(f"Source: {metadata['source']}")
        print(f"Page: {metadata['page']}")
        print(f"Cosine distance: {distance:.4f}")
        print(f"Text: {text[:700]}")


if __name__ == "__main__":
    main()
