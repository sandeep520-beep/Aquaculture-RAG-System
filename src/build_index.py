
from pathlib import Path

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_FILE = BASE_DIR / "data" / "chunks.csv"
DB_DIR = BASE_DIR / "chroma_db"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "aquaculture_documents"


def main():
    if not CSV_FILE.exists():
        print("chunks.csv not found. Process PDFs first.")
        return

    df = pd.read_csv(CSV_FILE).fillna("")

    if df.empty:
        print("No document chunks found.")
        return

    print(f"Loaded {len(df)} chunks.")
    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    texts = df["text"].astype(str).tolist()

    print("Creating embeddings...")

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    client = chromadb.PersistentClient(path=str(DB_DIR))

    # Remove the old index so the database matches the current CSV.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    batch_size = 100

    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        batch = df.iloc[start:end]

        metadata = []

        for _, row in batch.iterrows():
            metadata.append({
                "source": str(row["source"]),
                "page": int(row["page"]),
                "chunk_number": int(row["chunk_number"])
            })

        collection.add(
            ids=batch["chunk_id"].astype(str).tolist(),
            documents=batch["text"].astype(str).tolist(),
            metadatas=metadata,
            embeddings=embeddings[start:end].tolist()
        )

    print("\nIndex built successfully!")
    print("Chunks indexed:", collection.count())
    print("Database location:", DB_DIR)


if __name__ == "__main__":
    main()
