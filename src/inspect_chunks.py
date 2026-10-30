
import chromadb

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("aquaculture_documents")

results = collection.get(
    include=["documents", "metadatas"]
)

keywords = [
    "feed costs",
    "protein",
    "feeding rate",
    "dissolved oxygen",
    "water quality",
    "temperature",
    "pH",
    "Nile tilapia",
    "stocking density",
    "growth"
]

for keyword in keywords:
    print("\n" + "=" * 70)
    print("KEYWORD:", keyword)
    found = 0

    for document, metadata in zip(
        results["documents"],
        results["metadatas"]
    ):
        if keyword.lower() in document.lower():
            print(
                "\nPDF:", metadata.get("source"),
                "| Page:", metadata.get("page")
            )
            print(document[:800])
            found += 1

            if found == 2:
                break

    if found == 0:
        print("No matching passage found.")
