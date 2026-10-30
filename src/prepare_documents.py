
from pathlib import Path
import csv
import fitz

# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "data" / "pdfs"
OUTPUT_FILE = BASE_DIR / "data" / "chunks.csv"

# Chunk settings (measured in characters)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def split_text(text):
    """Split long text into smaller overlapping chunks."""
    text = " ".join(text.split())
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def process_pdfs():
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found!")
        print(f"Please add a PDF to: {PDF_DIR}")
        return

    all_chunks = []

    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path.name}")

        with fitz.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text")

                if not text.strip():
                    print(
                        f"Page {page_number}: no selectable text"
                    )
                    continue

                chunks = split_text(text)

                for chunk_number, chunk in enumerate(chunks, start=1):
                    all_chunks.append({
                        "chunk_id": (
                            f"{pdf_path.stem}_"
                            f"p{page_number}_c{chunk_number}"
                        ),
                        "source": pdf_path.name,
                        "page": page_number,
                        "chunk_number": chunk_number,
                        "text": chunk
                    })

        print("Finished:", pdf_path.name)

    if not all_chunks:
        print("\nNo text extracted from the PDFs.")
        print("The documents may require OCR.")
        return

    # Save the extracted chunks and source information
    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "chunk_id",
                "source",
                "page",
                "chunk_number",
                "text"
            ]
        )

        writer.writeheader()
        writer.writerows(all_chunks)

    print("\nPDF processing completed!")
    print("PDF files:", len(pdf_files))
    print("Total chunks:", len(all_chunks))
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    process_pdfs()
