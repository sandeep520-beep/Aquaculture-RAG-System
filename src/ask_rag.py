import re
from pathlib import Path
import chromadb
import ollama
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = str(PROJECT_ROOT / "chroma_db")
COLLECTION_NAME = "aquaculture_documents"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:3b"

TOP_K = 7
MAX_CONTEXT_CHARS = 18000

# This is a starting threshold, not a universal relevance rule.
# Chroma cosine distance depends on the collection's distance metric.
MAX_DISTANCE_WARNING = 0.50


# ============================================================
# 2. LOAD THE KNOWLEDGE BASE
# ============================================================

def load_resources():
    print("Loading embedding model...")

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    print("Connecting to ChromaDB...")

    client = chromadb.PersistentClient(path=DB_PATH)

    try:
        collection = client.get_collection(
            name=COLLECTION_NAME
        )
    except Exception as error:
        raise RuntimeError(
            f"Could not open collection '{COLLECTION_NAME}'. "
            "Check that the database exists and is indexed."
        ) from error

    total_chunks = collection.count()

    if total_chunks == 0:
        raise RuntimeError("The knowledge base contains no chunks.")

    print(f"Knowledge base loaded: {total_chunks} chunks")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Language model: {LLM_MODEL}")
    print("Aquaculture RAG Assistant is ready!")

    return embedding_model, collection


# ============================================================
# 3. RETRIEVE RELEVANT PASSAGES
# ============================================================



def retrieve_passages(question, embedding_model, collection):

    # --------------------------------------------------
    # 1. SEMANTIC SEARCH - SEARCH MORE CANDIDATES
    # --------------------------------------------------

    question_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    number_of_results = min(30, collection.count())

    semantic_results = collection.query(
        query_embeddings=[question_embedding],
        n_results=number_of_results,
        include=["documents", "metadatas", "distances"]
    )

    semantic_ids = (semantic_results.get("ids") or [[]])[0]
    semantic_docs = (semantic_results.get("documents") or [[]])[0]
    semantic_metadata = (semantic_results.get("metadatas") or [[]])[0]
    semantic_distances = (semantic_results.get("distances") or [[]])[0]

    # --------------------------------------------------
    # 2. KEYWORD SEARCH WITH AQUACULTURE SYNONYMS
    # --------------------------------------------------

    all_data = collection.get(
        include=["documents", "metadatas"]
    )

    query_words = set(
        re.findall(r"\b[a-z0-9]+(?:\.[0-9]+)?\b", question.lower())
    )

    stop_words = {
        "what", "is", "the", "a", "an", "and", "or",
        "of", "to", "in", "on", "for", "how", "does",
        "can", "are", "with", "between", "at"
    }

    query_words -= stop_words

    # Expand common aquaculture terms.
    if "oxygen" in query_words or "dissolved" in query_words:
        query_words.add("do")

    if "feeding" in query_words or "feed" in query_words:
        query_words.update({"eating", "eat", "intake"})

    if "growth" in query_words:
        query_words.update({"growing", "grow", "rate"})

    keyword_matches = []

    for chunk_id, document, metadata in zip(
        all_data["ids"],
        all_data["documents"],
        all_data["metadatas"]
    ):
        if not document:
            continue

        document_words = set(
            re.findall(
                r"\b[a-z0-9]+(?:\.[0-9]+)?\b",
                document.lower()
            )
        )

        matched_words = query_words.intersection(document_words)

        if matched_words:
            score = len(matched_words) / max(len(query_words), 1)

            # Give additional importance to direct growth evidence.
            document_lower = document.lower()

            if (
                ("growth" in query_words or "grow" in query_words)
                and ("do" in query_words)
                and (
                    "best growth rate" in document_lower
                    or "above 4 mg/l" in document_lower
                )
            ):
                score += 0.5

            keyword_matches.append({
                "id": chunk_id,
                "document": document,
                "metadata": metadata or {},
                "keyword_score": score
            })

    keyword_matches.sort(
        key=lambda item: item["keyword_score"],
        reverse=True
    )

    # --------------------------------------------------
    # 3. COMBINE SEMANTIC AND KEYWORD RESULTS
    # --------------------------------------------------

    combined = {}

    for rank, (chunk_id, document, metadata, distance) in enumerate(
        zip(
            semantic_ids,
            semantic_docs,
            semantic_metadata,
            semantic_distances
        ),
        start=1
    ):
        combined[chunk_id] = {
            "document": document,
            "metadata": metadata or {},
            "distance": distance,
            "score": 1 / (60 + rank)
        }

    for rank, item in enumerate(keyword_matches, start=1):
        chunk_id = item["id"]

        if chunk_id in combined:
            combined[chunk_id]["score"] += 1 / (60 + rank)
        else:
            combined[chunk_id] = {
                "document": item["document"],
                "metadata": item["metadata"],
                "distance": None,
                "score": 1 / (60 + rank)
            }

        # Add the keyword relevance score.
        combined[chunk_id]["score"] += (
            item["keyword_score"] * 0.01
        )

    ranked_results = sorted(
        combined.items(),
        key=lambda item: item[1]["score"],
        reverse=True
    )

    # --------------------------------------------------
    # 4. PREPARE FINAL PASSAGES
    # --------------------------------------------------

    sources = []

    for chunk_id, item in ranked_results[:TOP_K]:

        if not item["document"] or not item["document"].strip():
            continue

        sources.append({
            "source": item["metadata"].get(
                "source", "Unknown source"
            ),
            "page": item["metadata"].get(
                "page", "Unknown page"
            ),
            "distance": item["distance"],
            "text": item["document"]
        })

    # Assign citation IDs.
    for rank, source in enumerate(sources, start=1):
        source["id"] = f"S{rank}"

    return sources

# ============================================================
# 4. BUILD EVIDENCE CONTEXT
# ============================================================

def build_context(sources):
    context_parts = []
    used_characters = 0

    for source in sources:
        passage = (
            f"[{source['id']}]\n"
            f"Document: {source['source']}\n"
            f"Page: {source['page']}\n"
            f"Passage:\n{source['text']}"
        )

        remaining = MAX_CONTEXT_CHARS - used_characters

        if remaining <= 0:
            break

        if len(passage) > remaining:
            break

        context_parts.append(passage)
        used_characters += len(passage)

    return "\n\n".join(context_parts)


# ============================================================
# 5. GENERATE AN EVIDENCE-BASED ANSWER
# ============================================================


def generate_answer(question, context):
    
    system_prompt = """

You are an aquaculture research assistant.

Answer the user's question using ONLY the supplied reference passages.

CORE RULES:

1. Understand what the question is asking.
2. Start with a direct answer in the first sentence.
3. Answer all parts of the question.
4. Use simple, clear and scientifically accurate language.
5. Include only information that directly answers the question.
6. Never use outside knowledge or invent facts.
7. If information is missing, state exactly what is missing.
8. Combine information from different passages when they explain
   different parts of the same relationship.
9. Explain cause-and-effect relationships only when the evidence
   supports each step.
10. Do not repeat the same information in different sentences.
11. Do not say that a relationship is unknown if its individual
    steps are explained in the reference passages.

SCIENTIFIC VALUES:

12. Preserve numerical values, units and ranges exactly as given.
13. Explain what each numerical value represents.
14. Clearly distinguish short-term survival tolerance from
    recommended or optimal culture conditions.
15. If passages give different values, report the difference
    without silently choosing one.

SPECIAL INSTRUCTIONS FOR DO, FEEDING AND GROWTH:

When the question asks about dissolved oxygen, feeding activity
and Nile tilapia growth, organize the answer in this order:

1. Dissolved oxygen and feeding:
   Explain how low DO affects fish feeding, using the exact
   thresholds provided in the passages.

2. Feeding and growth:
   Explain how feed intake is related to growth, using the
   supplied evidence.

3. Dissolved oxygen and feed utilization:
   Explain how unsuitable DO affects the ability of fish to
   use feed efficiently and achieve maximum growth.

4. Recommended DO:
   State the DO level associated with the best growth rate.
   Distinguish this from the short-term survival tolerance.

Do not claim that high DO directly increases feeding unless
the reference passages explicitly establish this.

CITATIONS:

16. Cite every factual paragraph using square-bracket labels exactly
    like [S1] or [S2]. Never write (S1), S1, or another format.
17. Use only source IDs provided in the reference passages.
18. Put citations immediately after the specific supported claim.
19. Ensure every citation supports the statement it follows.
20. A single sentence may use multiple citations if necessary.
21. Do not make a factual claim unless at least one passage supports it.
22. If the passages do not support a requested detail, say it is not
    stated instead of guessing.

ANSWER STRUCTURE:

- Give a direct introductory sentence.
- Use short paragraphs or numbered points when appropriate.
- Explain the relationship logically.
- End with a brief conclusion.
- Avoid unnecessary repetition and unrelated details.

FINAL CHECK:

Before responding, verify that:
- Every part of the question is answered.
- Every factual claim is supported by the passages.
- All numerical values and units are accurate.
- No unsupported relationship has been introduced.
- No important evidence has been ignored.

Return only the final answer.
"""

    user_prompt = f"""
REFERENCE PASSAGES:

{context}

USER QUESTION:
{question}

Write a focused, evidence-based answer with citations in the
required [S1] format. If the evidence is insufficient, explain
the limitation clearly. Double-check all numerical ranges,
thresholds, units, and their associated effects.
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        options={
            "temperature": 0,
            "num_predict": 400
        }
    )

    # Extract the text returned by Ollama.
    answer = response["message"]["content"]

    if not isinstance(answer, str):
        raise TypeError(
            "Ollama returned a non-text answer. "
            f"Received type: {type(answer).__name__}"
        )

    answer = answer.strip()

    if not answer:
        raise RuntimeError("The language model returned an empty answer.")

    # Small local models sometimes output (S1) or (S1, page 4)
    # despite being instructed to use [S1]. Normalize citation
    # formatting without inventing citations or changing source IDs.
    answer = re.sub(
        r"\\((S\\d+)(?:\\s*,[^)]*)?\\)",
        r"[\\1]",
        answer,
        flags=re.IGNORECASE,
    )

    return answer



# ============================================================
# 6. BASIC CITATION VALIDATION
# ============================================================

def validate_citations(answer, sources):
    """
    Checks whether citation IDs in the answer exist in the
    retrieved evidence. This does NOT prove semantic support.
    """

    cited_ids = set(
        re.findall(r"\[(S\d+)\]", answer)
    )

    available_ids = {
        source["id"] for source in sources
    }

    invalid_ids = cited_ids - available_ids

    if not cited_ids:
        return {
            "status": "NO_CITATIONS",
            "message": (
                "No source citations were found. "
                "Check whether the answer needs supporting evidence."
            ),
            "invalid_ids": []
        }

    if invalid_ids:
        return {
            "status": "INVALID_CITATIONS",
            "message": (
                "The answer contains citation labels that were "
                "not included in the retrieved evidence."
            ),
            "invalid_ids": sorted(invalid_ids)
        }

    return {
        "status": "VALID_LABELS",
        "message": (
            "All citation labels match retrieved source IDs. "
            "Their factual support still needs to be checked."
        ),
        "invalid_ids": []
    }


# ============================================================
# 7. CHECK RETRIEVAL QUALITY
# ============================================================

def check_retrieval_quality(sources):
    if not sources:
        return "No passages were retrieved."

    distances = [
        source["distance"]
        for source in sources
        if source["distance"] is not None
    ]

    if not distances:
        return (
            "Retrieval distances are unavailable. "
            "Inspect the passages manually."
        )

    best_distance = min(distances)

    if best_distance > MAX_DISTANCE_WARNING:
        return (
            "Warning: even the closest retrieved passage has a "
            "relatively high distance. The evidence may not answer "
            "the question. Review the passages before trusting "
            "the generated answer."
        )

    return (
        "At least one passage is relatively close according to "
        "the configured distance metric. This does not guarantee "
        "that the passages answer the question."
    )


# ============================================================
# 8. DISPLAY RETRIEVED EVIDENCE
# ============================================================

def display_evidence(sources):
    print("\n" + "=" * 70)
    print("RETRIEVED EVIDENCE")
    print("=" * 70)

    if not sources:
        print("No relevant passages were retrieved.")
        return

    for source in sources:
        print("\n" + "-" * 70)

        print(
            f"[{source['id']}] {source['source']} "
            f"| Page {source['page']}"
        )

        distance = source["distance"]

        if distance is not None:
            print(f"Cosine distance: {distance:.4f}")

        print("\nPassage:")
        print(source["text"])

    print("\n" + "=" * 70)
    print(
        "Reminder: valid citation labels do not automatically "
        "mean the cited passage supports the claim."
    )
    print("=" * 70)


# ============================================================
# 9. ASK A QUESTION
# ============================================================

def ask_question(question, embedding_model, collection):
    print("\nSearching the aquaculture knowledge base...")

    sources = retrieve_passages(
        question,
        embedding_model,
        collection
    )

    if not sources:
        print("\nNo usable passages were retrieved.")
        print(
            "Try rephrasing the question or check the indexed PDFs."
        )
        return

    context = build_context(sources)

    if not context:
        print("\nNo evidence could be added to the prompt.")
        return

    # Check retrieval before generating an answer.
    retrieval_message = check_retrieval_quality(sources)

    print("Generating an answer with the local language model...")

    answer = generate_answer(question, context)

    citation_result = validate_citations(answer, sources)

    print("\n" + "=" * 70)
    print("QUESTION")
    print("=" * 70)
    print(question)

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(answer)

    print("\n" + "=" * 70)
    print("RETRIEVAL CHECK")
    print("=" * 70)
    print(retrieval_message)

    print("\n" + "=" * 70)
    print("CITATION LABEL CHECK")
    print("=" * 70)
    print(citation_result["status"])
    print(citation_result["message"])

    if citation_result["invalid_ids"]:
        print(
            "Unknown source IDs:",
            ", ".join(citation_result["invalid_ids"])
        )

    display_evidence(sources)


# ============================================================
# 10. MAIN PROGRAM
# ============================================================

def main():
    try:
        embedding_model, collection = load_resources()

    except Exception as error:
        print("\nSTARTUP ERROR")
        print(error)
        print(
            "\nCheck your database path, Python packages, "
            "and embedding model."
        )
        return

    print("\nAsk an aquaculture question.")
    print("Type 'exit' to close the assistant.")

    while True:
        try:
            question = input("\nYour question: ").strip()

            if question.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            if not question:
                print("Please enter a question.")
                continue

            ask_question(
                question,
                embedding_model,
                collection
            )

        except KeyboardInterrupt:
            print("\nAssistant stopped.")
            break

       
        except Exception as error:
            import traceback
            traceback.print_exc()
            print(f"\nQUESTION ERROR: {error}")


if __name__ == "__main__":
    main()
