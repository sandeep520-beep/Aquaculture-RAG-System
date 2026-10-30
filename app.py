


import streamlit as st

from src.ask_rag import (
    load_resources,
    retrieve_passages,
    build_context,
    generate_answer,
    validate_citations,
    check_retrieval_quality,
    EMBEDDING_MODEL,
    LLM_MODEL,
)

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Aquaculture RAG Assistant",
    page_icon="🐟",
    layout="wide",
)

st.title("🐟 Aquaculture RAG Assistant")
st.write(
    "Ask questions about aquaculture and receive "
    "evidence-based answers from your uploaded documents."
)

st.caption(
    f"Embedding model: {EMBEDDING_MODEL} | "
    f"Language model: {LLM_MODEL}"
)

# --------------------------------------------------
# 2. LOAD THE EXISTING KNOWLEDGE BASE
# --------------------------------------------------

@st.cache_resource
def get_resources():
    return load_resources()


try:
    embedding_model, collection = get_resources()

except Exception as error:
    st.error("Could not load the aquaculture knowledge base.")
    st.code(str(error))
    st.info(
        "Check that ChromaDB is available, the database is indexed, "
        "and the required Python packages are installed."
    )
    st.stop()

st.success(
    f"Knowledge base ready: {collection.count()} document chunks loaded."
)

# --------------------------------------------------
# 3. STORE CHAT HISTORY
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------------------------------------------
# 4. DISPLAY PREVIOUS MESSAGES
# --------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            if message.get("retrieval_check"):
                st.caption(
                    "Retrieval check: "
                    + message["retrieval_check"]
                )

            if message.get("citation_check"):
                st.caption(
                    "Citation label check: "
                    + message["citation_check"]
                )

            sources = message.get("sources", [])

            if sources:
                with st.expander("View retrieved passages and sources"):
                    for source in sources:
                        st.markdown(
                            f"**[{source['id']}] "
                            f"{source['source']}**"
                        )
                        st.write(f"Page: {source['page']}")

                        if source["distance"] is not None:
                            st.write(
                                "Cosine distance: "
                                f"{source['distance']:.4f}"
                            )

                        st.write(source["text"])
                        st.divider()

# --------------------------------------------------
# 5. ACCEPT THE USER'S QUESTION
# --------------------------------------------------

question = st.chat_input(
    "Ask an aquaculture question..."
)

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    try:
        with st.spinner(
            "Searching documents and generating an answer..."
        ):
            # Retrieve relevant passages from ChromaDB
            sources = retrieve_passages(
                question,
                embedding_model,
                collection,
            )

            if not sources:
                answer = (
                    "I could not retrieve usable passages from the "
                    "aquaculture documents. Please try rephrasing "
                    "your question."
                )

                retrieval_check = "No usable passages retrieved."
                citation_check = "No answer citations to validate."

            else:
                # Combine retrieved passages into evidence context
                context = build_context(sources)

                if not context:
                    answer = (
                        "I could not prepare evidence from the "
                        "retrieved passages. Please try another question."
                    )

                    retrieval_check = (
                        "Retrieved passages could not be added to context."
                    )
                    citation_check = "No answer citations to validate."

                else:
                    # Generate the answer using the existing Ollama code
                    answer = generate_answer(
                        question,
                        context,
                    )

                    # Check retrieval quality and citation labels
                    retrieval_check = check_retrieval_quality(
                        sources
                    )

                    citation_result = validate_citations(
                        answer,
                        sources,
                    )

                    citation_check = (
                        f"{citation_result['status']}: "
                        f"{citation_result['message']}"
                    )

                    if citation_result["invalid_ids"]:
                        citation_check += (
                            " Unknown IDs: "
                            + ", ".join(
                                citation_result["invalid_ids"]
                            )
                        )

    except Exception as error:
        answer = (
            "I encountered an error while answering your question. "
            "Please check that Ollama is running and the required "
            "models and database are available."
        )

        retrieval_check = "Answer generation failed."
        citation_check = str(error)
        sources = []

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "retrieval_check": retrieval_check,
            "citation_check": citation_check,
        }
    )

    # Refresh to display the new user and assistant messages
    st.rerun()
