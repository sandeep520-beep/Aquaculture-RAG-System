# Aquaculture RAG System 🐟

### Retrieval-Augmented Generation for Aquaculture Knowledge

An AI-powered question-answering system that retrieves relevant information from aquaculture technical documents and generates context-based answers using a Large Language Model (LLM).

The project combines semantic search, keyword-based retrieval, vector databases, and Generative AI to make aquaculture information easier to access.

## 🚀 Project Overview

The Aquaculture RAG System allows users to ask questions about aquaculture topics and receive answers grounded in technical PDF documents.

Instead of relying only on the LLM's pre-trained knowledge, the system retrieves relevant passages from a document knowledge base and uses them as context to generate answers with source citations.

## 🏗️ System Architecture

```text
Aquaculture PDF Documents
          |
          v
Document Processing
          |
          v
Text Extraction and Chunking
          |
          v
Sentence Transformers
          |
          v
       ChromaDB
     (Vector Database)
          |
          v
      User Question
          |
          v
Hybrid Retrieval
(Semantic + Keyword Search)
          |
          v
   Relevant Passages
          |
          v
    Llama 3.2 (3B)
          |
          v
  Context-Based Answer
          |
          v
   Source Citations
```

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Sentence Transformers | Text embeddings |
| ChromaDB | Vector database |
| Ollama | Local LLM execution |
| Llama 3.2 (3B) | Answer generation |
| Pandas | Data processing |
| Streamlit | Application interface, if using the provided app |

## ✨ Key Features

- PDF-based aquaculture knowledge retrieval.
- Text extraction and document chunking.
- Semantic search using Sentence Transformers.
- Keyword-based retrieval with synonym matching.
- Hybrid retrieval combining semantic and keyword search.
- Context-based answer generation using Llama 3.2.
- Source-based citations in generated answers.
- Retrieval and answer-quality evaluation.

## 📊 Evaluation Results

The system was evaluated using a baseline dataset of 40 questions.

| Evaluation Metric | Baseline Result |
|---|---:|
| Source-level Hit@5 | 100% |
| Exact source-and-page Hit@5 | 90% |
| Mean Reciprocal Rank (MRR) | 0.7625 |
| Average answer evaluation score | 4.65/5 |
| Answers passing evaluation | 33/40 |
| Answers requiring review | 7/40 |

*Note: These are baseline evaluation results. Citation-label validity does not necessarily guarantee factual support, and answer evaluation requires further improvement.*

## 📁 Project Structure

```text
Aquaculture-RAG-System/
│
├── data/
│   ├── chunks.csv
│   ├── evaluation_questions.json
│   ├── evaluation_questions_backup.json
│   └── unanswerable_candidates.json
│
├── experiments/
│   ├── answer_evaluation.csv
│   └── retrieval_results.csv
│
├── src/
│   ├── ask_rag.py
│   ├── build_index.py
│   ├── prepare_documents.py
│   ├── evaluation.py
│   ├── answer_evaluation.py
│   ├── search_documents.py
│   └── inspect_chunks.py
│
├── app.py
├── add_evaluation_questions.py
├── inspect_chunks.py
└── .gitignore
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/sandeep520-beep/Aquaculture-RAG-System.git
cd Aquaculture-RAG-System
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and start Ollama

Download Ollama from:

https://ollama.com/

Pull the required model:

```bash
ollama pull llama3.2:3b
```

### 5. Run the RAG assistant

```bash
python src/ask_rag.py
```

**Note:** The local ChromaDB index and original PDF documents are not included in this GitHub repository. Prepare your PDF documents and build the index using the provided document-processing and indexing scripts before running the system.

## 🔬 Future Improvements

- Improve answer accuracy and reduce unsupported statements.
- Optimize retrieval ranking and context selection.
- Expand the aquaculture document knowledge base.
- Improve citation verification.
- Enhance the user interface.
- Evaluate performance across a larger question dataset.

## 👨‍💻 Author

**Sandeep Upadhyay**

M.Tech Student – Aquacultural Engineering  
Indian Institute of Technology (IIT) Kharagpur

**GitHub:** https://github.com/sandeep520-beep

---

⭐ If you find this project interesting, feel free to explore the repository.
