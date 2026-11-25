# 📚 Document Intelligence RAG Assistant v2.0

A production-grade Retrieval-Augmented Generation (RAG) system for intelligent question answering over **PDF**, **DOCX**, and **HTML** documents using **LangChain**, **ChromaDB**, **Hugging Face Embeddings**, and **Groq LLMs**.

The system combines semantic retrieval, lexical retrieval, Reciprocal Rank Fusion (RRF), optional CrossEncoder reranking, chapter-aware search, document versioning, and source-grounded answer generation.

---

# 🚀 Features

## Multi-Format Document Ingestion

Supported formats:

- PDF
- DOCX
- HTML

---

## Advanced Document Extraction

### PDF Processing

- Page-by-page extraction using PyMuPDF
- Page metadata preservation
- Page-level source tracking
- Automatic empty-page filtering

### DOCX Processing

- Paragraph extraction
- Table extraction
- Table header preservation
- Original document order preservation
- Structured table serialization

### HTML Processing

- BeautifulSoup HTML parsing
- Removal of:
  - Scripts
  - Styles
  - Navigation menus
  - Headers
  - Footers
  - SVG graphics
  - Iframes

---

## Intelligent Metadata System

Every document is enriched with metadata including:

- Source filename
- Document fingerprint
- SHA-256 document ID
- Chapter information
- Page information
- Chunk identifier
- Format type

Example:

```json
{
  "source": "Agricultural_Science_Notes.pdf",
  "document_id": "8c4ef5f8b70...",
  "chapter": "CHAPTER 5 - IRRIGATION",
  "page": 42,
  "chunk_id": "8c4ef5f8b70_124"
}
```

---

# 🔒 Duplicate Detection & Version Control

The assistant generates a SHA-256 fingerprint for every indexed document.

Benefits:

- Duplicate prevention
- Change detection
- Automatic updates
- Persistent indexing
- Safe document replacement

If a file is already indexed:

```text
Document already indexed.
Skipping duplicate processing.
```

If the file has changed:

```text
Old document version removed.
New version indexed.
```

---

# 🧠 Hybrid Retrieval Architecture

Unlike traditional vector-only RAG pipelines, this system uses multiple retrieval strategies.

## Stage 1: Semantic Retrieval

Uses:

```text
HuggingFace Embeddings
+
ChromaDB
+
Max Marginal Relevance (MMR)
```

Benefits:

- Understands semantic meaning
- Handles synonyms
- Finds conceptually related passages
- Reduces duplicate retrievals

---

## Stage 2: Lexical Retrieval

Custom BM25-inspired retrieval implemented in Python.

Features:

- Inverse Document Frequency (IDF)
- Term Frequency weighting
- Query keyword matching
- No additional search engine dependency

Benefits:

- Strong keyword search
- Excellent for names and definitions
- Works well for chapter references

---

## Stage 3: Reciprocal Rank Fusion (RRF)

Combines rankings from:

- Semantic retrieval
- Lexical retrieval

Formula:

```text
RRF Score = Σ 1 / (k + rank)
```

Benefits:

- Better recall
- Better ranking stability
- Improved answer quality
- Reduced retrieval failures

---

## Stage 4: Optional CrossEncoder Reranking

Supported reranker:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

Workflow:

1. Retrieve candidate chunks
2. Evaluate query-document pairs
3. Re-rank results
4. Return highest relevance chunks

Benefits:

- Improved precision
- Better retrieval quality
- Stronger grounding

---

# 📖 Chapter-Aware Retrieval

The assistant automatically detects chapter headings:

```text
CHAPTER 1
CHAPTER IV
CHAPTER 5
CHAPTER TWELVE
```

Users can ask questions like:

```text
Summarize Chapter 3

Explain Chapter 7

Overview of Chapter 10
```

The system automatically searches within chunks belonging to that chapter.

---

# ✂️ Intelligent Chunking

Chunking is performed using:

```python
RecursiveCharacterTextSplitter
```

Default settings:

```env
CHUNK_SIZE=1200
CHUNK_OVERLAP=200
```

Chunk preservation:

- Source metadata
- Page metadata
- Chapter metadata
- Document relationships

Each chunk receives:

```text
chunk_index
chunk_id
```

allowing traceability and source attribution.

---

# 🗄 Persistent Vector Database

The project uses ChromaDB for vector storage.

Features:

- Persistent local storage
- Fast retrieval
- Reusable document indexes
- Automatic loading
- Multi-session support

Default location:

```text
./chroma_db
```

---

# 🤖 Language Model

Default LLM provider:

```text
Groq
```

Default model:

```text
qwen/qwen3-8-27b
```

Configurable via environment variables.

Benefits:

- High-speed inference
- Large context handling
- Strong reasoning capability
- Low latency responses

---

# 🎯 Grounded Answer Generation

The assistant follows strict grounding rules.

It will:

✅ Answer only from retrieved context

✅ Preserve technical terminology

✅ Reference source information

✅ Combine evidence from multiple chunks

✅ Identify missing information

It will NOT:

❌ Invent facts

❌ Use external knowledge

❌ Hallucinate names or numbers

❌ Assume missing information

If information is unavailable:

```text
I could not find that information in the document.
```

---

# 📑 Automatic Source Citations

Every answer includes source references.

Example:

```text
SOURCES

- Source: Agricultural_Science_Notes.pdf
  Chapter: CHAPTER 5 - IRRIGATION
  Page: 42

- Source: Agricultural_Science_Notes.pdf
  Chapter: CHAPTER 6 - DRAINAGE
  Page: 57
```

This makes every answer traceable and verifiable.

---

# 🏗 Architecture Overview

```text
Document
    │
    ▼
Document Validation
    │
    ▼
Document Loader
(PDF / DOCX / HTML)
    │
    ▼
Metadata Enrichment
    │
    ▼
Chapter Detection
    │
    ▼
Chunking
    │
    ▼
Embeddings
    │
    ▼
ChromaDB Indexing
    │
    ▼
Hybrid Retrieval
 ┌───────────────┐
 │ Semantic MMR  │
 │ Lexical Search│
 └──────┬────────┘
        ▼
Reciprocal Rank Fusion
        ▼
Optional CrossEncoder
        ▼
Groq LLM
        ▼
Grounded Answer
        ▼
Source Citations
```

---

# 📂 Project Structure

```text
project/
│
├── document_intelligence_rag.py
├── .env
├── chroma_db/
│
└── README.md
```

---

# ⚙️ Installation

## 1. Clone Repository

```bash
git clone https://github.com/yourusername/document-intelligence-rag.git

cd document-intelligence-rag
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -U \
langchain \
langchain-core \
langchain-text-splitters \
langchain-huggingface \
langchain-chroma \
langchain-groq \
chromadb \
sentence-transformers \
pymupdf \
python-docx \
beautifulsoup4 \
python-dotenv
```

---

# 🔐 Environment Configuration

Create a `.env` file:

```env
# GROQ

GROQ_API_KEY=your_groq_api_key

GROQ_MODEL=qwen/qwen3-8-27b

GROQ_TEMPERATURE=0

GROQ_MAX_TOKENS=1500

# EMBEDDINGS

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# CHROMA

CHROMA_DIRECTORY=./chroma_db

CHROMA_COLLECTION=document_knowledge

# CHUNKING

CHUNK_SIZE=1200

CHUNK_OVERLAP=200

# RETRIEVAL

RETRIEVAL_K=5

FETCH_K=15

LEXICAL_K=5

# CONTEXT

MAX_CONTEXT_CHARS=18000

# OPTIONAL RERANKER

ENABLE_RERANKER=False

RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

---

# ▶️ Running the Assistant

## Option 1: Load via Command Line

```bash
python document_intelligence_rag.py document.pdf
```

Example:

```bash
python document_intelligence_rag.py Agricultural_Science_Notes.pdf
```

---

## Option 2: File Picker

```bash
python document_intelligence_rag.py
```

A file selection dialog will open.

---

# 💬 Example Questions

```text
What are the types of pig breeds?

Explain irrigation systems.

Summarize Chapter 5.

What are the functions of farm drainage?

Describe the disease control measures in pigs.

Compare surface irrigation and sprinkler irrigation.

What are the advantages of livestock production?
```

---

# 🛠 Available Commands

View help:

```text
/help
```

Display conversation history:

```text
/history
```

Display database statistics:

```text
/stats
```

Exit application:

```text
/exit
```

---

# 📊 Database Statistics

Built-in analytics:

```text
DATABASE STATISTICS

============================================================

Total indexed chunks: 842

Unique documents: 4

Detected chapters: 28
```

You can also see:

- Indexed filenames
- Number of chunks
- Number of chapters

---

# ✅ Best Use Cases

Suitable for:

- Textbooks
- Academic notes
- Research papers
- Manuals
- Reports
- Internal documentation
- Agricultural science materials
- Study materials

---

# ⚠️ Limitations

Currently unsupported:

- Scanned PDFs without OCR
- Image-only documents
- Handwritten notes

For scanned PDFs, OCR should be performed before indexing.

---

# 🔮 Future Improvements

Planned enhancements:

- OCR support (Tesseract / PaddleOCR)
- Native Markdown support
- FastAPI API deployment
- Streamlit Web UI
- Multi-document conversations
- Metadata filtering
- Parent-Child Retrieval
- Contextual Compression Retrieval
- Hybrid Sparse + Dense Retrieval
- Knowledge Graph Integration
- Agent Workflows

---

# 🧰 Technology Stack

| Component | Technology |
|------------|------------|
| Framework | LangChain 1.x |
| Language | Python 3.12+ |
| Embeddings | all-MiniLM-L6-v2 |
| Vector Store | ChromaDB |
| LLM Provider | Groq |
| LLM Model | Qwen 3 27B |
| Retrieval | MMR + Lexical Search + RRF |
| Reranking | CrossEncoder |
| PDF Processing | PyMuPDF |
| DOCX Processing | python-docx |
| HTML Extraction | BeautifulSoup |
| Configuration | python-dotenv |

---

# 📜 License

This project is intended for:

- Educational use
- Research use
- Personal document intelligence systems
- Knowledge management workflows

Feel free to modify and extend it for your own projects.

---

# 👨‍💻 About This Project

Most beginner RAG projects rely entirely on vector similarity search.

This project improves retrieval quality through:

- Semantic retrieval
- Lexical retrieval
- Reciprocal Rank Fusion (RRF)
- Optional CrossEncoder reranking
- Chapter-aware retrieval
- Metadata-aware chunking
- Source-grounded responses
- Persistent document versioning

The result is a robust document intelligence assistant capable of answering questions across large PDF, DOCX, and HTML document collections while maintaining traceability, transparency, and retrieval accuracy.