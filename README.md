# RAG Document Chat System

A Retrieval-Augmented Generation (RAG) system with LangGraph agentic workflow, OCR support, and Streamlit UI.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit App                           │
│  ┌─────────────┐  ┌──────────────────────────────────────┐ │
│  │   Sidebar   │  │           Chat Interface              │ │
│  │  - Upload   │  │  - Message History                    │ │
│  │  - Process  │  │  - Source Citations                   │ │
│  │  - Stats    │  │                                       │ │
│  └─────────────┘  └──────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  LangGraph Workflow                         │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │ Retrieve │ → │ Generate │ → │ Validate │ → │ Respond  │ │
│  └──────────┘   └──────────┘   └────┬─────┘   └──────────┘ │
│                       ▲             │                       │
│                       └─────────────┘                       │
│                        (Retry if invalid)                   │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────────┐
│   ChromaDB   │    │    Ollama    │    │   Document       │
│ Vector Store │    │  LLM + OCR   │    │   Ingestion      │
└──────────────┘    └──────────────┘    └──────────────────┘
```

## 📦 Technologies

| Component | Technology |
|-----------|------------|
| LLM/OCR | Ollama (deepseek-ocr, llama3.2) |
| Orchestration | LangGraph |
| Vector Store | ChromaDB |
| Embeddings | Ollama (nomic-embed-text) |
| UI | Streamlit |
| Document Processing | pypdf, pdf2image, Pillow |

## 🚀 Setup

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running
- Poppler (for PDF to image conversion)

### 2. Install Ollama Models

```bash
ollama pull deepseek-ocr
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env with your model names if different
```

### 5. Run the App

```bash
streamlit run app.py
```

## 📁 Project Structure

```
rag_chatbot/
├── app.py                    # Streamlit entry point
├── config.py                 # Configuration from env vars
├── requirements.txt          # Python dependencies
├── .env.example              # Environment template
├── rag/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py        # PDF/Image loaders
│   │   ├── ocr_service.py   # Ollama vision OCR
│   │   └── processor.py     # Semantic chunking
│   ├── retriever/
│   │   ├── __init__.py
│   │   └── vector_store.py  # ChromaDB wrapper
│   └── graph/
│       ├── __init__.py
│       ├── state.py         # LangGraph state
│       ├── nodes.py         # Agent nodes
│       └── workflow.py      # LangGraph workflow
└── chroma_db/               # Persistent vector storage
```

## 🔄 LangGraph Workflow

The RAG pipeline uses 4 agents orchestrated by LangGraph:

1. **Retriever Agent**: Queries ChromaDB for relevant document chunks
2. **Generator Agent**: Uses Ollama LLM to generate answers from context
3. **Validator Agent**: Checks answer relevance and detects hallucinations
4. **Final Response Agent**: Formats response with source citations

### Retry Logic

If validation fails, the workflow retries generation (up to 3 times by default).

## 💬 Usage

1. **Upload Documents**: Use sidebar to upload PDF or image files
2. **Process**: Click "Process Documents" to OCR and index
3. **Chat**: Ask questions in the chat interface
4. **View Sources**: Expand sources to see where answers came from

## � Sample Data

The `sample_data/` folder contains:
- `sample_document.md` - Sample document about AI for testing
- `sample_chat_transcript.md` - Example chat interaction demonstrating the system

## �📝 Sample Interaction

**User**: What is the main topic of this document?

**Assistant**: Based on the uploaded document, the main topic is...

**Sources**:
- document.pdf (page 1)
- document.pdf (page 3)
