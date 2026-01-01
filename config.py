"""
Configuration settings for the RAG pipeline.
Uses environment variables with sensible defaults.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ===================================================
# OLLAMA MODELS
# ===================================================
OCR_MODEL = os.getenv("OCR_MODEL")
CHAT_MODEL = os.getenv("CHAT_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# ===================================================
# MODEL PARAMETERS
# ===================================================
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "8192"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))

# ===================================================
# CHUNKING SETTINGS
# ===================================================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# ===================================================
# VECTOR STORE
# ===================================================
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "rag_documents")

# ===================================================
# RAG SETTINGS
# ===================================================
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

# ===================================================
# API KEYS (for non-Ollama models)
# ===================================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ===================================================
# POPPLER PATH (Windows only, for PDF to image)
# ===================================================
POPPLER_PATH = os.getenv("POPPLER_PATH")  # e.g., C:\poppler\Library\bin
