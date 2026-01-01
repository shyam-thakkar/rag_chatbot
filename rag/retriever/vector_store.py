"""
Vector store using FAISS for document embeddings.
Supports Ollama embeddings and sentence-transformers fallback.
"""
import os
import pickle
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

# Try Ollama embeddings first, fall back to sentence-transformers
try:
    from langchain_ollama import OllamaEmbeddings
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL, RETRIEVAL_K

# Rename for FAISS context
FAISS_PERSIST_DIR = CHROMA_PERSIST_DIR.replace("chroma", "faiss") if "chroma" in CHROMA_PERSIST_DIR else CHROMA_PERSIST_DIR


class VectorStore:
    """
    FAISS vector store wrapper for document embeddings.
    """
    
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        index_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        use_ollama: bool = True
    ):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory to persist FAISS index
            index_name: Name of the FAISS index
            embedding_model: Model name for embeddings
            use_ollama: Whether to use Ollama embeddings (False = HuggingFace)
        """
        self.persist_directory = persist_directory or FAISS_PERSIST_DIR
        self.index_name = index_name or CHROMA_COLLECTION_NAME
        self.embedding_model = embedding_model or EMBEDDING_MODEL
        
        # Initialize embeddings
        self.embeddings = self._get_embeddings(use_ollama)
        
        # Initialize or load FAISS
        self.vectorstore = self._init_vectorstore()
    
    def _get_embeddings(self, use_ollama: bool):
        """Get embedding function based on availability and preference."""
        if use_ollama and OLLAMA_AVAILABLE:
            return OllamaEmbeddings(model=self.embedding_model)
        elif HF_AVAILABLE:
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        else:
            raise RuntimeError(
                "No embedding backend available. "
                "Install langchain-ollama or sentence-transformers."
            )
    
    def _get_index_path(self) -> str:
        """Get the full path for the FAISS index."""
        return os.path.join(self.persist_directory, self.index_name)
    
    def _init_vectorstore(self) -> Optional[FAISS]:
        """Initialize or load existing FAISS index."""
        os.makedirs(self.persist_directory, exist_ok=True)
        
        index_path = self._get_index_path()
        
        # Try to load existing index
        if os.path.exists(f"{index_path}.faiss"):
            try:
                return FAISS.load_local(
                    index_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"Could not load existing index: {e}")
        
        return None
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            List of document IDs
        """
        if not documents:
            return []
        
        if self.vectorstore is None:
            # Create new index with first batch
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        else:
            # Add to existing index
            self.vectorstore.add_documents(documents)
        
        # Persist the index
        self._save()
        
        # Generate IDs (FAISS doesn't return IDs by default)
        return [f"doc_{i}" for i in range(len(documents))]
    
    def _save(self):
        """Save the FAISS index to disk."""
        if self.vectorstore:
            self.vectorstore.save_local(self._get_index_path())
    
    def similarity_search(
        self, 
        query: str, 
        k: Optional[int] = None
    ) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of relevant Document objects
        """
        if self.vectorstore is None:
            return []
        
        k = k or RETRIEVAL_K
        return self.vectorstore.similarity_search(query, k=k)
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: Optional[int] = None
    ) -> List[tuple[Document, float]]:
        """
        Search for similar documents with relevance scores.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of (Document, score) tuples
        """
        if self.vectorstore is None:
            return []
        
        k = k or RETRIEVAL_K
        return self.vectorstore.similarity_search_with_score(query, k=k)
    
    def as_retriever(self, **kwargs):
        """
        Get a retriever interface for the vector store.
        
        Args:
            **kwargs: Arguments passed to as_retriever
            
        Returns:
            VectorStoreRetriever
        """
        if self.vectorstore is None:
            raise ValueError("No documents in vector store yet")
        
        search_kwargs = kwargs.pop("search_kwargs", {"k": RETRIEVAL_K})
        return self.vectorstore.as_retriever(
            search_kwargs=search_kwargs,
            **kwargs
        )
    
    def get_collection_stats(self) -> dict:
        """Get statistics about the index."""
        count = 0
        if self.vectorstore and hasattr(self.vectorstore, 'index'):
            count = self.vectorstore.index.ntotal
        
        return {
            "name": self.index_name,
            "count": count,
            "persist_directory": self.persist_directory,
        }
    
    def clear(self):
        """Clear all documents from the index."""
        import shutil
        
        index_path = self._get_index_path()
        
        # Remove index files
        for ext in [".faiss", ".pkl"]:
            path = f"{index_path}{ext}"
            if os.path.exists(path):
                os.remove(path)
        
        # Reset vectorstore
        self.vectorstore = None
