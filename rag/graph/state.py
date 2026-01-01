"""
State definition for the LangGraph RAG workflow.
"""
from typing import List, Optional, TypedDict
from langchain_core.documents import Document


class RAGState(TypedDict):
    """
    State object passed between agents in the LangGraph workflow.
    """
    # Input
    question: str
    
    # Retrieved context
    context: List[Document]
    
    # Generated answer
    answer: str
    
    # Validation
    is_valid: bool
    validation_feedback: str
    
    # Retry tracking
    retry_count: int
    
    # Final output
    final_response: str
    sources: List[str]
