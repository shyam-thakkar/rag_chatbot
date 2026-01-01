"""
LangGraph workflow for the RAG pipeline.
Orchestrates Retriever -> Generator -> Validator -> Response with retry logic.
"""
from langgraph.graph import StateGraph, END

from .state import RAGState
from .nodes import (
    create_retriever_node,
    create_generator_node,
    create_validator_node,
    create_final_response_node,
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import MAX_RETRIES
from rag.retriever import VectorStore


def should_retry(state: RAGState) -> str:
    """
    Conditional edge: Decide whether to retry generation or proceed to response.
    
    Returns:
        "generate" if retry needed, "respond" otherwise
    """
    is_valid = state.get("is_valid", True)
    retry_count = state.get("retry_count", 0)
    
    if is_valid:
        return "respond"
    elif retry_count < MAX_RETRIES:
        return "generate"
    else:
        # Max retries reached, proceed anyway
        return "respond"


def create_rag_workflow(vector_store: VectorStore) -> StateGraph:
    """
    Create the LangGraph RAG workflow.
    
    Flow:
    START -> retrieve -> generate -> validate -> (retry or respond) -> END
    
    Args:
        vector_store: VectorStore instance for retrieval
        
    Returns:
        Compiled StateGraph
    """
    # Initialize the graph
    workflow = StateGraph(RAGState)
    
    # Add nodes
    workflow.add_node("retrieve", create_retriever_node(vector_store))
    workflow.add_node("generate", create_generator_node())
    workflow.add_node("validate", create_validator_node())
    workflow.add_node("respond", create_final_response_node())
    
    # Define edges
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "validate")
    
    # Conditional edge: retry or respond
    workflow.add_conditional_edges(
        "validate",
        should_retry,
        {
            "generate": "generate",
            "respond": "respond",
        }
    )
    
    workflow.add_edge("respond", END)
    
    return workflow.compile()


def run_rag_query(
    question: str, 
    vector_store: VectorStore
) -> dict:
    """
    Run a RAG query through the workflow.
    
    Args:
        question: User's question
        vector_store: VectorStore instance
        
    Returns:
        Final state with response
    """
    workflow = create_rag_workflow(vector_store)
    
    # Initialize state
    initial_state: RAGState = {
        "question": question,
        "context": [],
        "answer": "",
        "is_valid": False,
        "validation_feedback": "",
        "retry_count": 0,
        "final_response": "",
        "sources": [],
    }
    
    # Run the workflow
    final_state = workflow.invoke(initial_state)
    
    return final_state
