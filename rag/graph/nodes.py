"""
Agent nodes for the LangGraph RAG workflow.
Implements: Retriever, Generator, Validator, and Final Response agents.
"""
from typing import List
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .state import RAGState

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import CHAT_MODEL, TEMPERATURE, MAX_TOKENS
from rag.retriever import VectorStore


# =====================================================
# SHARED LLM INSTANCE
# =====================================================
def get_llm():
    """Get the shared LLM instance."""
    return ChatOllama(
        model=CHAT_MODEL,
        temperature=TEMPERATURE,
        num_predict=MAX_TOKENS,
    )


# =====================================================
# RETRIEVER AGENT
# =====================================================
def retriever_agent(state: RAGState, vector_store: VectorStore) -> RAGState:
    """
    Retriever Agent: Fetches relevant document chunks from the vector store.
    
    Args:
        state: Current workflow state
        vector_store: VectorStore instance
        
    Returns:
        Updated state with retrieved context
    """
    question = state["question"]
    
    # Retrieve relevant documents
    documents = vector_store.similarity_search(question)
    
    # Extract source information
    sources = list(set([
        f"{doc.metadata.get('source', 'unknown')} (page {doc.metadata.get('page', '?')})"
        for doc in documents
    ]))
    
    return {
        **state,
        "context": documents,
        "sources": sources,
    }


# =====================================================
# GENERATOR AGENT
# =====================================================
GENERATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that answers questions based on the provided context.
Use ONLY the information from the context to answer the question.
If the context doesn't contain enough information to answer, say so clearly.
Be concise but thorough in your answer."""),
    ("human", """Context:
{context}

Question: {question}

Answer:"""),
])


def generator_agent(state: RAGState) -> RAGState:
    """
    Generator Agent: Uses LLM to generate answers based on retrieved context.
    
    Args:
        state: Current workflow state with context
        
    Returns:
        Updated state with generated answer
    """
    question = state["question"]
    context = state["context"]
    
    # Format context for the prompt
    context_text = "\n\n".join([
        f"[Source: {doc.metadata.get('source', 'unknown')}, Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in context
    ])
    
    # Generate answer
    llm = get_llm()
    chain = GENERATOR_PROMPT | llm | StrOutputParser()
    
    answer = chain.invoke({
        "context": context_text,
        "question": question,
    })
    
    return {
        **state,
        "answer": answer,
    }


# =====================================================
# VALIDATOR AGENT
# =====================================================
VALIDATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a validator that checks if an answer is relevant and accurate.
Evaluate the answer against the original question and context.

Check for:
1. Relevance: Does the answer address the question?
2. Groundedness: Is the answer supported by the context?
3. Completeness: Does the answer fully address the question?

Respond with either:
- "VALID" if the answer is good
- "INVALID: [reason]" if there are issues"""),
    ("human", """Question: {question}

Context:
{context}

Answer: {answer}

Evaluation:"""),
])


def validator_agent(state: RAGState) -> RAGState:
    """
    Validator Agent: Evaluates the generated answer for relevance and hallucinations.
    
    Args:
        state: Current workflow state with answer
        
    Returns:
        Updated state with validation result
    """
    question = state["question"]
    context = state["context"]
    answer = state["answer"]
    retry_count = state.get("retry_count", 0)
    
    # Format context for validation
    context_text = "\n\n".join([doc.page_content for doc in context])
    
    # Validate answer
    llm = get_llm()
    chain = VALIDATOR_PROMPT | llm | StrOutputParser()
    
    validation_result = chain.invoke({
        "question": question,
        "context": context_text,
        "answer": answer,
    })
    
    is_valid = validation_result.strip().upper().startswith("VALID")
    
    return {
        **state,
        "is_valid": is_valid,
        "validation_feedback": validation_result,
        "retry_count": retry_count + 1 if not is_valid else retry_count,
    }


# =====================================================
# FINAL RESPONSE AGENT
# =====================================================
def final_response_agent(state: RAGState) -> RAGState:
    """
    Final Response Agent: Formats and returns the validated answer.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with final formatted response
    """
    answer = state["answer"]
    sources = state.get("sources", [])
    is_valid = state.get("is_valid", True)
    retry_count = state.get("retry_count", 0)
    
    # Format the final response
    if sources:
        source_text = "\n\n**Sources:**\n" + "\n".join([f"- {s}" for s in sources])
    else:
        source_text = ""
    
    # Add warning if answer failed validation after max retries
    if not is_valid and retry_count > 0:
        warning = "\n\n*Note: This response may not fully address your question. Please verify the information.*"
    else:
        warning = ""
    
    final_response = f"{answer}{source_text}{warning}"
    
    return {
        **state,
        "final_response": final_response,
    }


# =====================================================
# NODE FUNCTIONS FOR LANGGRAPH
# =====================================================
def create_retriever_node(vector_store: VectorStore):
    """Create a retriever node with bound vector store."""
    def node(state: RAGState) -> RAGState:
        return retriever_agent(state, vector_store)
    return node


def create_generator_node():
    """Create a generator node."""
    return generator_agent


def create_validator_node():
    """Create a validator node."""
    return validator_agent


def create_final_response_node():
    """Create a final response node."""
    return final_response_agent
