"""
Streamlit RAG Application.
Upload documents, chat with their content using LangGraph pipeline.
"""
import os
import tempfile
import streamlit as st
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import CHROMA_COLLECTION_NAME
from rag.ingestion import OCRService, PDFLoader, ImageLoader, SemanticTextProcessor
from rag.retriever import VectorStore
from rag.graph import run_rag_query


# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="RAG Document Chat",
    page_icon="📚",
    layout="wide",
)

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stChatMessage {
        border-radius: 10px;
    }
    .source-box {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 10px;
        margin-top: 10px;
        font-size: 0.85rem;
    }
    .status-success {
        color: #059669;
        font-weight: 600;
    }
    .status-error {
        color: #DC2626;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================
# SESSION STATE INITIALIZATION
# =====================================================
def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False
    if "doc_count" not in st.session_state:
        st.session_state.doc_count = 0


# =====================================================
# DOCUMENT PROCESSING
# =====================================================
def process_uploaded_file(uploaded_file) -> tuple[bool, str, int]:
    """
    Process an uploaded file through the ingestion pipeline.
    
    Returns:
        Tuple of (success, message, chunk_count)
    """
    try:
        # Initialize services
        ocr_service = OCRService()
        processor = SemanticTextProcessor()
        
        # Get or create vector store
        if st.session_state.vector_store is None:
            st.session_state.vector_store = VectorStore()
        
        # Save uploaded file temporarily
        suffix = Path(uploaded_file.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Load document based on type
            if suffix == ".pdf":
                loader = PDFLoader(ocr_service=ocr_service)
                raw_texts = loader.load(tmp_path)
            elif suffix in [".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"]:
                loader = ImageLoader(ocr_service=ocr_service)
                raw_texts = loader.load(tmp_path)
            else:
                return False, f"Unsupported file type: {suffix}", 0
            
            # Process into chunks
            documents = processor.process(raw_texts, source=uploaded_file.name)
            
            if not documents:
                return False, "No text content extracted from document", 0
            
            # Add to vector store
            st.session_state.vector_store.add_documents(documents)
            st.session_state.documents_loaded = True
            st.session_state.doc_count += len(documents)
            
            return True, f"Successfully processed {uploaded_file.name}", len(documents)
            
        finally:
            # Clean up temp file
            os.unlink(tmp_path)
            
    except Exception as e:
        return False, f"Error processing file: {str(e)}", 0


def get_rag_response(question: str) -> tuple[str, list]:
    """
    Get RAG response for a question.
    
    Returns:
        Tuple of (response, sources)
    """
    if st.session_state.vector_store is None:
        return "Please upload a document first.", []
    
    try:
        result = run_rag_query(question, st.session_state.vector_store)
        response = result.get("final_response", "No response generated.")
        sources = result.get("sources", [])
        return response, sources
    except Exception as e:
        return f"Error generating response: {str(e)}", []


# =====================================================
# UI COMPONENTS
# =====================================================
def render_sidebar():
    """Render the sidebar with document upload."""
    with st.sidebar:
        st.markdown("## 📄 Document Upload")
        st.markdown("Upload PDF or image files to chat with their content.")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf", "png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            help="Upload PDF documents or images for OCR processing"
        )
        
        # Process button
        if uploaded_files:
            if st.button("🚀 Process Documents", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                total_chunks = 0
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {file.name}...")
                    success, message, chunks = process_uploaded_file(file)
                    
                    if success:
                        total_chunks += chunks
                        st.success(f"✓ {file.name}: {chunks} chunks")
                    else:
                        st.error(f"✗ {file.name}: {message}")
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text(f"Done! Total: {total_chunks} chunks indexed")
        
        # Stats
        st.markdown("---")
        st.markdown("## 📊 Statistics")
        
        if st.session_state.vector_store:
            try:
                stats = st.session_state.vector_store.get_collection_stats()
                st.metric("Documents Indexed", stats.get("count", 0))
            except:
                st.metric("Chunks Indexed", st.session_state.doc_count)
        else:
            st.info("No documents loaded yet")
        
        # Clear button
        if st.session_state.documents_loaded:
            st.markdown("---")
            if st.button("🗑️ Clear All Documents", use_container_width=True):
                if st.session_state.vector_store:
                    st.session_state.vector_store.clear()
                st.session_state.messages = []
                st.session_state.documents_loaded = False
                st.session_state.doc_count = 0
                st.rerun()


def render_chat():
    """Render the main chat interface."""
    # Header
    st.markdown('<h1 class="main-header">📚 RAG Document Chat</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload documents and ask questions about their content</p>', unsafe_allow_html=True)
    
    # Chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("📎 Sources"):
                    for source in message["sources"]:
                        st.markdown(f"- {source}")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            if not st.session_state.documents_loaded:
                response = "Please upload and process a document first using the sidebar."
                sources = []
            else:
                with st.spinner("Thinking..."):
                    response, sources = get_rag_response(prompt)
            
            st.markdown(response)
            
            if sources:
                with st.expander("📎 Sources"):
                    for source in sources:
                        st.markdown(f"- {source}")
        
        # Save assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources
        })


# =====================================================
# MAIN
# =====================================================
def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
