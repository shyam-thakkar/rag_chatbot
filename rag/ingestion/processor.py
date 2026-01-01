"""
Text processor with semantic chunking.
Preserves meaning by respecting sentence and paragraph boundaries.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import CHUNK_SIZE, CHUNK_OVERLAP
import re
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document



class SemanticTextProcessor:
    """
    Process and chunk text semantically to preserve meaning.
    Uses sentence and paragraph boundaries for better coherence.
    """
    
    def __init__(
        self, 
        chunk_size: Optional[int] = None, 
        chunk_overlap: Optional[int] = None
    ):
        """
        Initialize the semantic text processor.
        
        Args:
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks for context continuity
        """
        self.chunk_size = chunk_size or CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or CHUNK_OVERLAP
        
        # Semantic separators in order of preference:
        # 1. Double newlines (paragraphs)
        # 2. Single newlines 
        # 3. Sentence endings (. ! ?)
        # 4. Semicolons and colons
        # 5. Commas
        # 6. Spaces
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n",           # Paragraph breaks
                "\n",             # Line breaks
                ". ",             # Sentence endings
                "! ",
                "? ",
                "; ",             # Clause separators
                ": ",
                ", ",             # Phrase separators
                " ",              # Word boundaries
                ""                # Character level (last resort)
            ],
            length_function=len,
        )
    
    def clean_text(self, text: str) -> str:
        """
        Clean text while preserving semantic structure.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text with preserved structure
        """
        if not text:
            return ""
        
        # Remove excessive whitespace while keeping paragraph breaks
        # First, normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove excessive blank lines (more than 2 consecutive)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove trailing whitespace from each line
        lines = [line.rstrip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Remove leading/trailing whitespace from the entire text
        text = text.strip()
        
        return text
    
    def _extract_title_or_header(self, text: str) -> Optional[str]:
        """
        Try to extract a title or header from the beginning of text.
        
        Args:
            text: Text to search for headers
            
        Returns:
            Extracted header or None
        """
        lines = text.strip().split('\n')
        if lines:
            first_line = lines[0].strip()
            # Check if first line looks like a header (short, may be all caps or title case)
            if len(first_line) < 100 and first_line:
                return first_line
        return None
    
    def process(
        self, 
        raw_texts: List[str], 
        source: str,
        include_headers: bool = True
    ) -> List[Document]:
        """
        Process raw text pages into semantic chunks with metadata.
        
        Args:
            raw_texts: List of text strings (e.g., pages from a document)
            source: Source filename for metadata
            include_headers: Whether to try extracting section headers
            
        Returns:
            List of Document objects with metadata
        """
        documents = []
        
        for page_num, text in enumerate(raw_texts, start=1):
            clean_content = self.clean_text(text)
            
            if not clean_content:
                continue
            
            # Extract potential header for this page/section
            header = self._extract_title_or_header(clean_content) if include_headers else None
            
            # Create metadata
            base_metadata = {
                "source": source,
                "page": page_num,
            }
            if header:
                base_metadata["section"] = header
            
            # Split into semantic chunks
            chunks = self.text_splitter.create_documents(
                [clean_content],
                metadatas=[base_metadata]
            )
            
            # Add chunk index within page
            for chunk_idx, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = chunk_idx
                chunk.metadata["total_chunks_in_page"] = len(chunks)
            
            documents.extend(chunks)
        
        return documents
    
    def process_single(self, text: str, source: str = "unknown") -> List[Document]:
        """
        Process a single text string into semantic chunks.
        
        Args:
            text: Text to process
            source: Source identifier for metadata
            
        Returns:
            List of Document objects
        """
        return self.process([text], source)


# Keep backward compatibility
TextProcessor = SemanticTextProcessor
