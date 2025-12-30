from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import re

class TextProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def clean_text(self, text: str) -> str:
        """
        Clean the text by removing excessive whitespace and artifacts.
        """
        # Remove multiple newlines
        text = re.sub(r'\n+', '\n', text)
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def process(self, raw_texts: List[str], source: str) -> List[Document]:
        """
        Clean and chunk text into LangChain Documents.
        
        Args:
            raw_texts: List of strings (pages) from the loader.
            source: Source filename for metadata.
            
        Returns:
            List of Document objects with metadata.
        """
        documents = []
        for i, text in enumerate(raw_texts):
            clean_content = self.clean_text(text)
            if clean_content:
                # We create a temporary doc to split
                # Optional: You could split the whole text at once, but page-wise keeps context local
                chunks = self.text_splitter.create_documents(
                    [clean_content], 
                    metadatas=[{"source": source, "page": i + 1}]
                )
                documents.extend(chunks)
        return documents
