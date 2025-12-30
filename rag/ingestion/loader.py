import os
from typing import List, Optional
from abc import ABC, abstractmethod
from pypdf import PdfReader
from PIL import Image

class BaseLoader(ABC):
    """Abstract base class for document loaders."""
    
    @abstractmethod
    def load(self, file_path: str) -> List[str]:
        """Loads a file and returns a list of text pages or content."""
        pass

class PDFLoader(BaseLoader):
    """Loader for PDF documents."""
    
    def load(self, file_path: str) -> List[str]:
        """Extracts text from a PDF file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        text_content = []
        try:
            reader = PdfReader(file_path)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_content.append(text)
                else:
                    text_content.append("") # Empty page or image-only
        except Exception as e:
            print(f"Error loading PDF {file_path}: {e}")
            raise
            
        return text_content

class ImageLoader(BaseLoader):
    """Loader for Image-based documents (requires OCR later)."""
    
    def load(self, file_path: str) -> List[Image.Image]:
        """Loads an image file and returns the Image object."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        try:
            image = Image.open(file_path)
            return [image]
        except Exception as e:
            print(f"Error loading Image {file_path}: {e}")
            raise
