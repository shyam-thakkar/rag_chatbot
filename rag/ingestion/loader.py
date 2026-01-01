"""
Document loaders for PDF and image files.
Supports OCR for scanned documents and images.
"""
import os
import sys
from pathlib import Path
from typing import List, Union
from abc import ABC, abstractmethod
from pypdf import PdfReader
from PIL import Image

# Import config for poppler path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import POPPLER_PATH

# Try to import pdf2image for converting PDF pages to images
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    print("Warning: pdf2image not available. PDF OCR will be limited.")


def get_poppler_path():
    """Get poppler path for Windows."""
    # First check config/env
    if POPPLER_PATH and os.path.exists(POPPLER_PATH):
        return POPPLER_PATH
    
    # Common Windows locations
    common_paths = [
        r"C:\poppler\poppler-25.12.0\Library\bin",  # User's path
        r"C:\poppler\Library\bin",
        r"C:\poppler\bin",
        r"C:\Program Files\poppler\bin",
        r"C:\Program Files\poppler\Library\bin",
        r"C:\ProgramData\chocolatey\lib\poppler\tools\Library\bin",
        os.path.expanduser(r"~\poppler\Library\bin"),
        os.path.expanduser(r"~\poppler\bin"),
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            print(f"[OCR] Found poppler at: {path}")
            return path
    
    return None  # Let pdf2image try system PATH


class BaseLoader(ABC):
    """Abstract base class for document loaders."""
    
    @abstractmethod
    def load(self, file_path: str) -> List[str]:
        """Loads a file and returns a list of text pages or content."""
        pass


class PDFLoader(BaseLoader):
    """
    Loader for PDF documents.
    Supports both text-based PDFs and scanned PDFs (with OCR).
    """
    
    def __init__(self, ocr_service=None, force_ocr: bool = True):
        """
        Initialize PDF loader.
        
        Args:
            ocr_service: Optional OCRService instance for OCR
            force_ocr: If True, always use OCR (recommended for accuracy).
                       If False, only use OCR for pages with no extractable text.
        """
        self.ocr_service = ocr_service
        self.force_ocr = force_ocr
    
    def load(self, file_path: str) -> List[str]:
        """
        Extract text from a PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of text strings, one per page
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # If force_ocr and we have OCR service, use OCR for all pages
        if self.force_ocr and self.ocr_service and PDF2IMAGE_AVAILABLE:
            return self._load_with_ocr(file_path)
        else:
            return self._load_with_text_extraction(file_path)
    
    def _load_with_ocr(self, file_path: str) -> List[str]:
        """Load PDF using OCR for all pages."""
        print(f"[OCR] Processing PDF with Ollama OCR: {file_path}")
        
        try:
            # Get poppler path for Windows
            poppler_path = get_poppler_path()
            if poppler_path:
                print(f"[OCR] Using poppler from: {poppler_path}")
            
            # Convert all pages to images
            images = convert_from_path(file_path, poppler_path=poppler_path)
            print(f"[OCR] Converted {len(images)} pages to images")
            
            text_content = []
            for i, image in enumerate(images):
                print(f"[OCR] Processing page {i + 1}/{len(images)}...")
                text = self.ocr_service.extract_text(image)
                text_content.append(text)
                print(f"[OCR] Page {i + 1} extracted: {len(text)} chars")
            
            return text_content
            
        except Exception as e:
            print(f"[OCR] Error during OCR: {e}")
            print("[OCR] Falling back to text extraction...")
            return self._load_with_text_extraction(file_path)
    
    def _load_with_text_extraction(self, file_path: str) -> List[str]:
        """Load PDF using pypdf text extraction (fallback)."""
        text_content = []
        pages_needing_ocr = []
        
        try:
            reader = PdfReader(file_path)
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                
                if text and text.strip():
                    text_content.append(text)
                else:
                    # Page has no extractable text - might be scanned
                    text_content.append("")  # Placeholder
                    pages_needing_ocr.append(page_num)
            
            # If OCR is available and needed, process scanned pages
            if pages_needing_ocr and self.ocr_service and PDF2IMAGE_AVAILABLE:
                try:
                    print(f"[OCR] Running OCR on {len(pages_needing_ocr)} scanned pages...")
                    
                    # Convert all pages and pick the ones we need
                    poppler_path = get_poppler_path()
                    images = convert_from_path(file_path, poppler_path=poppler_path)
                    
                    for page_num in pages_needing_ocr:
                        if page_num < len(images):
                            print(f"[OCR] Processing page {page_num + 1}...")
                            ocr_text = self.ocr_service.extract_text(images[page_num])
                            text_content[page_num] = ocr_text
                            
                except Exception as e:
                    print(f"[OCR] OCR fallback failed: {e}")
                    
        except Exception as e:
            print(f"Error loading PDF {file_path}: {e}")
            raise
        
        return text_content
    
    def load_with_images(self, file_path: str) -> tuple[List[str], List[Image.Image]]:
        """
        Load PDF and return both text and page images.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Tuple of (text_list, image_list)
        """
        if not PDF2IMAGE_AVAILABLE:
            raise RuntimeError("pdf2image is required for load_with_images")
        
        text_content = self.load(file_path)
        images = convert_from_path(file_path)
        
        return text_content, images


class ImageLoader(BaseLoader):
    """Loader for image-based documents."""
    
    def __init__(self, ocr_service=None):
        """
        Initialize image loader.
        
        Args:
            ocr_service: OCRService instance for text extraction
        """
        self.ocr_service = ocr_service
    
    def load(self, file_path: str) -> List[str]:
        """
        Load an image and extract text using OCR.
        
        Args:
            file_path: Path to image file
            
        Returns:
            List with single text string from OCR
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            image = Image.open(file_path).convert("RGB")
            
            if self.ocr_service:
                print(f"[OCR] Processing image with Ollama OCR: {file_path}")
                text = self.ocr_service.extract_text(image)
                print(f"[OCR] Extracted: {len(text)} chars")
                return [text]
            else:
                print("[OCR] Warning: No OCR service configured!")
                return [""]
                
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            raise
    
    def load_image(self, file_path: str) -> Image.Image:
        """
        Load and return the image object.
        
        Args:
            file_path: Path to image file
            
        Returns:
            PIL Image object
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        return Image.open(file_path).convert("RGB")
