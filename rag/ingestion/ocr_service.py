"""
OCR Service using Ollama vision model (deepseek-ocr).
Extracts text from images using multimodal LLM.
"""
import base64
from io import BytesIO
from typing import List, Optional
from PIL import Image
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import OCR_MODEL, MAX_TOKENS, TEMPERATURE


class OCRService:
    """OCR service using Ollama vision model."""
    
    # Instruction prompt for OCR
    OCR_INSTRUCTION = """<image>\nFree OCR."""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the OCR service with Ollama vision model.
        
        Args:
            model_name: Ollama model name (default: from config)
        """
        self.model_name = model_name or OCR_MODEL
        self.llm = ChatOllama(
            model=self.model_name,
            temperature=TEMPERATURE,
            num_predict=MAX_TOKENS,
        )
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string (PNG)."""
        buffer = BytesIO()
        # Convert to RGB if necessary (handles RGBA, P mode, etc.)
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    def extract_text(self, image: Image.Image) -> str:
        """
        Extract text from a single image using OCR.
        
        Args:
            image: PIL Image object
            
        Returns:
            Extracted text string
        """
        img_b64 = self._image_to_base64(image)
        
        message = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_b64}"
                    },
                },
                {"type": "text", "text": self.OCR_INSTRUCTION},
            ]
        )
        
        response = self.llm.invoke([message])
        return response.content
    
    def extract_text_batch(self, images: List[Image.Image]) -> List[str]:
        """
        Extract text from multiple images in batch.
        
        Args:
            images: List of PIL Image objects
            
        Returns:
            List of extracted text strings
        """
        messages = []
        for image in images:
            img_b64 = self._image_to_base64(image)
            messages.append(
                HumanMessage(
                    content=[
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}"
                            },
                        },
                        {"type": "text", "text": self.OCR_INSTRUCTION},
                    ]
                )
            )
        
        # Batch process all images
        responses = self.llm.batch([[msg] for msg in messages])
        return [resp.content for resp in responses]
    
    def extract_text_from_path(self, image_path: str) -> str:
        """
        Extract text from an image file path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text string
        """
        image = Image.open(image_path).convert("RGB")
        return self.extract_text(image)
