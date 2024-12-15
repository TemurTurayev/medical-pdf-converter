import os
import json
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import cv2
import numpy as np
from tqdm import tqdm
import logging

@dataclass
class ConversionConfig:
    """Configuration settings for document conversion"""
    mode: str = 'basic'  # 'basic' or 'advanced'
    use_ocr: bool = False
    extract_images: bool = True
    recognize_tables: bool = True
    detect_terms: bool = True
    output_format: str = 'json'

class MedicalDocumentConverter:
    def __init__(self, poppler_path: Optional[str] = None, tesseract_path: Optional[str] = None):
        """
        Initialize the medical document converter.
        
        Args:
            poppler_path: Path to poppler binaries (required for PDF to image conversion)
            tesseract_path: Path to tesseract executable (required for OCR)
        """
        self.poppler_path = poppler_path
        self.tesseract_path = tesseract_path
        self.setup_logging()
        
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        self.stats = {
            'start_time': None,
            'processed_pages': 0,
            'extracted_images': 0,
            'detected_tables': 0,
            'processing_time': 0
        }

    def setup_logging(self):
        """Configure logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('medical_converter.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for better OCR results
        
        Args:
            image: Input PIL Image
        Returns:
            Preprocessed PIL Image
        """
        # Convert PIL Image to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(binary)
        
        # Convert back to PIL Image
        return Image.fromarray(denoised)

    def extract_figures(self, pdf_path: str, output_dir: str) -> List[Dict]:
        """
        Extract figures and their metadata from PDF
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save extracted images
        Returns:
            List of dictionaries containing figure metadata
        """
        figures = []
        doc = fitz.open(pdf_path)
        
        os.makedirs(output_dir, exist_ok=True)
        
        for page_num, page in enumerate(doc):
            # Extract images
            image_list = page.get_images(full=True)
            
            for img_idx, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Save image
                    image_filename = f"figure_{page_num + 1}_{img_idx + 1}.png"
                    image_path = os.path.join(output_dir, image_filename)
                    
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)
                    
                    # Extract figure reference from surrounding text
                    rect = page.get_image_bbox(img)
                    text_around = page.get_text("text", clip=rect.expand(20))
                    
                    figure = {
                        "type": "figure",
                        "id": f"{page_num + 1}.{img_idx + 1}",
                        "page": page_num + 1,
                        "image_path": image_path,
                        "surrounding_text": text_around
                    }
                    
                    figures.append(figure)
                    self.stats['extracted_images'] += 1
                    
                except Exception as e:
                    self.logger.error(f"Error extracting image: {str(e)}")
                    continue
        
        return figures

    def convert(self, input_path: str, output_path: str, config: ConversionConfig = None) -> None:
        """
        Convert medical document with specified configuration
        
        Args:
            input_path: Path to input PDF file
            output_path: Path for output file
            config: Conversion configuration settings
        """
        if config is None:
            config = ConversionConfig()

        self.stats['start_time'] = datetime.now()
        self.logger.info(f"Starting conversion of {input_path} in {config.mode} mode")
        
        try:
            # Create output directory if needed
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # Basic text extraction
            doc = fitz.open(input_path)
            content = []
            
            # Process pages
            for page_num, page in enumerate(tqdm(doc, desc="Processing pages")):
                page_content = {
                    "type": "text",
                    "page": page_num + 1,
                    "content": page.get_text()
                }
                content.append(page_content)
                self.stats['processed_pages'] += 1

            # Advanced processing if needed
            if config.mode == 'advanced':
                # Extract figures
                if config.extract_images:
                    figures_dir = os.path.join(output_dir, "figures")
                    figures = self.extract_figures(input_path, figures_dir)
                    content.extend(figures)
            
            # Prepare output
            result = {
                "metadata": {
                    "title": os.path.basename(input_path),
                    "date_processed": datetime.now().isoformat(),
                    "pages_count": len(doc),
                    "conversion_mode": config.mode,
                    "ocr_used": config.use_ocr
                },
                "content": content,
                "stats": self.stats
            }
            
            # Save output
            if config.output_format == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
            else:
                # Save as plain text
                with open(output_path, 'w', encoding='utf-8') as f:
                    for item in content:
                        if item['type'] == 'text':
                            f.write(item['content'] + '\n')
            
            self.stats['processing_time'] = (datetime.now() - self.stats['start_time']).total_seconds()
            self.logger.info(f"Conversion completed successfully in {self.stats['processing_time']:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Error during conversion: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    converter = MedicalDocumentConverter()
    
    # Basic conversion
    converter.convert(
        "sample.pdf",
        "output_basic.txt",
        ConversionConfig(mode='basic', output_format='txt')
    )
    
    # Advanced conversion
    converter.convert(
        "sample.pdf",
        "output_advanced.json",
        ConversionConfig(
            mode='advanced',
            extract_images=True,
            recognize_tables=True,
            detect_terms=True
        )
    )