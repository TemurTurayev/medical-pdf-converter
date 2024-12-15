import os
import json
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import fitz
import cv2
import numpy as np
from tqdm import tqdm
import logging

from .special_elements import SpecialElementsDetector
from .terms_processor import MedicalTermsProcessor
from .table_processor import TableProcessor
from .performance_optimizer import PerformanceOptimizer, ProcessingTask

@dataclass
class ConversionConfig:
    """Configuration settings for document conversion"""
    mode: str = 'basic'  # 'basic' or 'advanced'
    use_ocr: bool = False
    extract_images: bool = True
    recognize_tables: bool = True
    detect_terms: bool = True
    detect_special_elements: bool = True
    output_format: str = 'json'
    optimization_level: str = 'medium'  # 'low', 'medium', 'high'

class MedicalDocumentConverter:
    def __init__(self, poppler_path: Optional[str] = None, tesseract_path: Optional[str] = None):
        self.poppler_path = poppler_path
        self.tesseract_path = tesseract_path
        self.setup_logging()
        
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        self.special_elements_detector = SpecialElementsDetector()
        self.terms_processor = MedicalTermsProcessor()
        self.table_processor = TableProcessor(tesseract_path)
        self.performance_optimizer = PerformanceOptimizer()
        
        self.stats = {
            'start_time': None,
            'processed_pages': 0,
            'extracted_images': 0,
            'detected_tables': 0,
            'detected_terms': 0,
            'processing_time': 0
        }

    def setup_logging(self):
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
        """Preprocess image for better OCR results"""
        img_array = np.array(image)
        optimized = self.performance_optimizer.optimize_image_processing(
            img_array,
            target_size=(2000, 2000)
        )
        
        gray = cv2.cvtColor(optimized, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        denoised = cv2.fastNlMeansDenoising(binary)
        
        return Image.fromarray(denoised)

    def process_page(self, page: fitz.Page, config: ConversionConfig) -> Dict:
        """Process single page with all enabled features"""
        page_content = {
            'type': 'page',
            'number': page.number + 1,
            'content': [],
            'metadata': {}
        }

        text = page.get_text()
        page_content['content'].append({
            'type': 'text',
            'content': text
        })

        if config.mode == 'advanced':
            pix = page.get_pixmap()
            img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
            img_array = np.array(img)

            if config.detect_special_elements:
                special_elements = self.special_elements_detector.process_page(text, img_array)
                if special_elements:
                    page_content['content'].append({
                        'type': 'special_elements',
                        'elements': [elem.__dict__ for elem in special_elements]
                    })

            if config.recognize_tables:
                tables = self.table_processor.detect_tables(img_array)
                for table_box in tables:
                    table = self.table_processor.extract_table_structure(img_array, table_box)
                    if table:
                        self.stats['detected_tables'] += 1
                        page_content['content'].append({
                            'type': 'table',
                            'content': table.__dict__
                        })

            if config.detect_terms:
                terms = self.terms_processor.process_text(text)
                if terms:
                    self.stats['detected_terms'] += len(terms)
                    page_content['content'].append({
                        'type': 'medical_terms',
                        'terms': [term.__dict__ for term in terms]
                    })

        return page_content

    def convert(self, input_path: str, output_path: str, config: ConversionConfig = None) -> None:
        """Convert medical document with specified configuration"""
        if config is None:
            config = ConversionConfig()

        self.stats['start_time'] = datetime.now()
        self.logger.info(f"Starting conversion of {input_path} in {config.mode} mode")
        
        try:
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            
            doc = fitz.open(input_path)
            result = {
                'metadata': {
                    'title': os.path.basename(input_path),
                    'date_processed': datetime.now().isoformat(),
                    'pages_count': len(doc),
                    'conversion_mode': config.mode,
                    'optimization_level': config.optimization_level
                },
                'content': []
            }

            tasks = []
            for page_num in range(len(doc)):
                task = ProcessingTask(
                    task_id=f'page_{page_num}',
                    function=self.process_page,
                    args=(doc[page_num], config),
                    kwargs={},
                    priority=1
                )
                tasks.append(task)

            use_processes = config.optimization_level == 'high'
            pages_content = self.performance_optimizer.process_batch(
                tasks,
                use_processes=use_processes
            )

            for page_num in range(len(doc)):
                task_id = f'page_{page_num}'
                if task_id in pages_content:
                    result['content'].append(pages_content[task_id])

            result = self.performance_optimizer.optimize_memory(result)
            performance_report = self.performance_optimizer.get_performance_report()
            result['metadata']['performance'] = performance_report

            if config.output_format == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
            else:
                with open(output_path, 'w', encoding='utf-8') as f:
                    for page in result['content']:
                        for content_item in page['content']:
                            if content_item['type'] == 'text':
                                f.write(content_item['content'] + '\n')

            self.stats['processed_pages'] = len(doc)
            self.stats['processing_time'] = (
                datetime.now() - self.stats['start_time']
            ).total_seconds()

            self.logger.info(
                f"Conversion completed in {self.stats['processing_time']:.2f} seconds. "
                f"Processed {self.stats['processed_pages']} pages, "
                f"detected {self.stats['detected_tables']} tables and "
                f"{self.stats['detected_terms']} medical terms."
            )
            
        except Exception as e:
            self.logger.error(f"Error during conversion: {str(e)}")
            raise

        finally:
            # Очистка кэша и освобождение ресурсов
            self.performance_optimizer.clear_cache()

if __name__ == "__main__":
    # Пример использования
    converter = MedicalDocumentConverter()
    
    # Расширенная конвертация с оптимизацией
    converter.convert(
        "sample.pdf",
        "output_advanced.json",
        ConversionConfig(
            mode='advanced',
            extract_images=True,
            recognize_tables=True,
            detect_terms=True,
            detect_special_elements=True,
            optimization_level='high'
        )
    )