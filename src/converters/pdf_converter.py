import pdfplumber
import pytesseract
from PIL import Image
import io
from typing import Dict, Any, List
from ..core.base_converter import BaseConverter

class PDFConverter(BaseConverter):
    def __init__(self):
        self.supported_formats = ['txt', 'md', 'html']

    def convert(self, input_file: str, output_format: str) -> Dict[str, Any]:
        if output_format not in self.supported_formats:
            raise ValueError(f'Формат {output_format} не поддерживается')

        result = {
            'content': '',
            'images': [],
            'tables': []
        }

        with pdfplumber.open(input_file) as pdf:
            for page in pdf.pages:
                # Извлечение текста
                result['content'] += page.extract_text() + '\n'

                # Извлечение таблиц
                tables = page.extract_tables()
                if tables:
                    result['tables'].extend(tables)

                # Извлечение и OCR изображений
                for image in page.images:
                    img = Image.open(io.BytesIO(image['stream']))
                    # OCR для изображений
                    text = pytesseract.image_to_string(img, lang='rus+eng')
                    result['images'].append({
                        'image': img,
                        'text': text
                    })

        return result

    def get_supported_formats(self) -> List[str]:
        return self.supported_formats

    def extract_metadata(self, input_file: str) -> Dict[str, Any]:
        with pdfplumber.open(input_file) as pdf:
            metadata = pdf.metadata
            return {
                'title': metadata.get('Title', ''),
                'author': metadata.get('Author', ''),
                'creator': metadata.get('Creator', ''),
                'pages': len(pdf.pages),
                'has_images': any(page.images for page in pdf.pages),
                'has_tables': any(page.extract_tables() for page in pdf.pages)
            }