import os
import magic
from typing import Optional

def detect_file_type(file_path: str) -> str:
    """Определение типа файла по содержимому и расширению"""
    mime = magic.Magic(mime=True)
    file_type = mime.from_file(file_path)
    extension = os.path.splitext(file_path)[1].lower()
    
    type_mapping = {
        'application/pdf': 'pdf',
        'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'pptx',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
        'text/html': 'html',
        'text/plain': 'text',
        'application/json': 'json',
        'text/csv': 'csv',
        'application/xml': 'xml'
    }
    
    return type_mapping.get(file_type, 'unknown')