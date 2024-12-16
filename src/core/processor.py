from typing import Optional, Dict, Any
from pathlib import Path
from .base import BaseConverter

class MedicalDocumentProcessor:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.converters = {}
        self.language_model = None

    def register_converter(self, file_type: str, converter: BaseConverter) -> None:
        """Register a converter for a specific file type"""
        self.converters[file_type] = converter

    def _detect_file_type(self, input_path: str) -> str:
        """Smart file type detection based on content and extension"""
        ext = Path(input_path).suffix.lower()
        # TODO: Implement smart detection using file signatures
        return ext

    def process_document(self, input_path: str, output_path: str) -> None:
        """Process a medical document with the appropriate converter"""
        file_type = self._detect_file_type(input_path)
        if file_type not in self.converters:
            raise ValueError(f'Unsupported file type: {file_type}')

        converter = self.converters[file_type]
        converter.convert(input_path, output_path, self.config)
