import os
from typing import Dict, Any
from .base_converter import BaseConverter
from ..utils.file_utils import detect_file_type
from ..utils.quality_analyzer import QualityAnalyzer

class FileProcessor:
    """Основной класс для обработки файлов"""
    
    def __init__(self):
        self.converters: Dict[str, BaseConverter] = {}
        self.quality_analyzer = QualityAnalyzer()
    
    def register_converter(self, file_type: str, converter: BaseConverter):
        """Регистрация конвертера для определенного типа файлов"""
        self.converters[file_type] = converter
    
    def process_file(self, input_file: str, output_format: str, save_all: bool = False) -> Dict[str, Any]:
        """Обработка файла с выбранными параметрами"""
        file_type = detect_file_type(input_file)
        
        if file_type not in self.converters:
            raise ValueError(f'Неподдерживаемый тип файла: {file_type}')
        
        converter = self.converters[file_type]
        result = converter.convert(input_file, output_format)
        
        quality_score = self.quality_analyzer.analyze(result)
        metadata = converter.extract_metadata(input_file)
        
        if save_all:
            self._save_all_formats(result)
        
        return {
            'result': result,
            'quality_score': quality_score,
            'metadata': metadata
        }
    
    def _save_all_formats(self, result: Any):
        """Сохранение результата во всех поддерживаемых форматах"""
        # TODO: Реализовать сохранение во всех форматах
        pass