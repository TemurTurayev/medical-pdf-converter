from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseConverter(ABC):
    """Базовый класс для всех конвертеров"""
    
    @abstractmethod
    def convert(self, input_file: str, output_format: str) -> Dict[str, Any]:
        """Конвертировать файл в указанный формат"""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Получить список поддерживаемых форматов"""
        pass
    
    @abstractmethod
    def extract_metadata(self, input_file: str) -> Dict[str, Any]:
        """Извлечь метаданные из файла"""
        pass

class ConversionResult:
    def __init__(self, content: Any, metadata: Dict[str, Any], quality_score: float):
        self.content = content
        self.metadata = metadata
        self.quality_score = quality_score
        self.output_files = []
    
    def add_output_file(self, path: str, format: str):
        self.output_files.append({
            'path': path,
            'format': format
        })