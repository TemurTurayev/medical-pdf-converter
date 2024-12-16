from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseConverter(ABC):
    @abstractmethod
    def convert(self, input_path: str, output_path: str, config: Dict[str, Any]) -> None:
        """Convert input document to desired format"""
        pass

    @abstractmethod
    def extract_metadata(self, input_path: str) -> Dict[str, Any]:
        """Extract metadata from the document"""
        pass
