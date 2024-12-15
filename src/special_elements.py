import cv2
import numpy as np
from typing import Dict, List, Tuple
import re
from dataclasses import dataclass

@dataclass
class SpecialElement:
    """Class to store information about detected special elements"""
    type: str  # 'chemical_formula', 'medical_symbol', 'diagram', etc.
    content: str
    confidence: float
    location: Tuple[int, int, int, int]  # x, y, width, height
    metadata: Dict

class SpecialElementsDetector:
    def __init__(self):
        # Регулярные выражения для распознавания химических формул
        self.chemical_formula_patterns = [
            r'[A-Z][a-z]?\d*',  # Простые элементы с числами (Na, Ca2)
            r'\([^)]+\)\d*',    # Группы в скобках ((SO4)2)
            r'·\d*H2O',         # Гидраты (·2H2O)
        ]
        
        # Словарь медицинских символов и их Unicode-представлений
        self.medical_symbols = {
            '♀': 'female',
            '♂': 'male',
            '℞': 'prescription',
            '†': 'deceased',
            '‡': 'double_dagger',
            '°': 'degree',
            'μ': 'micro',
            '±': 'plus_minus'
        }

    def detect_chemical_formulas(self, text: str) -> List[SpecialElement]:
        """Detect chemical formulas in text"""
        formulas = []
        combined_pattern = '|'.join(self.chemical_formula_patterns)
        
        for match in re.finditer(combined_pattern, text):
            formula = SpecialElement(
                type='chemical_formula',
                content=match.group(),
                confidence=0.9,  # Можно уточнить на основе контекста
                location=(0, 0, 0, 0),  # Требуется определение реальных координат
                metadata={'context': text[max(0, match.start()-20):match.end()+20]}
            )
            formulas.append(formula)
        
        return formulas

    def detect_medical_symbols(self, text: str) -> List[SpecialElement]:
        """Detect medical symbols in text"""
        symbols = []
        for char in text:
            if char in self.medical_symbols:
                symbol = SpecialElement(
                    type='medical_symbol',
                    content=char,
                    confidence=1.0,
                    location=(0, 0, 0, 0),
                    metadata={'meaning': self.medical_symbols[char]}
                )
                symbols.append(symbol)
        return symbols

    def detect_diagrams(self, image: np.ndarray) -> List[SpecialElement]:
        """Detect diagrams in images using computer vision"""
        diagrams = []
        
        # Преобразование в градации серого
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Обнаружение границ
        edges = cv2.Canny(gray, 50, 150)
        
        # Поиск контуров
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Фильтрация по размеру и форме
            area = cv2.contourArea(contour)
            if area > 1000:  # Минимальный размер для диаграммы
                x, y, w, h = cv2.boundingRect(contour)
                
                diagram = SpecialElement(
                    type='diagram',
                    content='',  # Можно добавить извлечение текста из области
                    confidence=0.8,
                    location=(x, y, w, h),
                    metadata={
                        'area': area,
                        'aspect_ratio': w/h
                    }
                )
                diagrams.append(diagram)
        
        return diagrams

    def detect_molecular_structures(self, image: np.ndarray) -> List[SpecialElement]:
        """Detect molecular structures in images"""
        structures = []
        
        # Преобразование в градации серого
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Применение адаптивного порога
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Поиск линий (связей между атомами)
        lines = cv2.HoughLinesP(
            binary, 1, np.pi/180, 50, 
            minLineLength=30, maxLineGap=10
        )
        
        if lines is not None:
            # Анализ геометрии линий для определения молекулярных структур
            structure = SpecialElement(
                type='molecular_structure',
                content='',
                confidence=0.7,
                location=(0, 0, 0, 0),  # Нужно вычислить общую область
                metadata={'line_count': len(lines)}
            )
            structures.append(structure)
        
        return structures

    def process_page(self, text: str, image: np.ndarray = None) -> List[SpecialElement]:
        """Process a page to detect all special elements"""
        elements = []
        
        # Обработка текста
        elements.extend(self.detect_chemical_formulas(text))
        elements.extend(self.detect_medical_symbols(text))
        
        # Обработка изображения, если оно предоставлено
        if image is not None:
            elements.extend(self.detect_diagrams(image))
            elements.extend(self.detect_molecular_structures(image))
        
        return elements