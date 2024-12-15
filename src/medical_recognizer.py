import re
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import cv2
from PIL import Image

@dataclass
class MedicalTerm:
    term: str
    category: str  # e.g., 'drug', 'disease', 'procedure'
    position: tuple  # (page_num, start_pos, end_pos)
    context: str

@dataclass
class TableData:
    page: int
    data: pd.DataFrame
    title: Optional[str]
    reference: Optional[str]

class MedicalRecognizer:
    def __init__(self):
        # Загрузка словарей медицинских терминов
        self.drug_patterns = self._load_drug_patterns()
        self.disease_patterns = self._load_disease_patterns()
        self.procedure_patterns = self._load_procedure_patterns()

    def _load_drug_patterns(self) -> List[str]:
        """Загрузка паттернов для лекарственных препаратов"""
        # В будущем можно загружать из файла или базы данных
        return [
            r'\b\d+(?:\.\d+)?\s*(?:мг|г|мл|МЕ)\b',  # дозировки
            r'\b[A-ZА-Я][a-zа-я]+(?:um|in|ol|il|en|on)\b',  # типичные окончания лекарств
        ]

    def _load_disease_patterns(self) -> List[str]:
        """Загрузка паттернов для заболеваний"""
        return [
            r'\b(?:syndrome|disease|disorder|infection)\b',
            r'\b(?:синдром|болезнь|расстройство|инфекция)\b',
        ]

    def _load_procedure_patterns(self) -> List[str]:
        """Загрузка паттернов для медицинских процедур"""
        return [
            r'\b(?:therapy|surgery|procedure|treatment)\b',
            r'\b(?:терапия|операция|процедура|лечение)\b',
        ]

    def detect_tables(self, image: np.ndarray) -> List[np.ndarray]:
        """Обнаружение таблиц на изображении"""
        # Преобразование в оттенки серого
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Бинаризация
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Поиск горизонтальных и вертикальных линий
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)

        # Объединение линий
        table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0)
        
        # Поиск контуров таблиц
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        tables = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 100 and h > 100:  # фильтрация маленьких контуров
                table_region = image[y:y+h, x:x+w]
                tables.append(table_region)
        
        return tables

    def extract_table_data(self, table_image: np.ndarray) -> pd.DataFrame:
        """Извлечение данных из изображения таблицы"""
        # Преобразование изображения для OCR
        pil_image = Image.fromarray(cv2.cvtColor(table_image, cv2.COLOR_BGR2RGB))
        
        # TODO: Добавить OCR для извлечения текста из таблицы
        # Пока возвращаем пустой DataFrame
        return pd.DataFrame()

    def find_medical_terms(self, text: str, page_num: int) -> List[MedicalTerm]:
        """Поиск медицинских терминов в тексте"""
        terms = []
        
        # Поиск лекарств
        for pattern in self.drug_patterns:
            for match in re.finditer(pattern, text):
                term = MedicalTerm(
                    term=match.group(),
                    category='drug',
                    position=(page_num, match.start(), match.end()),
                    context=text[max(0, match.start()-50):min(len(text), match.end()+50)]
                )
                terms.append(term)
        
        # Поиск заболеваний
        for pattern in self.disease_patterns:
            for match in re.finditer(pattern, text):
                term = MedicalTerm(
                    term=match.group(),
                    category='disease',
                    position=(page_num, match.start(), match.end()),
                    context=text[max(0, match.start()-50):min(len(text), match.end()+50)]
                )
                terms.append(term)
        
        # Поиск процедур
        for pattern in self.procedure_patterns:
            for match in re.finditer(pattern, text):
                term = MedicalTerm(
                    term=match.group(),
                    category='procedure',
                    position=(page_num, match.start(), match.end()),
                    context=text[max(0, match.start()-50):min(len(text), match.end()+50)]
                )
                terms.append(term)
        
        return terms

    def analyze_medical_image(self, image: np.ndarray) -> Dict:
        """Анализ медицинского изображения"""
        # Определение типа изображения (рентген, МРТ, схема и т.д.)
        image_type = self._detect_image_type(image)
        
        # Поиск важных областей
        regions_of_interest = self._find_roi(image)
        
        # Определение ориентации и масштаба
        orientation = self._detect_orientation(image)
        
        return {
            'type': image_type,
            'regions': regions_of_interest,
            'orientation': orientation
        }

    def _detect_image_type(self, image: np.ndarray) -> str:
        """Определение типа медицинского изображения"""
        # TODO: Реализовать определение типа изображения
        return 'unknown'

    def _find_roi(self, image: np.ndarray) -> List[Dict]:
        """Поиск областей интереса на изображении"""
        # TODO: Реализовать поиск ROI
        return []

    def _detect_orientation(self, image: np.ndarray) -> str:
        """Определение ориентации изображения"""
        # TODO: Реализовать определение ориентации
        return 'unknown'
