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
        # Регулярные выражения для формул
        self.chemical_formula_patterns = [
            r'[A-Z][a-z]?\d*',  # Elements with numbers (Na, Ca2)
            r'\([^)]+\)\d*',    # Groups with parentheses ((SO4)2)
            r'·\d*H2O',         # Hydrates (·2H2O)
        ]
        
        # Медицинские символы
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

        # Шаблоны для медицинских диаграмм
        self.diagram_types = {
            'ecg': {'min_peaks': 3, 'aspect_ratio': (3, 1)},
            'brain_scan': {'min_area': 10000, 'aspect_ratio': (1, 1)},
            'xray': {'intensity_range': (0.2, 0.8), 'min_area': 50000}
        }

    def detect_chemical_formulas(self, text: str) -> List[SpecialElement]:
        """Detect chemical formulas in text"""
        formulas = []
        combined_pattern = '|'.join(self.chemical_formula_patterns)
        
        for match in re.finditer(combined_pattern, text):
            formula = SpecialElement(
                type='chemical_formula',
                content=match.group(),
                confidence=0.9,
                location=(0, 0, 0, 0),
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
        """Detect and classify medical diagrams"""
        diagrams = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum size threshold
                x, y, w, h = cv2.boundingRect(contour)
                
                # Extract region
                roi = gray[y:y+h, x:x+w]
                
                # Analyze region properties
                diagram_type = self._classify_diagram(roi, area, w/h)
                
                if diagram_type:
                    diagram = SpecialElement(
                        type=f'diagram_{diagram_type}',
                        content='',
                        confidence=0.8,
                        location=(x, y, w, h),
                        metadata={
                            'area': area,
                            'aspect_ratio': w/h,
                            'classification': diagram_type
                        }
                    )
                    diagrams.append(diagram)
        
        return diagrams

    def _classify_diagram(self, roi: np.ndarray, area: float, aspect_ratio: float) -> str:
        """Classify diagram type based on image properties"""
        # Check for ECG pattern
        if aspect_ratio > 2.5 and self._count_peaks(roi) > 5:
            return 'ecg'
        
        # Check for brain scan
        elif 0.8 < aspect_ratio < 1.2 and area > 10000:
            if self._check_brain_scan_features(roi):
                return 'brain_scan'
        
        # Check for X-ray
        elif area > 50000:
            intensity = np.mean(roi) / 255.0
            if 0.2 < intensity < 0.8:
                return 'xray'
        
        return 'unknown'

    def _count_peaks(self, signal: np.ndarray) -> int:
        """Count number of peaks in signal (for ECG detection)"""
        # Simplify signal to 1D
        if len(signal.shape) > 1:
            signal = np.mean(signal, axis=0)
        
        # Find peaks
        peaks, _ = np.histogram(signal, bins=50)
        return len(peaks[peaks > np.mean(peaks)])

    def _check_brain_scan_features(self, image: np.ndarray) -> bool:
        """Check if image has characteristics of brain scan"""
        # Calculate image statistics
        mean = np.mean(image)
        std = np.std(image)
        
        # Brain scans typically have good contrast
        if std < 30:
            return False
        
        # Check for symmetry
        height, width = image.shape
        left_half = image[:, :width//2]
        right_half = np.fliplr(image[:, width//2:])
        symmetry = np.corrcoef(left_half.flatten(), right_half.flatten())[0,1]
        
        return symmetry > 0.7

    def process_page(self, text: str, image: np.ndarray = None) -> List[SpecialElement]:
        """Process a page to detect all special elements"""
        elements = []
        
        # Process text
        elements.extend(self.detect_chemical_formulas(text))
        elements.extend(self.detect_medical_symbols(text))
        
        # Process image if provided
        if image is not None:
            elements.extend(self.detect_diagrams(image))
        
        return elements