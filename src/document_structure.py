from typing import List, Dict, Optional
from dataclasses import dataclass
import re

@dataclass
class Section:
    title: str
    level: int
    content: str
    subsections: List['Section']
    page: int
    start_pos: int
    end_pos: int

class DocumentStructureAnalyzer:
    def __init__(self):
        self.heading_patterns = self._load_heading_patterns()
        self.reference_patterns = self._load_reference_patterns()

    def _load_heading_patterns(self) -> List[Dict]:
        return [
            {
                'pattern': r'^\d+\.\s+[A-ZА-Я][^\.\n]+$',
                'level': 1
            },
            {
                'pattern': r'^\d+\.\d+\.\s+[A-ZА-Я][^\.\n]+$',
                'level': 2
            },
            {
                'pattern': r'^\d+\.\d+\.\d+\.\s+[A-ZА-Я][^\.\n]+$',
                'level': 3
            }
        ]

    def _load_reference_patterns(self) -> Dict[str, str]:
        return {
            'figure': r'(?:рис(?:унок|\.)?|fig\.?)\s*(?:\d+(?:\.\d+)*)',
            'table': r'(?:табл(?:ица|\.)?|tab\.?)\s*(?:\d+(?:\.\d+)*)',
            'reference': r'\[\d+(?:-\d+)?\]'
        }

    def analyze_structure(self, text: str, page: int) -> Section:
        """Анализ структуры документа"""
        root = Section(
            title='root',
            level=0,
            content='',
            subsections=[],
            page=page,
            start_pos=0,
            end_pos=len(text)
        )
        
        current_section = root
        current_level = 0
        start_pos = 0
        
        lines = text.split('\n')
        for i, line in enumerate(lines):
            # Поиск заголовков
            for pattern in self.heading_patterns:
                if re.match(pattern['pattern'], line.strip()):
                    level = pattern['level']
                    
                    # Закрытие текущей секции
                    if current_section != root:
                        current_section.content = '\n'.join(lines[start_pos:i])
                        current_section.end_pos = i
                    
                    # Создание новой секции
                    new_section = Section(
                        title=line.strip(),
                        level=level,
                        content='',
                        subsections=[],
                        page=page,
                        start_pos=i,
                        end_pos=len(lines)
                    )
                    
                    # Определение родительской секции
                    parent = root
                    for section in self._find_parent_section(root, level):
                        parent = section
                    
                    parent.subsections.append(new_section)
                    current_section = new_section
                    current_level = level
                    start_pos = i + 1
                    break
        
        # Закрытие последней секции
        if current_section != root:
            current_section.content = '\n'.join(lines[start_pos:])
            current_section.end_pos = len(lines)
        
        return root

    def _find_parent_section(self, section: Section, target_level: int) -> List[Section]:
        """Поиск родительской секции для заданного уровня"""
        path = []
        
        def find_path(current: Section, level: int) -> bool:
            if not current.subsections:
                return False
            
            # Поиск подходящей секции
            for subsection in current.subsections:
                if subsection.level < level:
                    path.append(subsection)
                    if find_path(subsection, level):
                        return True
                    path.pop()
            
            return len(path) > 0 and path[-1].level == level - 1
        
        find_path(section, target_level)
        return path

    def find_references(self, text: str) -> Dict[str, List[Dict]]:
        """Поиск ссылок на рисунки, таблицы и источники"""
        references = {
            'figures': [],
            'tables': [],
            'citations': []
        }
        
        # Поиск ссылок на рисунки
        for match in re.finditer(self.reference_patterns['figure'], text):
            references['figures'].append({
                'text': match.group(),
                'position': (match.start(), match.end()),
                'context': text[max(0, match.start()-50):min(len(text), match.end()+50)]
            })
        
        # Поиск ссылок на таблицы
        for match in re.finditer(self.reference_patterns['table'], text):
            references['tables'].append({
                'text': match.group(),
                'position': (match.start(), match.end()),
                'context': text[max(0, match.start()-50):min(len(text), match.end()+50)]
            })
        
        # Поиск цитирования
        for match in re.finditer(self.reference_patterns['reference'], text):
            references['citations'].append({
                'text': match.group(),
                'position': (match.start(), match.end()),
                'context': text[max(0, match.start()-50):min(len(text), match.end()+50)]
            })
        
        return references