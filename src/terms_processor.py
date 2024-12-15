import re
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
import spacy
from collections import defaultdict

@dataclass
class MedicalTerm:
    """Class to store information about medical terms"""
    term: str
    category: str  # 'disease', 'drug', 'procedure', etc.
    normalized_form: str
    context: str
    confidence: float
    related_terms: List[str]

class MedicalTermsProcessor:
    def __init__(self, spacy_model: str = 'en_core_sci_md'):
        """Initialize terms processor with medical NLP model"""
        self.nlp = spacy.load(spacy_model)
        
        # Загрузка базовых медицинских словарей
        self.load_medical_dictionaries()
        
    def load_medical_dictionaries(self):
        """Load medical terminology dictionaries"""
        # Базовые категории терминов
        self.term_categories = {
            'diseases': set(['cancer', 'diabetes', 'hypertension']),
            'drugs': set(['aspirin', 'insulin', 'paracetamol']),
            'procedures': set(['biopsy', 'surgery', 'ultrasound']),
            'anatomy': set(['heart', 'liver', 'brain']),
            'measurements': set(['blood pressure', 'temperature', 'pulse'])
        }
        
        # Словарь сокращений
        self.abbreviations = {
            'BP': 'blood pressure',
            'HR': 'heart rate',
            'Tx': 'treatment',
            'Dx': 'diagnosis',
            'abd': 'abdominal',
            'temp': 'temperature'
        }
        
        # Связанные термины
        self.related_terms = defaultdict(set)
        for category, terms in self.term_categories.items():
            for term in terms:
                self.related_terms[term].update(
                    other for other in terms if other != term
                )
    
    def normalize_term(self, term: str) -> str:
        """Normalize medical term to standard form"""
        # Приведение к нижнему регистру
        term = term.lower()
        
        # Замена сокращений
        if term.upper() in self.abbreviations:
            term = self.abbreviations[term.upper()]
        
        # Удаление лишних пробелов
        term = ' '.join(term.split())
        
        return term
    
    def categorize_term(self, term: str) -> Tuple[str, float]:
        """Determine category of medical term"""
        normalized = self.normalize_term(term)
        
        # Проверка по словарям
        for category, terms in self.term_categories.items():
            if normalized in terms:
                return category, 1.0
        
        # Использование NLP для неизвестных терминов
        doc = self.nlp(normalized)
        
        # Анализ именованных сущностей
        if doc.ents:
            for ent in doc.ents:
                if ent.label_ in ['DISEASE', 'CHEMICAL', 'PROCEDURE']:
                    return ent.label_.lower(), 0.8
        
        return 'unknown', 0.5
    
    def extract_context(self, text: str, term: str, window: int = 50) -> str:
        """Extract context around medical term"""
        term_pos = text.lower().find(term.lower())
        if term_pos == -1:
            return ''
        
        start = max(0, term_pos - window)
        end = min(len(text), term_pos + len(term) + window)
        
        return text[start:end]
    
    def find_related_terms(self, term: str) -> List[str]:
        """Find terms related to given medical term"""
        normalized = self.normalize_term(term)
        related = set()
        
        # Прямые связи из словаря
        if normalized in self.related_terms:
            related.update(self.related_terms[normalized])
        
        # Анализ с помощью NLP
        doc = self.nlp(normalized)
        for token in doc:
            # Добавление синонимов и похожих терминов
            if token.has_vector:
                similar = token.vocab.vectors.most_similar(
                    token.vector.reshape(1, -1), n=3
                )
                for word in similar:
                    if word in self.related_terms:
                        related.update(self.related_terms[word])
        
        return list(related)
    
    def process_text(self, text: str) -> List[MedicalTerm]:
        """Process text to extract medical terms"""
        terms = []
        doc = self.nlp(text)
        
        # Обработка каждого предложения
        for sent in doc.sents:
            # Поиск именованных сущностей
            for ent in sent.ents:
                if ent.label_ in ['DISEASE', 'CHEMICAL', 'PROCEDURE', 'ANATOMY']:
                    category, confidence = self.categorize_term(ent.text)
                    term = MedicalTerm(
                        term=ent.text,
                        category=category,
                        normalized_form=self.normalize_term(ent.text),
                        context=self.extract_context(text, ent.text),
                        confidence=confidence,
                        related_terms=self.find_related_terms(ent.text)
                    )
                    terms.append(term)
            
            # Поиск терминов по словарю
            for category, term_set in self.term_categories.items():
                for dict_term in term_set:
                    if dict_term in sent.text.lower():
                        term = MedicalTerm(
                            term=dict_term,
                            category=category,
                            normalized_form=self.normalize_term(dict_term),
                            context=self.extract_context(text, dict_term),
                            confidence=1.0,
                            related_terms=self.find_related_terms(dict_term)
                        )
                        terms.append(term)
        
        return terms
    
    def create_glossary(self, terms: List[MedicalTerm]) -> Dict[str, Dict]:
        """Create glossary from extracted terms"""
        glossary = {}
        
        for term in terms:
            if term.normalized_form not in glossary:
                glossary[term.normalized_form] = {
                    'category': term.category,
                    'variations': set([term.term]),
                    'related_terms': set(term.related_terms),
                    'contexts': set([term.context])
                }
            else:
                glossary[term.normalized_form]['variations'].add(term.term)
                glossary[term.normalized_form]['related_terms'].update(term.related_terms)
                glossary[term.normalized_form]['contexts'].add(term.context)
        
        # Преобразование множеств в списки для JSON-совместимости
        for term_info in glossary.values():
            term_info['variations'] = list(term_info['variations'])
            term_info['related_terms'] = list(term_info['related_terms'])
            term_info['contexts'] = list(term_info['contexts'])
        
        return glossary