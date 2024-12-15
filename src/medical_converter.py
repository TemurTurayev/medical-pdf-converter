from .document_preprocessor import DocumentPreprocessor

class MedicalDocumentConverter:
    def __init__(self, poppler_path: Optional[str] = None, tesseract_path: Optional[str] = None):
        self.poppler_path = poppler_path
        self.tesseract_path = tesseract_path
        self.setup_logging()
        
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Инициализация компонентов
        self.document_preprocessor = DocumentPreprocessor()
        self.special_elements_detector = SpecialElementsDetector()
        self.terms_processor = MedicalTermsProcessor()
        self.table_processor = TableProcessor(tesseract_path)
        self.performance_optimizer = PerformanceOptimizer()
        
        # ... (остальной код класса остается без изменений)

    def convert(self, input_path: str, output_path: str, config: ConversionConfig = None) -> None:
        if config is None:
            config = ConversionConfig()

        self.stats['start_time'] = datetime.now()
        self.logger.info(f"Starting conversion of {input_path} in {config.mode} mode")
        
        try:
            # Преобразование входного файла в PDF
            doc = self.document_preprocessor.process_document(input_path)
            
            # ... (остальной код метода convert остается без изменений)