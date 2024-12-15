import os
from typing import Union, List, BinaryIO
from PIL import Image
import fitz
from pdf2image import convert_from_path
import pythoncom
from win32com import client
import numpy as np
import djvu.decode
import pptx
from io import BytesIO

class DocumentPreprocessor:
    def __init__(self, temp_dir: str = 'temp'):
        self.temp_dir = temp_dir
        self.supported_formats = {
            'pdf': self._process_pdf,
            'djvu': self._process_djvu,
            'pptx': self._process_pptx,
            'ppt': self._process_ppt,
            'png': self._process_image,
            'jpg': self._process_image,
            'jpeg': self._process_image,
            'tiff': self._process_image,
            'bmp': self._process_image
        }
        os.makedirs(temp_dir, exist_ok=True)
    
    def process_document(self, input_path: str) -> fitz.Document:
        """Convert any supported document to PDF format"""
        extension = input_path.lower().split('.')[-1]
        
        if extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {extension}")
        
        return self.supported_formats[extension](input_path)
    
    def _process_pdf(self, path: str) -> fitz.Document:
        """Process PDF files directly"""
        return fitz.open(path)
    
    def _process_djvu(self, path: str) -> fitz.Document:
        """Convert DjVu to PDF"""
        # Инициализация DjVu декодера
        context = djvu.decode.Context()
        document = context.new_document(djvu.decode.FileURI(path))
        document.decoding_job.wait()
        
        # Путь к временному PDF
        temp_pdf = os.path.join(self.temp_dir, 'temp_djvu.pdf')
        
        pdf_doc = fitz.open()
        for page_num in range(document.pages.count()):
            page = document.pages[page_num]
            # Получение изображения страницы
            pil_image = page.render(
                djvu.decode.RENDER_COLOR,
                scale=1,
                paged=True
            ).convert('RGB')
            
            # Добавление страницы в PDF
            img_bytes = BytesIO()
            pil_image.save(img_bytes, format='PNG')
            pdf_doc.insert_page(-1, stream=img_bytes.getvalue())
        
        return pdf_doc
    
    def _process_pptx(self, path: str) -> fitz.Document:
        """Convert PPTX to PDF"""
        # Открытие PPTX
        presentation = pptx.Presentation(path)
        pdf_doc = fitz.open()
        
        for slide in presentation.slides:
            # Сохранение слайда как изображения
            img_stream = BytesIO()
            slide.save(img_stream, 'PNG')
            
            # Добавление в PDF
            pdf_doc.insert_page(-1, stream=img_stream.getvalue())
        
        return pdf_doc
    
    def _process_ppt(self, path: str) -> fitz.Document:
        """Convert PPT to PDF using PowerPoint"""
        temp_pdf = os.path.join(self.temp_dir, 'temp_ppt.pdf')
        
        pythoncom.CoInitialize()
        powerpoint = client.Dispatch('PowerPoint.Application')
        presentation = powerpoint.Presentations.Open(path)
        
        try:
            # Сохранение как PDF
            presentation.SaveAs(temp_pdf, 32)  # 32 = PDF format
        finally:
            presentation.Close()
            powerpoint.Quit()
        
        # Открытие созданного PDF
        pdf_doc = fitz.open(temp_pdf)
        os.remove(temp_pdf)
        
        return pdf_doc
    
    def _process_image(self, path: str) -> fitz.Document:
        """Convert image to PDF"""
        image = Image.open(path)
        pdf_doc = fitz.open()
        
        # Преобразование в RGB если нужно
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Сохранение во временный буфер
        img_stream = BytesIO()
        image.save(img_stream, format='PNG')
        
        # Добавление в PDF
        pdf_doc.insert_page(-1, stream=img_stream.getvalue())
        
        return pdf_doc
        
    def __del__(self):
        """Cleanup temporary files"""
        try:
            for file in os.listdir(self.temp_dir):
                os.remove(os.path.join(self.temp_dir, file))
            os.rmdir(self.temp_dir)
        except:
            pass