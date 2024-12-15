import os
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
from tqdm import tqdm

class OCRConverter:
    def __init__(self, poppler_path: str, tesseract_path: str):
        """
        Initialize OCR converter with required paths.
        
        Args:
            poppler_path (str): Path to poppler binaries
            tesseract_path (str): Path to tesseract executable
        """
        self.poppler_path = poppler_path
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

    def convert_pdf(self, pdf_path: str, output_path: str, 
                   languages: str = 'eng+rus', dpi: int = 300) -> None:
        """
        Convert PDF to text using OCR.
        
        Args:
            pdf_path (str): Path to input PDF file
            output_path (str): Path for output text file
            languages (str): Languages for OCR (default: 'eng+rus')
            dpi (int): DPI for PDF to image conversion (default: 300)
        """
        try:
            # Convert PDF to images
            images = convert_from_path(pdf_path, dpi=dpi, 
                                     poppler_path=self.poppler_path)
            
            # Process each page
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, image in enumerate(tqdm(images, desc="Processing pages")):
                    # Extract text using OCR
                    text = pytesseract.image_to_string(image, lang=languages)
                    if text:
                        f.write(f"\n--- Page {i+1} ---\n\n")
                        f.write(text)
                        
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            raise

    def batch_convert(self, input_folder: str, output_folder: str, 
                      languages: str = 'eng+rus', dpi: int = 300) -> None:
        """
        Convert multiple PDF files using OCR.
        
        Args:
            input_folder (str): Folder containing PDF files
            output_folder (str): Folder for output text files
            languages (str): Languages for OCR (default: 'eng+rus')
            dpi (int): DPI for PDF to image conversion (default: 300)
        """
        os.makedirs(output_folder, exist_ok=True)
        
        pdf_files = [f for f in os.listdir(input_folder) 
                    if f.lower().endswith('.pdf')]
        
        for pdf_file in tqdm(pdf_files, desc="Converting files"):
            try:
                pdf_path = os.path.join(input_folder, pdf_file)
                txt_file = os.path.splitext(pdf_file)[0] + '.txt'
                txt_path = os.path.join(output_folder, txt_file)
                
                self.convert_pdf(pdf_path, txt_path, languages, dpi)
                
            except Exception as e:
                print(f"Error processing {pdf_file}: {str(e)}")
                continue

if __name__ == "__main__":
    # Example usage
    POPPLER_PATH = r"C:\path\to\poppler\bin"
    TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    
    converter = OCRConverter(POPPLER_PATH, TESSERACT_PATH)
    converter.convert_pdf("input.pdf", "output.txt")