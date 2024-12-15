import pytesseract
from pdf2image import convert_from_path
import os
from pathlib import Path
from tqdm import tqdm
import time

class OCRConverter:
    def __init__(self, poppler_path, tesseract_path):
        self.poppler_path = poppler_path
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

    def save_progress(self, text_list, output_path, current_page):
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(text_list))
        print(f"\nProgress saved: {current_page} pages processed")

    def convert_pdf(self, pdf_path, output_path, start_page=1):
        start_time = time.time()
        print(f"\nProcessing file: {pdf_path.name}")
        
        try:
            print("Getting document info...")
            images = convert_from_path(
                pdf_path,
                poppler_path=self.poppler_path,
                first_page=start_page
            )
            
            total_pages = len(images)
            print(f"Total pages to process: {total_pages}")
            
            total_text = []
            
            for i, image in enumerate(tqdm(images, desc="Processing pages", unit="pg")):
                current_page = start_page + i
                
                if i % 5 == 0:
                    elapsed_time = time.time() - start_time
                    pages_per_minute = (i + 1) / (elapsed_time / 60) if elapsed_time > 0 else 0
                    remaining_pages = total_pages - (i + 1)
                    estimated_time = remaining_pages / pages_per_minute if pages_per_minute > 0 else 0
                    
                    print(f"\nPage {current_page}/{total_pages}")
                    print(f"Speed: {pages_per_minute:.1f} pages/min")
                    print(f"Est. time remaining: {estimated_time:.1f} minutes")
                
                try:
                    text = pytesseract.image_to_string(image, lang='rus+eng')
                    if text.strip():
                        total_text.append(f"[Page {current_page}]\n{text}")
                    
                    if i % 20 == 0 and i > 0:
                        self.save_progress(total_text, output_path, current_page)
                        
                except Exception as e:
                    print(f"\nError on page {current_page}: {str(e)}")
                    continue
            
            self.save_progress(total_text, output_path, current_page)
            
            total_time = time.time() - start_time
            print(f"\nDone! Processed {total_pages} pages in {total_time/60:.1f} minutes")
            print(f"Results saved to: {output_path}")
            
        except Exception as e:
            print(f"\nError processing file: {str(e)}")

def main():
    POPPLER_PATH = r"C:\Program Files\poppler\poppler-24.08.0\Library\bin"
    TESSERACT_PATH = r"C:\Users\user\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
    
    pdf_path = Path(r"C:\Users\user\Desktop\pdf\input.pdf")
    output_path = Path(r"C:\Users\user\Desktop\txt\output.txt")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=== Starting PDF to Text OCR Conversion ===")
    converter = OCRConverter(POPPLER_PATH, TESSERACT_PATH)
    converter.convert_pdf(pdf_path, output_path)
    
    print("\nPress Enter to exit...")
    input()

if __name__ == "__main__":
    main()