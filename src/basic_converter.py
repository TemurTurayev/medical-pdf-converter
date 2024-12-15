import os
from pypdf import PdfReader
from pathlib import Path
import time
import re

def clean_text(text):
    if not text or text.isspace():
        return ""
    
    text = re.sub(r'\t+', ' ', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    special_chars = {
        '®': '(R)',
        '©': '(c)',
        '™': '(TM)',
        '°': ' degrees ',
        '±': '+/-',
        '≤': '<=',
        '≥': '>=',
        '−': '-',
        '–': '-',
        '—': '-',
        '"': '"',
        '"': '"',
        ''': "'",
        ''': "'"
    }
    for char, replacement in special_chars.items():
        text = text.replace(char, replacement)
    
    text = re.sub(r'(?<=\n)\s*•\s*', '• ', text)
    text = re.sub(r'(?<=\n)\s*\d+\.\s*', lambda m: m.group().strip() + ' ', text)
    text = '\n'.join(line.strip() for line in text.splitlines())
    return text.strip()

def convert_pdf_to_txt(pdf_folder, output_folder):
    print(f"\nНачало конвертации файлов")
    pdf_files = list(Path(pdf_folder).glob('*.pdf'))
    total_files = len(pdf_files)
    print(f"Найдено PDF файлов: {total_files}")
    
    for index, pdf_path in enumerate(pdf_files, 1):
        try:
            print(f"\nОбработка файла {index}/{total_files}: {pdf_path.name}")
            reader = PdfReader(pdf_path)
            txt_path = Path(output_folder) / f"{pdf_path.stem}.txt"
            
            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                full_text = []
                for page_num, page in enumerate(reader.pages, 1):
                    text = page.extract_text()
                    if text and not text.isspace():
                        text = clean_text(text)
                        if text:
                            full_text.append(text)
                    print(f"  Страница {page_num}/{len(reader.pages)}", end='\r')
                
                if not full_text:
                    print(f"\nПРЕДУПРЕЖДЕНИЕ: Не удалось извлечь текст из {pdf_path.name}")
                else:
                    final_text = '\n\n'.join(full_text)
                    txt_file.write(final_text)
                    print(f"\nФайл обработан успешно")
                    
        except Exception as e:
            print(f"ОШИБКА при обработке {pdf_path.name}: {str(e)}")

if __name__ == "__main__":
    pdf_folder = r"C:\Users\user\Desktop\pdf"
    output_folder = r"C:\Users\user\Desktop\txt"
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    convert_pdf_to_txt(pdf_folder, output_folder)