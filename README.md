# Medical PDF Converter

Instruments for converting medical PDF documents to text format, specially designed for preparing training data for Large Language Models (LLMs). This toolkit supports both native PDFs and scanned documents through OCR capabilities.

## Features

- **Basic PDF Conversion**: Extract text from native PDF files
- **OCR Support**: Convert scanned medical documents using Tesseract OCR
- **Unicode Support**: Properly handle multiple languages and special characters
- **Progress Tracking**: Real-time conversion progress with ETA
- **Batch Processing**: Process multiple PDF files in one go
- **Auto-recovery**: Save intermediate results to prevent data loss

## Prerequisites

- Python 3.8 or higher
- Tesseract OCR (with language packs)
- Poppler

## Installation

1. Clone this repository:
```bash
git clone https://github.com/TemurTurayev/medical-pdf-converter.git
cd medical-pdf-converter
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install Tesseract OCR:
   - Windows: Download from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
   - During installation, select required languages (at least English and Russian)

4. Install Poppler:
   - Windows: Download from [poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases/)

## Usage

### Basic PDF Conversion

For native PDF files with selectable text:

```python
from src.basic_converter import convert_pdf_to_txt

pdf_folder = "path/to/pdf/folder"
output_folder = "path/to/output/folder"

convert_pdf_to_txt(pdf_folder, output_folder)
```

### OCR Conversion

For scanned documents or PDFs without selectable text:

```python
from src.ocr_converter import OCRConverter

POPPLER_PATH = "path/to/poppler/bin"
TESSERACT_PATH = "path/to/tesseract.exe"

converter = OCRConverter(POPPLER_PATH, TESSERACT_PATH)
converter.convert_pdf("input.pdf", "output.txt")
```

## Project Structure

```
medical-pdf-converter/
├── src/
│   ├── basic_converter.py     # Basic PDF to text converter
│   └── ocr_converter.py       # OCR-enabled converter
└── requirements.txt           # Python dependencies
```

## License

This project is licensed under the MIT License.