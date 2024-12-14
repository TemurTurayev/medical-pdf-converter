# Medical PDF Converter

A comprehensive tool for converting medical PDF documents to text format, specifically designed for preparing training data for Large Language Models (LLMs). This tool supports both native PDFs and scanned documents through OCR capabilities.

## Features

- **Basic PDF Conversion**: Extract text from native PDF files
- **OCR Support**: Convert scanned medical documents using Tesseract OCR
- **Unicode Support**: Properly handle multiple languages and special characters
- **Progress Tracking**: Real-time conversion progress with ETA
- **Batch Processing**: Process multiple PDF files in one go
- **Auto-recovery**: Save intermediate results to prevent data loss
- **Clean Output**: Remove unnecessary formatting while preserving document structure

## Installation

1. Clone this repository:
```bash
git clone https://github.com/TemurTurayev/medical-pdf-converter.git
cd medical-pdf-converter
```

2. Install required Python packages:
```bash
pip install -r requirements.txt
```

3. Install additional dependencies:

For OCR support:
- Install Tesseract OCR: [Windows Installation Guide](https://github.com/UB-Mannheim/tesseract/wiki)
- Install Poppler: [Windows Installation Guide](https://github.com/oschwartz10612/poppler-windows/releases/)

## Usage

### Basic PDF Conversion
```python
python src/basic_converter.py --input "path/to/pdf" --output "path/to/output"
```

### OCR-enabled Conversion
```python
python src/ocr_converter.py --input "path/to/pdf" --output "path/to/output"
```

## Project Structure

```
medical-pdf-converter/
├── src/
│   ├── basic_converter.py     # Basic PDF to text converter
│   └── ocr_converter.py       # OCR-enabled converter
├── requirements.txt           # Python dependencies
└── docs/
    ├── installation.md        # Detailed installation instructions
    └── usage.md              # Usage examples and documentation
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.