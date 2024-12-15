import os
from PyPDF2 import PdfReader
from tqdm import tqdm

def convert_pdf_to_txt(pdf_folder: str, output_folder: str) -> None:
    """
    Convert PDF files from a folder to text files.
    
    Args:
        pdf_folder (str): Path to folder containing PDF files
        output_folder (str): Path to output folder for text files
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get list of PDF files
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
    
    # Process each PDF file
    for pdf_file in tqdm(pdf_files, desc="Converting PDFs"):
        try:
            # Open PDF file
            pdf_path = os.path.join(pdf_folder, pdf_file)
            reader = PdfReader(pdf_path)
            
            # Create output text file path
            txt_file = os.path.splitext(pdf_file)[0] + '.txt'
            txt_path = os.path.join(output_folder, txt_file)
            
            # Extract text from each page
            with open(txt_path, 'w', encoding='utf-8') as f:
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        f.write(text + '\n\n')
                        
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")
            continue

if __name__ == "__main__":
    # Example usage
    pdf_folder = "pdfs"
    output_folder = "output"
    convert_pdf_to_txt(pdf_folder, output_folder)