from pathlib import Path
from .pdf_loader import PDFLoader
from .csv_loader import csv_loader
from .document_splitter import split_documents


class DocumentLoader:
    def __init__(self, pdf_dir: Path, csv_dir: Path):
        """
        Initialize the DocumentLoader with paths to directories containing PDFs and CSVs.

        Args:
            pdf_dir (str): Path to the directory containing PDF files
            csv_dir (str): Path to the directory containing CSV files
        """
        self.pdf_dir = pdf_dir
        self.csv_dir = csv_dir


    def load_documents(self):
        """
        Load documents.

        Returns:
            list: chunks from pdfs
            list: chunks from csvs (data contained in each row)
        """
        pdf_data = self.load_pdfs(self.pdf_dir)
        csv_data = self.load_csvs(self.csv_dir)

        return pdf_data, csv_data
    

    def load_pdfs(self, directory: Path):
        """
        Load PDF files.
        
        Args: 
            directory (str): Directory containing the PDF files

        Returns:
            list: Loaded PDF data splitted in chunks
        """
        pdf_loader = PDFLoader(directory)
        pdf_documents = pdf_loader.load_pdf()
        pdf_data = split_documents(pdf_documents)
        return pdf_data
    

    def load_csvs(self, directory: Path):
        """
        Load CSV files.
        
        Args: 
            directory (str): Directory containing the CSV files

        Returns:
            list: Loaded CSV data
        """
        csv_data = csv_loader(directory)
        return csv_data
