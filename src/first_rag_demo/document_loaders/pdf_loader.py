from pathlib import Path
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.schema.document import Document


class PDFLoader:
    def __init__(self, pdf_dir: Path):
        """
        Initialize the PDFLoader with the path to a directory containing PDF files.

        Args:
            pdf_dir (str): Path to the directory containing PDF files.
        """
        self.pdf_dir = pdf_dir


    def load_pdf(self):
        """
        Initialize the PDFLoader with the path to a directory containing PDF files.

        Args:
            pdf_dir (str): Path to the directory containing PDF files.
        """
        pdf_documents = self.parse_pdf(self.pdf_dir)
        return pdf_documents


    def parse_pdf(self, directory: Path):
        """
        Processes a list of PDF files, loads data, and returns the aggregated data.

        Returns:
            list: A list containing the content of the PDF files.
        """

        print("\n----- LOADING PDF FILES -----\n")
        print(f"PDF directory: {str(directory)}")
        number_of_pdfs = sum(1 for f in directory.iterdir() if f.is_file() and f.suffix == '.pdf')
        print(f"PDF files to be loaded: {number_of_pdfs}")

        document_loader = PyPDFDirectoryLoader(str(directory))
        documents = document_loader.load()

        print("PDFs loading completed.\n")

        return documents