import os
from pathlib import Path
from dotenv import load_dotenv
from document_loaders.document_loader import DocumentLoader
from db.chroma_db_menager import ChromaDBManager
from get_embedding_function import get_embedding_function


load_dotenv()
PDF_PATH = Path(os.getenv("PDF_PATH"))
CSV_PATH = Path(os.getenv("CSV_PATH"))
CHROMA_PATH = Path(os.getenv("CHROMA_PATH"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME")


def main():

    document_loader = DocumentLoader(pdf_dir = PDF_PATH, csv_dir = CSV_PATH)
    pdf_chunks, csv_chunks = document_loader.load_documents()
    chunks = pdf_chunks + csv_chunks

    chroma_db_meneger = ChromaDBManager(
        persist_directory = CHROMA_PATH,
        collection_name = COLLECTION_NAME,
        embedding_function = get_embedding_function()
    )
    chroma_db_meneger.connect()
    chroma_db_meneger.upload_data(chunks = chunks)


if __name__ == "__main__":
    main()