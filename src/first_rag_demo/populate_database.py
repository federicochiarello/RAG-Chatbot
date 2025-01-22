import os
from pathlib import Path
from dotenv import load_dotenv
from document_loaders.document_loader import DocumentLoader
from db.chroma_db_manager import ChromaDBManager
from get_embedding_function import get_embedding_function
import pickle
import os

#load env variables from the .env file
load_dotenv()
PDF_PATH = Path(os.getenv("PDF_PATH"))
CSV_PATH = Path(os.getenv("CSV_PATH"))
CHROMA_PATH = Path(os.getenv("CHROMA_PATH"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

assert len(COLLECTION_NAME) > 0

def load_documents():
    '''Retrieve chunks from pdf and csv files. 

    This function caches the chunks into a .pkl file.
    To refresh the data, delete or rename the existing chunks.pkl file
    '''
    json_path = CHROMA_PATH / "chunks.pkl"
    if json_path.exists():
        with open(json_path, 'rb') as f:
            chunks = pickle.load(f, encoding='utf8')
            return chunks

    document_loader = DocumentLoader(pdf_dir = PDF_PATH, csv_dir = CSV_PATH)
    chunks = document_loader.load_pdfs(directory = PDF_PATH)
    print(len(chunks))

    # Use those lines if you need to load and embed csv files
    #pdf_chunks, csv_chunks = document_loader.load_documents()
    #chunks = pdf_chunks + csv_chunks

    with open(json_path, 'wb') as f:
        pickle.dump(chunks, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    return chunks


def main(chunks):

    chroma_db_manager = ChromaDBManager(
        persist_directory = CHROMA_PATH,
        collection_name = COLLECTION_NAME,
        embedding_function = get_embedding_function()
    )
    chroma_db_manager.connect()
    chroma_db_manager.upload_data(chunks = chunks)


if __name__ == "__main__":
    chunks = load_documents()
    main(chunks)