from pathlib import Path
from langchain_chroma import Chroma
from langchain.schema.document import Document
from typing import Callable
from .vector_db_interface import VectorDatabaseManager



class ChromaDBManager(VectorDatabaseManager):
    def __init__(self, persist_directory: Path, collection_name: str, embedding_function: Callable):
        self.db = None
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.persist_directory = str(persist_directory)
        self.collection_metadata = {"hnsw:space": "cosine"}

    def connect(self) -> None:
        """Establish a connection to ChromaDB."""
        self.db = Chroma(
            collection_name = self.collection_name,
            embedding_function = self.embedding_function,
            persist_directory = self.persist_directory,
            collection_metadata = self.collection_metadata,
        )
        print(f"\n----- ChromaDB CONNECTION -----\n\n - Persist Directory:\t{self.persist_directory}")
        print(f" - Collection Name:\t{self.collection_name}\n - Embedding Function:\t{self.embedding_function.__class__}\n - Collection Metadata:\t{self.collection_metadata}\n")

    def disconnect(self) -> None:
        pass

    def get_db(self):
        return self.db

    def upload_data(self, chunks: list[Document]) -> None:
        """Upload a list of documents to ChromaDB."""

        print(f"----- UPLOADING DOCUMENTS -----\n")

        # Get IDs of the documents in the DB
        # include=[] - get only the IDs
        existing_items = self.db.get(include=[])
        existing_ids = set(existing_items["ids"])
        print(f"Number of existing documents in DB: {len(existing_ids)}")

        # Calculate Page IDs for the new documents to be inserted in the DB.
        chunks_with_ids = calculate_chunk_ids(chunks)

        # Add documents that don't exist in the DB.
        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)

        if len(new_chunks):
            print(f"Number of new documents added to the DB: {len(new_chunks)}")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            self.db.add_documents(new_chunks, ids=new_chunk_ids)
            print("Upload completed.\n")
        else:
            print("No new documents to add.\n")

    def similarity_search_with_score(self, query_text: str, k=5):
        return self.db.similarity_search_with_score(query_text, k)



def calculate_chunk_ids(chunks):
    """
    Create chunks IDs like "data/example.pdf:6:2"
    [Page Source : Page Number : Chunk Index]

    Returns:
        list: A list of chunks with their IDs.
    """
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks