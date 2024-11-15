from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)