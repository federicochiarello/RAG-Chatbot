import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from db.chroma_db_menager import ChromaDBManager
from get_embedding_function import get_embedding_function


load_dotenv()
CHROMA_PATH = Path(os.getenv("CHROMA_PATH"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME")


PROMPT_TEMPLATE = """You are a helpful and informative chatbot that answers questions using text from the reference material included below.
Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. 
However, you may be talking to a non-technical audience, so be sure to break down complicated concepts and strike a friendly and converstional tone. 
The main domain of the reference material provided is climate change and Nature-Based Solutions (NBS).

If the passage is irrelevant to the answer, you may ignore it.
If you don't know the answer, just say that "I don't know", don't try to make up an answer.

Answer this question: {question}

Base your answer on the following reference material: 

---

{context}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    query_rag(query_text)


def query_rag(query_text: str):

    chroma_db_manager = ChromaDBManager(
        persist_directory = CHROMA_PATH,
        collection_name = COLLECTION_NAME,
        embedding_function = get_embedding_function()
    )
    chroma_db_manager.connect()

    results = chroma_db_manager.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(f"----- PROMPT -----\n\n{prompt}\n")

    model = OllamaLLM(model="llama3.1", temperature=0.7)
    response_text = model.invoke(prompt)

    # sources = [doc.metadata.get("id", None) for doc, _score in results]
    
    print(f"----- RESOURCES -----\n")
    for doc, score in results: print(f"ID: {doc.metadata.get("id", None)}\tScore: {score}")

    print(f"\n----- RESPONSE -----\n\n{response_text}\n")

    return response_text


if __name__ == "__main__":
    main()