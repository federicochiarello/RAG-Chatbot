import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from db.chroma_db_manager import ChromaDBManager
from get_embedding_function import get_embedding_function


load_dotenv()
CHROMA_PATH = Path(os.getenv("CHROMA_PATH"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
LLAMA_MODEL = os.getenv("LLAMA_MODEL")
assert (len(LLAMA_MODEL)>0)


PROMPT_TEMPLATE = """You are a helpful and informative chatbot that answers questions using text from the reference material included below.
Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. 
However, you may be talking to a non-technical audience, so be sure to break down complicated concepts and strike a friendly and converstional tone. 

If the passage is irrelevant to the answer, you may ignore it.
If you don't know the answer, just say that "I don't know", don't try to make up an answer.
If the question is poorly worded or not meaningful and you feel it is unrelated to the reference material provided, then ask the user to rephrase the question.
No pre-amble in your answers.

You are a chatbot built to support transformative climate adaptation and achieve the following goals:
- Manage knowledge: Managing and organizing a growing volume and variety of knowledge, from different European projects
- Increased Efficiency: Promoting shared, participatory, and cross-sectoral practices for more robust and integrated decision-making and planning
- Bridging Knowledge and Accessibility: Eliminating barriers between scientific knowledge and regions, as well as among regional-to-local actors
- Operationalize NBS Strategies: Prioritize and select NBS strategies based on key data and different parameters
- Quick Access to Information: Rapid management and consultation of a wide knowledge base
- Support for Adaptation Pathways: Offering guidance on optimal strategies for climate adaptation across different landscape archetypes

---

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

    model = OllamaLLM(model=LLAMA_MODEL, temperature=0.7)
    #print("Calling Llama, waiting for response...")
    response_text = model.invoke(prompt)

    # sources = [doc.metadata.get("id", None) for doc, _score in results]
    
    print(f"----- RESOURCES -----\n")
    for doc, score in results: 
        id = doc.metadata.get("id", None)
        print(f"ID: {id}\tScore: {score}")

    print(f"\n----- RESPONSE -----\n\n{response_text}\n")

    return response_text


if __name__ == "__main__":
    main()