import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM, ChatOllama


from db.chroma_db_menager import ChromaDBManager
from get_embedding_function import get_embedding_function





load_dotenv()
CHROMA_PATH = Path(os.getenv("CHROMA_PATH"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME")


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




#######################################################################################

chroma_db_manager = ChromaDBManager(
    persist_directory = '../../chroma_pdf_only',
    collection_name = COLLECTION_NAME,
    embedding_function = get_embedding_function()
)
chroma_db_manager.connect()
db = chroma_db_manager.get_db()
retriever = db.as_retriever()




### Generate
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

# Prompt
prompt = hub.pull("rlm/rag-prompt")

# LLM
local_llm = "llama3.1"
llm = ChatOllama(
    model=local_llm, 
    temperature=0,
    #base_url="http://host.docker.internal:11434"
)


# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = prompt | llm | StrOutputParser()





from typing import List
from typing_extensions import TypedDict

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]



def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}



def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}






from langgraph.graph import END, StateGraph, START

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)  # retrieve
#workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae
#workflow.add_node("transform_query", transform_query)  # transform_query

# Build graph
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

'''
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    },
)
'''

# Compile
graph = workflow.compile()













def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text


    from pprint import pprint

    # Run
    inputs = {"question": query_text}
    for output in graph.stream(inputs):
        for key, value in output.items():
            # Node
            pprint(f"Node '{key}':")
            # Optional: print full state at each node
            pprint(value, indent=2, width=80, depth=None)
        pprint("\n---\n")

    # Final generation
    pprint(value["generation"])


    #query_rag(query_text)

'''
def query_rag(query_text: str):

    chroma_db_manager = ChromaDBManager(
        persist_directory = '../../chroma_pdf_only',
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
'''

if __name__ == "__main__":
    main()