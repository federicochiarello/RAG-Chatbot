from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage
from langchain_ollama import ChatOllama
import json

from langgraph.graph import StateGraph, START, END




# https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag_local/#local-models



import sys
sys.path.append('../')


import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
CHROMA_PATH = Path(os.getenv("CHROMA_PATH"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME")


from db.chroma_db_menager import ChromaDBManager
from get_embedding_function import get_embedding_function


chroma_db_manager = ChromaDBManager(
    persist_directory = '../../chroma_pdf_only',
    collection_name = COLLECTION_NAME,
    embedding_function = get_embedding_function()
)
chroma_db_manager.connect()
db = chroma_db_manager.get_db()
retriever = db.as_retriever(k=4)


local_llm = "llama3.1"
llm = ChatOllama(model=local_llm, temperature=0)
llm_json_mode = ChatOllama(model=local_llm, temperature=0, format="json")



#####################################################################################################################

"""




You a question re-writer that converts an input question to a better version that is optimized \n 
for vectorstore retrieval. Look at the initial and formulate an improved question. \n
Here is the initial question: \n\n {question}. Improved question with no preamble: \n 


Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}




Your goal is to re-write the user question , without the context provided by previous messages.
If the user question is already clear on its own, than do not modify it.
Do not change the meaning of the original user question or add unnecessary informations.
"""



contextualize_prompt = """You are an AI language model assistant.
Your task is to re-write the user question into a better version that is clear and can be understood on its own 
without the context provided by previous messages.
You have access to the context provided by previous messages and a summary of the chat history.

If the user question is already clear on its own, than do not modify it.
Do not change the meaning of the original user question or add unnecessary informations.

summary: {summary}

messages: [

{messages}

]

question: {question}

Contextualize the question using messages and summary. Return only the final re-written question. Do not include any preamble.
"""


# Pre-processing
def format_messages(messages):
    return "\n\n".join(msg.content for msg in messages)


def contextualize(state):
    """
    Contextualize user query using previous messages as context.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Update the user query using previous messages and chat history as context
    """
    print("\n---CONTEXTUALIZE---\n")

    #summary = state["summary"]
    summary = state.get("summary", "")
    raw_messages = state["messages"]
    messages = format_messages(raw_messages)
    question = raw_messages[-1].content
    #question = state["question"]

    contextualize_prompt_formatted = contextualize_prompt.format(summary=summary, messages=messages, question=question)

    print(contextualize_prompt_formatted)

    contextualized_question = llm.invoke([HumanMessage(content=contextualize_prompt_formatted)])

    #print(contextualized_question)

    return {"question": contextualized_question.content}



#####################################################################################################################



def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("\n---RETRIEVE---\n")
    question = state["question"]

    print(question)

    # Write retrieved documents to documents key in state
    documents = retriever.invoke(question)

    print("\nRetrieved documents:")
    for doc in documents:
        print("\n", doc, "\n")

    return {"documents": documents}



#####################################################################################################################



# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_prompt = """You are an assistant for question-answering tasks. 
Here is the context to use to answer the question:

{context} 

Think carefully about the above context. 
Now, review the user question:

{question}

Provide an answer to this questions using only the above context. 
Use three sentences maximum and keep the answer concise.

Answer:"""


def generate_from_resources(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("\n---GENERATE---\n")

    question = state["question"]
    documents = state["documents"]
    #loop_step = state.get("loop_step", 0)

    # RAG generation
    docs_txt = format_docs(documents)
    rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
    response = llm.invoke([HumanMessage(content=rag_prompt_formatted)])

    return {"messages": response}#, "loop_step": loop_step + 1}



#####################################################################################################################



def generate_without_resources(state):
    pass



#####################################################################################################################



def rewrite_query(state):
    pass




#####################################################################################################################



def need_summary(state):
    pass



#####################################################################################################################





# Doc grader instructions
doc_grader_instructions = """You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."""

# Grader prompt
doc_grader_prompt = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}. 
This carefully and objectively assess whether the document contains at least some information that is relevant to the question.
Return JSON with single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains at least some information that is relevant to the question."""



def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        doc_grader_prompt_formatted = doc_grader_prompt.format(
            document=d.page_content, question=question
        )
        result = llm_json_mode.invoke(
            [SystemMessage(content=doc_grader_instructions)]
            + [HumanMessage(content=doc_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)["binary_score"]
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "web_search": web_search}



#####################################################################################################################



### Edges



router_instructions = """You are an expert at routing user questions to the right source.

Those are the available sources:
 - 'vectorstore': source that retrieve documents from a vectorstore. The vectorstore contains documents related to climate change adaptation strategies and nature-based solutions.
 - 'unrelated': source that answer directly without retrieving documents (the question is unrelated to the documents).

Route the user to the right source:
 - Route to 'vectorstore': if the question is on topics related to climate change and nature-based solutions.
 - Route to 'unrelated': if the question can be answered directly without retrieving documents.

Route to 'unrelated' only if you are sure that the question does not need documents to be answered, otherwise if you are not sure, than route it to 'vectorstore'.

Return JSON with single key, datasource, that is 'vectorstore' or 'unrelated' depending on the question."""



def route_question(state):
    """
    Route question to web search or RAG

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    route_question = llm_json_mode.invoke(
        [SystemMessage(content=router_instructions)]
        + [HumanMessage(content=state["question"])]
    )
    source = json.loads(route_question.content)["datasource"]
    if source == "websearch":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"



#####################################################################################################################



def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"



#####################################################################################################################



# Hallucination grader instructions
hallucination_grader_instructions = """
You are a teacher grading a quiz. 
You will be given FACTS and a STUDENT ANSWER. 

Here is the grade criteria to follow:

(1) Ensure the STUDENT ANSWER is grounded in the FACTS. 
(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Score:
A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 
A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.
Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 
Avoid simply stating the correct answer at the outset."""

# Grader prompt
hallucination_grader_prompt = """FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}. 

Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER is grounded in the FACTS. And a key, explanation, that contains an explanation of the score."""




# Answer grader instructions
answer_grader_instructions = """You are a teacher grading a quiz. 
You will be given a QUESTION and a STUDENT ANSWER. 
Here is the grade criteria to follow:
(1) The STUDENT ANSWER helps to answer the QUESTION
Score:
A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 
The student can receive a score of yes if the answer contains extra information that is not explicitly asked for in the question.
A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.
Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 
Avoid simply stating the correct answer at the outset."""

# Grader prompt
answer_grader_prompt = """QUESTION: \n\n {question} \n\n STUDENT ANSWER: {generation}. 

Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER meets the criteria. And a key, explanation, that contains an explanation of the score."""





def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    max_retries = state.get("max_retries", 3)  # Default to 3 if not provided

    hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
        documents=format_docs(documents), generation=generation.content
    )
    result = llm_json_mode.invoke(
        [SystemMessage(content=hallucination_grader_instructions)]
        + [HumanMessage(content=hallucination_grader_prompt_formatted)]
    )
    grade = json.loads(result.content)["binary_score"]

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        # Test using question and generation from above
        answer_grader_prompt_formatted = answer_grader_prompt.format(
            question=question, generation=generation.content
        )
        result = llm_json_mode.invoke(
            [SystemMessage(content=answer_grader_instructions)]
            + [HumanMessage(content=answer_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)["binary_score"]
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        elif state["loop_step"] <= max_retries:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
        else:
            print("---DECISION: MAX RETRIES REACHED---")
            return "max retries"
    elif state["loop_step"] <= max_retries:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
    else:
        print("---DECISION: MAX RETRIES REACHED---")
        return "max retries"



#####################################################################################################################





