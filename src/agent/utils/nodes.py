from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama

from .state import GraphState

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv


# https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag_local/#local-models


sys.path.append('../')
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
#
#   CONTEXTUALIZE
#
#####################################################################################################################


contextualize_prompt = """You are an AI language model assistant. Your task is to re-write the user question into a better version 
that is clear and can be understood on its own without the context provided by previous messages.
You have access previous messages and a summary of the chat history.

summary: {summary}

messages: [

{messages}]

question: {question}

Contextualize the question using messages and summary. 

- Do not change the meaning of the original user question or add unnecessary informations.
- Give higher priority to the information arriving from recent messages.
- Return only the final re-written question. 
- Do not include any preamble.
"""


# Pre-processing
def format_messages(messages):
    """
    Convert a list of LangChain Messages into a formatted string.
    
    Example:

    messages = [
    HumanMessage(content='Hi', additional_kwargs={}, response_metadata={}, id='123'), 
    AIMessage(content='Hola', additional_kwargs={}, response_metadata={...}, id='456')
    ]

    formatted_messages = "User message 1: Hi

    Model Answer 1: Hola
    
    "

    Args:
        messages (list): list of LangChain Messages

    Returns:
        formatted_messages(str): String containing the formatted messages
    """
    formatted_messages = ""
    for i, message in enumerate(messages):
        if message.__class__.__name__ == "HumanMessage":
            role = "User question"
            index = i + 1
        else:
            role = "Model answer"
            index = i
        formatted_messages = formatted_messages + f"{role} {index}: {message.content}\n\n"

    return formatted_messages


def contextualize(state):
    """
    Contextualize user query using previous messages and chat history as context.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updated user query field
    """
    print("\n---CONTEXTUALIZE---\n")

    langchain_messages = state["messages"]      # list of Lanchain Messages with metadata           
    messages = format_messages(langchain_messages)
    summary = state.get("summary", "")
    question = langchain_messages[-1].content

    # if it is the first message, skip contextualization
    # set both questions to original value
    if len(langchain_messages) == 1:
        return {"user_question": question, "contextualized_question": question}
    
    contextualize_prompt_formatted = contextualize_prompt.format(summary=summary, messages=messages, question=question)
    print(contextualize_prompt_formatted)
    contextualized_question = llm.invoke([HumanMessage(content=contextualize_prompt_formatted)])

    return {"user_question": question, "contextualized_question": contextualized_question.content}




#####################################################################################################################
#
#   RETRIEVE
#
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
    question = state["contextualized_question"]
    print("Question used for retrieval: ", question)
    documents = retriever.invoke(question)

    print("\nRetrieved documents:")
    for index, document in enumerate(documents):
        print(f"\nDocument {index+1}:\n\n", document, "\n")

    return {"documents": documents}




#####################################################################################################################
#
#   GENERATE WITH RESOURCES
#
#####################################################################################################################


# Pre-processing
def format_docs(docs):
    # return "\n\n".join(f"Document {index+1}:\n\n" + doc.page_content for index, doc in enumerate(docs))
    return "\n\n".join(doc.page_content for doc in docs)


rag_prompt = """You are a helpful and informative assistent for question-answering tasks.
You have been designed to support transformative climate adaptation by managing knowledge from different European projects, supporting decision-making and planning, 
bridging accessibility gaps between regional and local actors, operationalizing nature-based solutions (nbs), enabling quick information access, and guiding adaptation strategies.

Here is the context to use to answer the question:

[

{context}

]

Think carefully about the above context. 
Now, review the user question:

{question}

Provide an answer to this questions using only the above context. 

- You may be talking to a non-technical audience. Break down complicated concepts and strike a friendly and converstional tone.
- Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. 
- Keep the answer concise. Do not include any preamble.
- If a document form the context is irrelevant to the answer, you may ignore it.
- If you don't know the answer, just say it, don't try to make up an answer.
- If the question is not clear or meaningful or unrelated to the context, then ask the user to rephrase the question.

Answer:"""


def generate_with_resources(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("\n---GENERATE WITH RESOURCES---\n")

    # Using the contextualized (reformulated) question
    question = state["contextualized_question"]
    documents = state["documents"]
    loop_step = state.get("loop_step", 0)

    # RAG generation
    docs_txt = format_docs(documents)
    rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
    print(rag_prompt_formatted)

    response = llm.invoke([HumanMessage(content=rag_prompt_formatted)])

    return {"messages": response, "loop_step": loop_step + 1}




#####################################################################################################################
#
#   GENERATE WITHOUT RESOURCES
#
#####################################################################################################################


direct_prompt = """You are a helpful and informative assistent for question-answering tasks.
You have been designed to support transformative climate adaptation by managing knowledge from different European projects, supporting decision-making and planning, 
bridging accessibility gaps between regional and local actors, operationalizing nature-based solutions (nbs), enabling quick information access, and guiding adaptation strategies.

Do not answer to question not related to your domain.
"""


def generate_without_resources(state: GraphState):

    print("\n---GENERATE WITHOUT RESOURCES---\n")

    # question = state["user_question"]
    # print("Question: ", question)
   
    # Get summary if it exists
    summary = state.get("summary", "")

    # If there is summary, add it
    if summary:
        summary_message = f"Summary of conversation earlier: {summary}"
        messages = [SystemMessage(content=direct_prompt)] + [SystemMessage(content=summary_message)] + state["messages"]
    else:
        messages = [SystemMessage(content=direct_prompt)] + state["messages"]
    
    # print(messages)

    response = llm.invoke(messages)
    return {"messages": response}




#####################################################################################################################
#
#   SUMMARIZE
#
#####################################################################################################################


def summarize_conversation(state: GraphState):

    print("\n---SUMMARIZING CONVERSATION---\n")
    
    # First, we get any existing summary
    summary = state.get("summary", "")

    # Create our summarization prompt 
    if summary:
        
        # A summary already exists
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above. Do not use any preamble."
        )
        
    else:
        summary_message = "Create a summary of the conversation above. Do not use any preamble."

    # Add prompt to our history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = llm.invoke(messages)
    
    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}




#####################################################################################################################
#
#   SUMMARIZE OR END
#
#####################################################################################################################


# Determine whether to end or summarize the conversation
def summarize_or_end(state: GraphState):
    
    """Return the next node to execute."""
    
    messages = state["messages"]
    
    # If there are more than four messages, then we summarize the conversation
    if len(messages) > 4:
        return "summarize_conversation"
    
    # Otherwise we can just end
    return END




#####################################################################################################################
#
#   REWRITE QUERY
#
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

def rewrite_query(state):
    pass




#####################################################################################################################
#
#   GRADE DOCUMENTS
#
#####################################################################################################################


# Doc grader instructions
doc_grader_instructions = """You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."""


# Grader prompt
doc_grader_prompt = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}. 
Assess whether the document contains at least some information that is relevant to the question.
Return JSON with single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains at least some information that is relevant to the question."""


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    Remove documents that are not relevant

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents
    """

    print("\n---CHECK DOCUMENT RELEVANCE TO QUESTION---\n")
    question = state["contextualized_question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []

    for d in documents:
        doc_grader_prompt_formatted = doc_grader_prompt.format(
            document=d.page_content, question=question
        )
        result = llm_json_mode.invoke(
            [SystemMessage(content=doc_grader_instructions)]
            + [HumanMessage(content=doc_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)["binary_score"]

        if grade.lower() == "yes":
            print("\n---GRADE: DOCUMENT RELEVANT---\n")
            filtered_docs.append(d)
        else:
            print("\n---GRADE: DOCUMENT NOT RELEVANT---\n")
            # We do not include the document in filtered_docs
            
    return {"documents": filtered_docs}




#####################################################################################################################
#
#   ROUTE QUESTION
#
#####################################################################################################################


router_instructions = """You are an expert at routing a user question to the right source:
- 'vectorstore': source that retrieve documents from a vectorstore.
- 'direct': source that generate a direct answer.

Use the vectorstore for questions on those topics:
- climate change, climate risk, climate hazards
- transformative climate adaptation, Adaptation Pathways
- nature-based solutions (nbs)
- European projects, decision-making, stakeholders and regional-to-local actors

Return JSON with single key, datasource, that is 'vectorstore' or 'direct' depending on the question."""


def route_question(state):
    """
    Route question to vectorstore (RAG system) or direct answer

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    question = state["contextualized_question"]

    print("\n---ROUTE QUESTION---\n")
    route_question = llm_json_mode.invoke([SystemMessage(content=router_instructions)] + [HumanMessage(content=question)])
    source = json.loads(route_question.content)["datasource"]

    if source == "direct":
        print("\n---ROUTE QUESTION TO DIRECT ANSWER---\n")
        return "direct"
    elif source == "vectorstore":
        print("\n---ROUTE QUESTION TO VECTORSTORE (RAG)---\n")
        return "vectorstore"



#####################################################################################################################
#
#   GRADE DOCUMENTS
#
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
#
#   FFF
#
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
#
#   FFF
#
#####################################################################################################################

