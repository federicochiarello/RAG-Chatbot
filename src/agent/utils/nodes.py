from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage
from langgraph.graph import END
from langchain_ollama import ChatOllama

from .state import GraphState
from .db.chroma_db_manager import ChromaDBManager
from .get_embedding_function import get_embedding_function

import os
import json
from pathlib import Path
from dotenv import load_dotenv


# https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag_local/#local-models


#########################################################################################################

import os
import logging
import time
from datetime import datetime


# Setup logging configuration
log_dir = os.path.normpath('./logs')
os.makedirs(log_dir, exist_ok=True)

start_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_filename = f'execution_log_{start_time}.log'
log_file_path = os.path.join(log_dir, log_filename)

logging.basicConfig(
    filename=log_file_path,             # Log file name
    level=logging.INFO,                 # Log level
    format='%(asctime)s - %(levelname)s - %(message)s', # Log format
    datefmt='%Y-%m-%d %H:%M:%S'         # Date and time format
)

def log_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)
        end_time = time.time()    # Record the end time
        elapsed_time = end_time - start_time  # Calculate the elapsed time
        logging.info(f"Execution of {func.name} took {elapsed_time:.4f} seconds")
        return result
    return wrapper

# @log_execution_time
# def example_fun():
#    pass

# logging.info("info")

line_separator = "\n--------------------------------------------------------------------------------------------------------\n"

#########################################################################################################







env_path = os.path.join(os.path.dirname(__file__), '../.env')
load_dotenv(dotenv_path=env_path)

CHROMA_PATH = Path(os.getenv("CHROMA_PATH"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME")



chroma_db_manager = ChromaDBManager(
    persist_directory = CHROMA_PATH,
    collection_name = COLLECTION_NAME,
    embedding_function = get_embedding_function()
)
chroma_db_manager.connect()
db = chroma_db_manager.get_db()


K = 6
retriever = db.as_retriever(search_kwargs={"k": K})

logging.info(f"Retriever\n\n - Retriever config:\t{retriever}\n")


local_llm = "llama3.1"
llm = ChatOllama(model=local_llm)
llm_json_mode = ChatOllama(model=local_llm, temperature=0, format="json")

logging.info(f"LLM Model\n\n - LLM name:\t\t{local_llm}\n - LLM config:\t\t{llm}\n - LLM JSON config:\t{llm_json_mode}\n")



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

ATTENTION: if the original user message is not meaningful (random words or letters), return it as it is without any addition."""


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

    langchain_messages = state["messages"]      # list of Langchain Messages with metadata           
    messages = format_messages(langchain_messages)
    summary = state.get("summary", "")
    question = langchain_messages[-1].content

    # if it is the first message, skip contextualization
    # set both questions to original value
    # if len(langchain_messages) == 1:
    #    logging.info("CONTEXTUALIZE: skip for first message")
    #    return {"user_question": question, "contextualized_question": question}
    
    contextualize_prompt_formatted = contextualize_prompt.format(summary=summary, messages=messages, question=question)
    print(contextualize_prompt_formatted)
    contextualized_question = llm.invoke([HumanMessage(content=contextualize_prompt_formatted)])

    logging.info(f"""CONTEXTUALIZE\n\nPROMPT:{line_separator}{contextualize_prompt_formatted}{line_separator}
CONTEXTUALIZED QUESTION:{line_separator}{contextualized_question.content}{line_separator}""")

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
    loop_step_retrieval = state.get("loop_step_retrieval", 0)
    print("Question used for retrieval: ", question)
    documents = retriever.invoke(question)

    print("\nRetrieved documents:")
    docs_for_log = ""
    for index, document in enumerate(documents):
        docs_for_log += f"\nDocument {index+1}:\n\n{document}\n"
        print(f"\nDocument {index+1}:\n\n", document, "\n")

    logging.info(f"""RETRIEVE\n\nQUESTION USED FOR RETRIEVAL:{line_separator}{question}{line_separator}
RETRIEVED DOCUMENTS:{line_separator}{docs_for_log}{line_separator}""")

    return {"documents": documents, "loop_step_retrieval": loop_step_retrieval + 1}




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
    loop_step_generate = state.get("loop_step_generate", 0)

    # RAG generation
    docs_txt = format_docs(documents)
    rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
    print(rag_prompt_formatted)

    response = llm.invoke([HumanMessage(content=rag_prompt_formatted)])

    logging.info(f"GENERATE WITH RESOURCES\n\nPROMPT:{line_separator}{rag_prompt_formatted}{line_separator}\nANSWER:{line_separator}{response.content}{line_separator}")

    return {"messages": response, "loop_step_generate": loop_step_generate + 1}




#####################################################################################################################
#
#   GENERATE WITHOUT RESOURCES
#
#####################################################################################################################


direct_prompt = """You are an AI language model assistant for question-answering.
You have been designed to support transformative climate adaptation by managing knowledge from different European projects, supporting decision-making and planning, 
bridging accessibility gaps between regional and local actors, operationalizing nature-based solutions (nbs), enabling quick information access, and guiding adaptation strategies.

Do not answer to question not related to your domain."""


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

    response = llm.invoke(messages)

    logging.info(f"GENERATE WITHOUT RESOURCES\n\nPROMPT:{line_separator}{direct_prompt}{line_separator}\nANSWER:{line_separator}{response.content}{line_separator}")

    return {"messages": response}




#####################################################################################################################
#
#   GENERATE FOLLOW-UP QUESTIONS
#
#####################################################################################################################


follow_up_questions_prompt = """You are an AI language model assistant. Your task is to write 2 short follow-up questions,
given a question posed by the user and an answer generated by the model.

You have also access to previous messages and a summary of the chat history:

summary: {summary}

messages: [

{messages}]

Focus particularly on the last question and answer:

question: {question}

answer: {answer}

Keep in mind that you have been designed to support transformative climate adaptation. The generated questions must not stray too far from that domain.
Write 2 short questions. No preamble.
Keep the questions short. The questions should be concised.
Return JSON with keys 1, 2 and as value the corresponding generated question."""


def generate_follow_up_questions(state):
    """
    Generate follow-up questions.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updated follow_up_questions field
    """
    print("\n---GENERATE FOLLOW-UP QUESTIONS---\n")

    langchain_messages = state["messages"]      # list of Langchain Messages with metadata

    question = langchain_messages[-2].content
    answer = langchain_messages[-1].content
    messages = format_messages(langchain_messages)
    summary = state.get("summary", "")
    
    follow_up_questions_prompt_formatted = follow_up_questions_prompt.format(summary=summary, messages=messages, question=question, answer=answer)
    print(follow_up_questions_prompt_formatted)
    raw_answer = llm_json_mode.invoke([HumanMessage(content=follow_up_questions_prompt_formatted)])

    try:
        follow_up_questions = [json.loads(raw_answer.content)["1"], json.loads(raw_answer.content)["2"]]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logging.error(f"Error parsing LLM response: {e}")
        print(f"Error parsing LLM response: {e}")
        follow_up_questions = ["", "", ""]

    questions_for_log = ""
    print("Follow-up questions:")
    for i, q in enumerate(follow_up_questions):
        print(i+1, ": ", q)
        questions_for_log += f" - {i+1}: {q}\n"

    logging.info(f"GENERATE FOLLOW-UP QUESTIONS\n\nPROMPT:{line_separator}{follow_up_questions_prompt_formatted}{line_separator}\nFOLLOW-UP QUESTIONS:{line_separator}{questions_for_log}{line_separator}")

    return {"follow_up_questions": follow_up_questions}




#####################################################################################################################
#
#   SUMMARIZE
#
#####################################################################################################################


def summarize_conversation(state: GraphState):

    print("\n---SUMMARIZING CONVERSATION---\n")
    
    summary = state.get("summary", "")

    if summary:
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above."
        )
    else:
        summary_message = "Create a summary of the conversation above."

    system_message = """You are an AI language model assistant.
    Your task is to generate or extend a summary of a conversation between a user and a chatbot.

    The summary must contain the salient information derived from the conversation.
    Prioritize information about the user.
    Keep track of the topics covered.
    Do not use any preamble.
    """

    # Add prompt to our history
    messages = [SystemMessage(content=system_message)] + state["messages"] + [HumanMessage(content=summary_message)]
    response = llm.invoke(messages)
    
    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]

    logging.info(f"""SUMMARIZING CONVERSATION\n\nPROMPT:{line_separator}{system_message}\n{summary_message}{line_separator}
SUMMARY: {line_separator}{response.content}{line_separator}\n""")

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
        logging.info(f"DECIDE IF SUMMARIZE OR END:\tsummarize_conversation")
        return "summarize_conversation"
    
    # Otherwise we can just end
    logging.info(f"DECIDE IF SUMMARIZE OR END:\tgenerate_follow_up_questions")
    return END




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


MINIMUM_NUMEBER_OF_RELEVANT_DOCUMENTS = 2


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
    docs_for_log = ""

    for d in documents:
        doc_grader_prompt_formatted = doc_grader_prompt.format(
            document=d.page_content, question=question
        )
        result = llm_json_mode.invoke(
            [SystemMessage(content=doc_grader_instructions)]
            + [HumanMessage(content=doc_grader_prompt_formatted)]
        )

        try:
            grade = json.loads(result.content)["binary_score"]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logging.error(f"Error parsing LLM response: {e}")
            print(f"Error parsing LLM response: {e}")
            grade = "yes"
        

        if grade.lower() == "yes":
            docs_for_log += f"GRADE: DOCUMENT RELEVANT ({grade.lower()})\n\n{d}\n\n"
            print("\n---GRADE: DOCUMENT RELEVANT---\n")
            print("\n", d, "\n")
            filtered_docs.append(d)
        else:
            docs_for_log += f"GRADE: DOCUMENT NOT RELEVANT ({grade.lower()})\n\n{d}\n\n"
            print("\n---GRADE: DOCUMENT NOT RELEVANT---\n")
            print("\n", d, "\n")
            # We do not include the document in filtered_docs

    # If not enought retrieved docs are relevant, re-write the query and perform a new retrieval
    re_write_query = (len(filtered_docs) < MINIMUM_NUMEBER_OF_RELEVANT_DOCUMENTS)
    print(f"\n---RELEVANT DOCUMENTS: {len(filtered_docs)}. MINIMUM NUMBER: {MINIMUM_NUMEBER_OF_RELEVANT_DOCUMENTS}---\n")
    if re_write_query: 
        print("\n---NOT ENOUGHT. RE-WRITE QUERY---\n")
    else:
        print("\n---ENOUGHT RELEVANT DOCUMENTS---\n")

    logging.info(f"""GRADING DOCUMENTS\n\n{line_separator}{docs_for_log}{line_separator}\nRELEVANT DOCUMENTS: {len(filtered_docs)}\nMINIMUM: {MINIMUM_NUMEBER_OF_RELEVANT_DOCUMENTS}\n""")

    return {"documents": filtered_docs, "re_write_query": re_write_query}




#####################################################################################################################
#
#   DECIDE WHETHER GENERATE OR RE-WRITE QUERY
#
#####################################################################################################################


def generate_or_rewrite_query(state):
    """
    Determines whether to generate an answer, or rewrite the query and perform a new retrival step

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("\n---ASSESS GRADED DOCUMENTS---\n")

    loop_step_retrieve = state.get("loop_step_retrieval")
    max_retries_retrieval = state.get("max_retries_retrieval", 3)

    re_write_query = state["re_write_query"]

    if re_write_query & (loop_step_retrieve < max_retries_retrieval):
        print("\n---DECISION: NOT ENOUGHT DOCUMENTS ARE RELEVANT TO QUESTION. RE-WRITE QUERY---\n")
        logging.info(f"DECIDE IF GENERATE OR RE-WRITE QUERY:\tre_write_query")
        return "re_write_query"
    else:
        print("\n---DECISION: GENERATE---\n")
        logging.info(f"DECIDE IF GENERATE OR RE-WRITE QUERY:\tgenerate_with_resources")
        return "generate_with_resources"




#####################################################################################################################
#
#   RE-WRITE QUERY
#
#####################################################################################################################


re_write_prompt = """You are an AI language model assistant.
Re-write an input question into an improved version that is optimized for vectorstore retrieval. 

- Output only the final improved question
- Do not include any preamble
- Do not motivate or comment your decisions

Initial question: {question}
"""


def re_write_query(state):
    """
    Re-write the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updated contextualized_question field
    """
    print("\n---RE-WRITE QUERY---\n")

    question = state["contextualized_question"]
    re_write_prompt_formatted = re_write_prompt.format(question=question)
    print(re_write_prompt_formatted)
    reformulated_question = llm.invoke([HumanMessage(content=re_write_prompt_formatted)])
    print("Re-written question: ", reformulated_question.content)

    logging.info(f"RE-WRITE QUERY\n\nPROMPT:{line_separator}{re_write_prompt_formatted}{line_separator}\nREFORMULATED QUESTION:\t{line_separator}{reformulated_question.content}{line_separator}")

    # set re_write_query back to False
    return {"contextualized_question": reformulated_question.content, "re_write_query": False}




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

    # Attempt to parse the JSON response
    try:
        source = json.loads(route_question.content)["datasource"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logging.error(f"Error parsing LLM response: {e}")
        print(f"Error parsing LLM response: {e}")
        source = "direct"

    logging.info(f"ROUTING QUESTION:\t{source}")

    if source == "direct":
        print("\n---ROUTE QUESTION TO DIRECT ANSWER---\n")
        return "direct"
    elif source == "vectorstore":
        print("\n---ROUTE QUESTION TO VECTORSTORE (RAG)---\n")
        return "vectorstore"




#####################################################################################################################
#
#   HALLUCINATION GRADER
#
#####################################################################################################################


MAX_GEN = 3


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


"""You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""


def check_hallucinations(state):
    """
    Determines whether the answer is grounded in the document and does not contain hallucinations

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("\n---CHECK HALLUCINATIONS---\n")

    messages = state["messages"]      # list of Langchain Messages with metadata
    answer = messages[-1].content
    documents = state["documents"]
    max_retries_generate = state.get("max_retries_generate", 3)

    if max_retries_generate >= MAX_GEN:
        print("\n---DECISION: MAX RETRIES REACHED---\n")
        return "max retries"

    hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
        documents=format_docs(documents), generation=answer.content
    )
    result = llm_json_mode.invoke(
        [SystemMessage(content=hallucination_grader_instructions)]
        + [HumanMessage(content=hallucination_grader_prompt_formatted)]
    )
    grade = json.loads(result.content)["binary_score"]


    if grade == "yes":
        print("\n---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---\n")
        return "grounded in documents"
    else:
        print("\n---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---\n")
        return "not grounded in documents"




#####################################################################################################################
#
#   ANSWER GRADER
#
#####################################################################################################################


"""You are a grader assessing whether an answer addresses / resolves a question \n 
Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""


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


def check_relevance(state):
    """
    Determines whether the gneration answer the question

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """
    print("\n---GRADE ANSWER vs QUESTION---\n")

    messages = state["messages"]      # list of Langchain Messages with metadata
    question = messages[-2].content
    answer = messages[-1].content

    max_retries_generate = state.get("max_retries_generate", 3)

    if max_retries_generate >= MAX_GEN:
        print("---DECISION: MAX RETRIES REACHED---")
        return "max retries"

    # Check question-answering

    # Test using question and generation from above
    answer_grader_prompt_formatted = answer_grader_prompt.format(
        question=question, generation=answer.content
    )
    result = llm_json_mode.invoke(
        [SystemMessage(content=answer_grader_instructions)]
        + [HumanMessage(content=answer_grader_prompt_formatted)]
    )
    grade = json.loads(result.content)["binary_score"]
    if grade == "yes":
        print("---DECISION: GENERATION ADDRESSES QUESTION---")
        return "addressing question"
    else:
        print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
        return "not addressing question"




#####################################################################################################################
#
#   HALLUCINATION & ANSWER GRADER
#
#####################################################################################################################


def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")

    messages = state["messages"]      # list of Langchain Messages with metadata
    question = messages[-2].content
    answer = messages[-1].content
    documents = state["documents"]
    max_retries = 3
    #max_retries = state.get("max_retries_generate", 3)  # Default to 3 if not provided


    hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
        documents=format_docs(documents), generation=answer
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
            question=question, generation=answer
        )
        result = llm_json_mode.invoke(
            [SystemMessage(content=answer_grader_instructions)]
            + [HumanMessage(content=answer_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)["binary_score"]
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        elif state["loop_step_retrieval"] <= max_retries:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
        else:
            print("---DECISION: MAX RETRIES REACHED---")
            return "max retries"
    elif state["loop_step_generate"] <= max_retries:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
    else:
        print("---DECISION: MAX RETRIES REACHED---")
        return "max retries"