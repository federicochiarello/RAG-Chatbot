import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from state import GraphState


local_llm = "llama3.1"
llm = ChatOllama(model=local_llm, temperature=0.0)



# Prompt
contextualize_prompt = """You are an expert at contextualizing user questions using previous messages and a summary of the chat history.
If needed rewrite the user question in such a way that it can be understood on its own, without the context provided by previous messages.
If the user question is already clear on its own, than do not modify it.

summary: {summary}

messages: {messages}

question: {question}

Contextualize the question using messages and summary. Return only the final rewritten question.
"""



def contextualize(state):
    """
    Contextualize user query using previous messages as context.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Update the user query using previous messages and chat history as context
    """
    print("---CONTEXTUALIZE---")

    summary = state["summary"]
    messages = state["messages"]
    question = state["question"]

    contextualize_prompt_formatted = contextualize_prompt.format(summary=summary, messages=messages, question=question)

    print(contextualize_prompt_formatted)

    contextualized_question = llm.invoke([HumanMessage(content=contextualize_prompt_formatted)])

    print(contextualized_question)
    return {"question": contextualized_question}







# Test router
test_vector_store_1 = llm.invoke(
    [SystemMessage(content=router_instructions)]
    + [HumanMessage(content="Which NBS solutions do you know for storm surges and high tides?")]
)
test_vector_store_2 = llm.invoke(
    [SystemMessage(content=router_instructions)]
    + [HumanMessage(content="Can you summarize the use of bio-retention areas by positive and negative points?")]
)




print(
    json.loads(test_vector_store_1.content),
    json.loads(test_vector_store_2.content),
)