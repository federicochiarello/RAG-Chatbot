import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama



local_llm = "llama3.1"
llm_json_mode = ChatOllama(model=local_llm, temperature=0.0, format="json")



'''
 - 'tabular': source that generate a query over tabular documents. The tabular documents contains informations on datasets of nature-based solutions proposals in different countries.
  - Route to 'tabular': if the question can be answered by retrieving data from a tabular file.
'''


# Prompt
router_instructions = """You are an expert at routing user questions to the right source.

Those are the available sources:
 - 'vectorstore': source that retrieve documents from a vectorstore. The vectorstore contains documents related to climate change adaptation strategies and nature-based solutions.
 - 'unrelated': source that answer directly without retrieving documents (the question is unrelated to the documents).

Route the user to the right source:
 - Route to 'vectorstore': if the question is on topics related to climate change and nature-based solutions.
 - Route to 'unrelated': if the question can be answered directly without retrieving documents.

Route to 'unrelated' only if you are sure that the question does not need documents to be answered, otherwise if you are not sure, than route it to 'vectorstore'.

Return JSON with single key, datasource, that is 'vectorstore' or 'unrelated' depending on the question."""


# Test router
test_vector_store_1 = llm_json_mode.invoke(
    [SystemMessage(content=router_instructions)]
    + [HumanMessage(content="Which NBS solutions do you know for storm surges and high tides?")]
)
test_vector_store_2 = llm_json_mode.invoke(
    [SystemMessage(content=router_instructions)]
    + [HumanMessage(content="Can you summarize the use of bio-retention areas by positive and negative points?")]
)

test_tabular_1 = llm_json_mode.invoke(
    [SystemMessage(content=router_instructions)]
    + [HumanMessage(content="Which NBS solutions can be used in Austria for high tides?")]
)
test_tabular_2 = llm_json_mode.invoke(
    [SystemMessage(content=router_instructions)]
    + [HumanMessage(content="Give me some proposals of solutions of flooding in Spain?")]
)

test_unrelated_1 = llm_json_mode.invoke(
    [SystemMessage(content=router_instructions)]
    + [HumanMessage(content="What are the models released for llama3.2?")]
)
test_unrelated_2 = llm_json_mode.invoke(
    [SystemMessage(content=router_instructions)]
    + [HumanMessage(content="Will it rain tomorrow in Venice?")]
)


print(
    json.loads(test_vector_store_1.content),
    json.loads(test_vector_store_2.content),
    json.loads(test_tabular_1.content),
    json.loads(test_tabular_2.content),
    json.loads(test_unrelated_1.content),
    json.loads(test_unrelated_2.content),
)