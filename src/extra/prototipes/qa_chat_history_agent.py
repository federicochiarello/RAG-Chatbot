from langchain_core.messages import HumanMessage
from langchain.tools.retriever import create_retriever_tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent


memory = MemorySaver()


#######################################################

import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_ollama import ChatOllama
from get_embedding_function import get_embedding_function
from db.chroma_db_menager import ChromaDBManager

load_dotenv()
CHROMA_PATH = Path(os.getenv("CHROMA_PATH"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

chroma_db_manager = ChromaDBManager(
        persist_directory = CHROMA_PATH,
        collection_name = COLLECTION_NAME,
        embedding_function = get_embedding_function()
    )
chroma_db_manager.connect()

db = chroma_db_manager.get_db()

# Retrieve and generate using the relevant snippets of the blog.
retriever = db.as_retriever()

llm = ChatOllama(model="llama3.1", temperature=0.7)

#######################################################




config = {"configurable": {"thread_id": "abc123"}}



### Build retriever tool ###
tool = create_retriever_tool(
    retriever,
    "resources_retriever",
    "Searches and returns excerpts from the knowledge base.",
)
tools = [tool]


agent_executor = create_react_agent(llm, tools, checkpointer=memory) # debug = True)



query = "Hi, my name is Federico"

for event in agent_executor.stream(
    {"messages": [HumanMessage(content=query)]},
    config=config,
    stream_mode="values",
):
    event["messages"][-1].pretty_print()


query = "What is a nbs?"

for event in agent_executor.stream(
    {"messages": [HumanMessage(content=query)]},
    config=config,
    stream_mode="values",
):
    event["messages"][-1].pretty_print()


query = "How can we use then to solve flooding problems?"

for event in agent_executor.stream(
    {"messages": [HumanMessage(content=query)]},
    config=config,
    stream_mode="values",
):
    event["messages"][-1].pretty_print()
