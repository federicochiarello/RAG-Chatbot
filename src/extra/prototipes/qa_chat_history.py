from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function
from db.chroma_db_menager import ChromaDBManager


from langchain_chroma import Chroma


# retriever = vectorstore.as_retriever()



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
# vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Retrieve and generate using the relevant snippets of the blog.
retriever = db.as_retriever()



# 2. Incorporate the retriever into a question-answering chain.
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

model = OllamaLLM(model="llama3.1", temperature=0.7)

question_answer_chain = create_stuff_documents_chain(model, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

#response = rag_chain.invoke({"input": "What is a Nature-based solution?"})
#print(response["answer"])



##############################################################################


from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    model, retriever, contextualize_q_prompt
)





from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)






from langchain_core.messages import AIMessage, HumanMessage

chat_history = []

question = "What is a nature-based solution?"
ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})

print(ai_msg_1["answer"])

chat_history.extend(
    [
        HumanMessage(content=question),
        AIMessage(content=ai_msg_1["answer"]),
    ]
)

second_question = "How could it be used in a urban environment?"
ai_msg_2 = rag_chain.invoke({"input": second_question, "chat_history": chat_history})

print(ai_msg_2["answer"])