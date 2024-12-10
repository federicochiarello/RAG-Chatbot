import time
import streamlit as st
from langchain_core.messages import HumanMessage
from .agent import build_graph


st.title("💬 Chatbot with RAG")


# Build Graph
if 'graph' not in st.session_state:
    st.session_state.graph = build_graph()

# Create a thread
config = {"configurable": {"thread_id": "1"}}


def response_generator(prompt: str):

    output = st.session_state.graph.invoke({"messages": [HumanMessage(content=prompt)]}, config)
    response = output['messages'][-1].content

    print(response)

    for line in response.splitlines(keepends=True):
        for word in line.split():
            yield word + " "
            time.sleep(0.05)
        if line.endswith("\n"):
            yield "\n"
            time.sleep(0.05)


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = response_generator(prompt)
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
