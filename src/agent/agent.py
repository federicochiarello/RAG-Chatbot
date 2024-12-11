from langchain_core.messages import HumanMessage
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from utils.state import GraphState
from utils.nodes import (
    contextualize,
    retrieve,
    generate_with_resources,
    generate_without_resources,
    summarize_conversation,
    summarize_or_end,
    generate_follow_up_questions,

    grade_documents,
    generate_or_rewrite_query,
    re_write_query,

    route_question,
    check_hallucinations,
    check_relevance,
    grade_generation_v_documents_and_question
)



import os
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()

CHROMA_PATH = Path(os.getenv("CHROMA_PATH"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "local-llama-rag-test1"




def build_graph_small():


    workflow = StateGraph(GraphState)


    # Define nodes
    workflow.add_node("contextualize", contextualize)
    workflow.add_node("retrieve", retrieve)
    #workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate_with_resources", generate_with_resources)
    workflow.add_node("generate_without_resources", generate_without_resources)
    workflow.add_node("summarize_conversation", summarize_conversation)
    workflow.add_node("generate_follow_up_questions", generate_follow_up_questions)
    # forced to use re_write_q
    # ValueError: 're_write_query' is already being used as a state key
    #workflow.add_node("re_write_q", re_write_query)
    #workflow.add_node("check_hallucinations", check_hallucinations)
    #workflow.add_node("check_relevance", check_relevance)
    

    # Add egdes
    workflow.add_edge(START, "contextualize")
    workflow.add_conditional_edges(
        "contextualize",
        route_question,
        {
            "direct": "generate_without_resources",
            "vectorstore": "retrieve",
        },
    )
    workflow.add_edge("retrieve", "generate_with_resources")
    workflow.add_edge("generate_with_resources", "generate_follow_up_questions")
    workflow.add_edge("generate_without_resources", "generate_follow_up_questions")
    workflow.add_conditional_edges(
        "generate_follow_up_questions", 
        summarize_or_end,
        {
            "summarize_conversation": "summarize_conversation",
            END: END,
        }
    )
    workflow.add_edge("summarize_conversation", END)


    # Compile
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)

    return graph





def build_graph():


    workflow = StateGraph(GraphState)


    # Define nodes
    workflow.add_node("contextualize", contextualize)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate_with_resources", generate_with_resources)
    workflow.add_node("generate_without_resources", generate_without_resources)
    workflow.add_node("summarize_conversation", summarize_conversation)
    workflow.add_node("generate_follow_up_questions", generate_follow_up_questions)
    # forced to use re_write_q
    # ValueError: 're_write_query' is already being used as a state key
    workflow.add_node("re_write_q", re_write_query)
    #workflow.add_node("check_hallucinations", check_hallucinations)
    #workflow.add_node("check_relevance", check_relevance)
    

    # Add egdes
    workflow.add_edge(START, "contextualize")
    workflow.add_conditional_edges(
        "contextualize",
        route_question,
        {
            "direct": "generate_without_resources",
            "vectorstore": "retrieve",
        },
    )
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents", 
        generate_or_rewrite_query,
        {
            "generate_with_resources": "generate_with_resources",
            "re_write_query": "re_write_q",
        }
    )
    workflow.add_edge("re_write_q", "retrieve")
    workflow.add_conditional_edges(
        "generate_with_resources",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate_with_resources",
            "useful": "generate_follow_up_questions",
            "not useful": "re_write_q",
            "max retries": "generate_follow_up_questions", # to be implemented
        },
    )
    workflow.add_edge("generate_without_resources", "generate_follow_up_questions")
    workflow.add_conditional_edges(
        "generate_follow_up_questions", 
        summarize_or_end,
        {
            "summarize_conversation": "summarize_conversation",
            END: END,
        }
    )
    workflow.add_edge("summarize_conversation", END)


    # Compile
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)

    return graph




def main():

    # Build Graph
    graph = build_graph_small()


    # Create a thread
    config = {"configurable": {"thread_id": "1"}}


    input_message = HumanMessage(content="Hi my name is Federico. Who are you?")
    output = graph.invoke({"messages": [input_message]}, config) 
    for m in output['messages'][-2:]:
        m.pretty_print()


    input_message = HumanMessage(content="I live in Venice. Can you tell me what nbs stands for in climate change domain?")
    output = graph.invoke({"messages": [input_message]}, config) 
    for m in output['messages'][-2:]:
        m.pretty_print()


    input_message = HumanMessage(content="Can they be used to solve flooding problems?")
    output = graph.invoke({"messages": [input_message]}, config) 
    for m in output['messages'][-2:]:
        m.pretty_print()


    input_message = HumanMessage(content="Can you give me some examples of nbs?")
    output = graph.invoke({"messages": [input_message]}, config) 
    for m in output['messages'][-2:]:
        m.pretty_print()

    """

    input_message = HumanMessage(content="Is my area subject to flooding?")
    output = graph.invoke({"messages": [input_message]}, config) 
    for m in output['messages'][-2:]:
        m.pretty_print()


    input_message = HumanMessage(content="Tell me the rules of tennis")
    output = graph.invoke({"messages": [input_message]}, config) 
    for m in output['messages'][-2:]:
        m.pretty_print()

    """


    # Draw Graph

    img = graph.get_graph().draw_mermaid_png(
        draw_method=MermaidDrawMethod.API,
    )
    output_file = "graph_small.png"
    with open(output_file, 'wb') as f:
        f.write(img)
    


if __name__ == "__main__":
    main()