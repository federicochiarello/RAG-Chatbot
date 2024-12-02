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

    grade_documents,
    rewrite_query,

    route_question,
    decide_to_generate,
    grade_generation_v_documents_and_question,
)



'''
# Define the nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("contextualize", contextualize)
#workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate_from_resources", generate_from_resources)
workflow.add_node("generate_without_resources", generate_without_resources)
workflow.add_node("summarize_conversation", summarize_conversation)
#workflow.add_node("rewrite_query", rewrite_query)


# Build graph
workflow.add_edge(START, "contextualize")
workflow.add_conditional_edges(
    "contextualize", 
    route_question,
    {
        "vectorstore": "retrieve",
        "unrelated": "generate_without_resources",
    },
)
workflow.add_edge("retrieve", "generate_from_resources")
workflow.add_edge("generate_from_resources", END)
workflow.add_conditional_edges(
    "generate_from_resources", 
    need_summary,
    {
        "summarize": "summarize_conversation",
    },
)
workflow.add_edge("generate_without_resources", END)
workflow.add_conditional_edges(
    "generate_without_resources", 
    need_summary,
    {
        "summarize": "summarize_conversation",
    },
)
'''





# Define the graph
workflow = StateGraph(GraphState)

workflow.add_node("contextualize", contextualize)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate_with_resources", generate_with_resources)
workflow.add_node("generate_without_resources", generate_without_resources)
workflow.add_node("summarize_conversation", summarize_conversation)



# Set the entrypoint
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
workflow.add_conditional_edges(
    "generate_with_resources", 
    summarize_or_end,
    {
        "summarize_conversation": "summarize_conversation",
        END: END,
    }
)
workflow.add_conditional_edges(
    "generate_without_resources", 
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






















import argparse

def main():
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    '''


    from langchain_core.messages import HumanMessage, AIMessage


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


    input_message = HumanMessage(content="Is my area subject to flooding?")
    output = graph.invoke({"messages": [input_message]}, config) 
    for m in output['messages'][-2:]:
        m.pretty_print()


    input_message = HumanMessage(content="Tell me the rules of tennis")
    output = graph.invoke({"messages": [input_message]}, config) 
    for m in output['messages'][-2:]:
        m.pretty_print()




    






    '''

    input_message = HumanMessage(content="What is my name?")
    output = graph.invoke({"messages": [input_message]}, config) 
    for m in output['messages'][-2:]:
        m.pretty_print()

    print("\n\nSUMMARY:\n\n")
    
    print(graph.get_state(config).values.get("summary",""))

    input_message = HumanMessage(content="i like boats!")
    output = graph.invoke({"messages": [input_message]}, config) 
    for m in output['messages'][-2:]:
        m.pretty_print()

        

        


    from pprint import pprint

    # Run
    inputs = {"question": query_text}
    for output in graph.stream(inputs):
        for key, value in output.items():
            # Node
            pprint(f"Node '{key}':")
            # Optional: print full state at each node
            #pprint(value, indent=2, width=80, depth=None)
        pprint("\n---\n")

    # Final generation
    pprint(value["generation"])
    '''

    
    from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

    img = graph.get_graph().draw_mermaid_png(
        draw_method=MermaidDrawMethod.API,
    )
    output_file = "graph.png"
    with open(output_file, 'wb') as f:
        f.write(img)
    



if __name__ == "__main__":
    main()









'''
workflow.set_conditional_entry_point(
    route_question,
    {
        "unrelated": "generate",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "websearch",
        "max retries": END,
    },
)

'''