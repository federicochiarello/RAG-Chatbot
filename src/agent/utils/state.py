import operator
from langgraph.graph import MessagesState
from typing import List, Annotated


class GraphState(MessagesState):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """

    # default "messages" key from MessageState class

    user_question: str              # Original user question
    contextualized_question: str    # Contextualized (reformulated) question
    documents: List[str]            # List of retrieved documents
    summary: str                    # Summary of older messages
    follow_up_questions: List[str]  # Generated follow-up questions

    loop_step_generate: Annotated[int, operator.add]    # Number of answer generation calls
    loop_step_retrieval: Annotated[int, operator.add]   # Number of retrieval calls
    max_retries_generate: int                           # Max number of retries for answer generation
    max_retries_retrieval: int                          # Max number of retries for document retrieval

    re_write_query: bool            # If True trigger rewrite node, otherwise continue on the graph
