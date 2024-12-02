import operator
from langgraph.graph import MessagesState
from typing import List, Annotated


class GraphState(MessagesState):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """

    # default "messages" key from MessageState class

    user_question: str # User question
    contextualized_question: str # Contextualized question
    documents: List[str]  # List of retrieved documents
    summary: str  # Summary of older messages
    loop_step: Annotated[int, operator.add] # Number of answer generation

    '''
    generation: str  # LLM generation
    max_retries: int  # Max number of retries for answer generation
    answers: int  # Number of answers generated
    '''