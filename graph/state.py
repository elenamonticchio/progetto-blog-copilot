"""Stato condiviso del grafo LangGraph."""
from typing import Annotated, Sequence, TypedDict
import operator

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

MAX_ITERATIONS = 3
MAX_GROUNDING_RETRIES = 1   # quante volte il loop di self-correction può ritentare il draft


class PlannedPost(TypedDict):
    topic: str
    post_type: str
    justification: str
    priority: int
    publishing_date: str  


class AgentState(TypedDict):
    # ------------------------------------------------------------------ #
    # CONVERSAZIONE                                                        #
    # ------------------------------------------------------------------ #
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_input: str

    # ------------------------------------------------------------------ #
    # PIANIFICAZIONE                                                       #
    # ------------------------------------------------------------------ #
    suggested_topics: list[dict]
    editorial_plan: list[PlannedPost]
    current_post_index: int
    post_type: str
    current_topic: str

    # Valori: "movie" | "tv_series" | "person" | "concept" | "event"
    topic_kind: str

    # ------------------------------------------------------------------ #
    # RETRIEVAL E KNOWLEDGE                                                #
    # ------------------------------------------------------------------ #
    kg_context: str
    retrieved_docs: list[dict]
    tool_outputs: Annotated[list[dict], operator.add]
    reasoning_trace: Annotated[list[dict], operator.add]

    # ------------------------------------------------------------------ #
    # GENERAZIONE BOZZA                                                    #
    # ------------------------------------------------------------------ #
    current_title: str
    current_draft: str
    citations: list[dict]
    key_claims: list[str]

    # ------------------------------------------------------------------ #
    # HUMAN-IN-THE-LOOP                                                    #
    # ------------------------------------------------------------------ #
    user_status: str
    user_feedback: str
    approved: bool
    iteration_count: int

    # ------------------------------------------------------------------ #
    # SELF-CORRECTION (verify_grounding)                                   #
    # ------------------------------------------------------------------ #
    grounding_feedback: list[dict]  
    grounding_retries: int          
    grounding_passed: bool          

    # ------------------------------------------------------------------ #
    # CADENZA / SEQUENZA                                                   #
    # ------------------------------------------------------------------ #
    posts_per_session: int   
    days_between_posts: int  
    tool_outputs_offset: int  
