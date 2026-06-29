"""
Costruzione e cablaggio del grafo LangGraph.
La logica dei nodi vive in agents/nodes.py — qui c'è solo il wiring.
"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from graph.state import AgentState
from agents.nodes import (
    suggest_topics,
    planner,
    research,
    verify_and_select,
    draft,
    verify_grounding,
    human_review,
    update_kg,
    advance_post,
    route_after_review,
    route_after_kg,
    route_after_grounding,
)


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("suggest_topics", suggest_topics)
    builder.add_node("planner", planner)
    builder.add_node("research", research)
    builder.add_node("verify_and_select", verify_and_select)
    builder.add_node("draft", draft)
    builder.add_node("verify_grounding", verify_grounding)
    builder.add_node("human_review", human_review)
    builder.add_node("update_kg", update_kg)
    builder.add_node("advance_post", advance_post)

    builder.add_edge(START, "suggest_topics")
    builder.add_edge("suggest_topics", "planner")
    builder.add_edge("planner", "research")
    builder.add_edge("research", "verify_and_select")
    builder.add_edge("verify_and_select", "draft")
    builder.add_edge("draft", "verify_grounding")

    builder.add_conditional_edges(
        "verify_grounding",
        route_after_grounding,
        {"human_review": "human_review", "draft": "draft"},
    )
    builder.add_conditional_edges(
        "human_review",
        route_after_review,
        {
            "update_kg": "update_kg",
            "draft": "draft",     
            "end": END,
        },
    )
    builder.add_conditional_edges(
        "update_kg",
        route_after_kg,
        {"advance_post": "advance_post", "end": END},
    )
    builder.add_edge("advance_post", "research")

    # MemorySaver abilita interrupt/resume e persistenza dello stato tra invocazioni.
    return builder.compile(checkpointer=MemorySaver())


graph = build_graph()
