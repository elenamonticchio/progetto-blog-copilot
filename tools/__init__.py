"""Registro centrale dei tool del progetto."""
from tools.search_tool import web_search
from tools.rag_tool import rag_search
from tools.tmdb_tool import tmdb_fact_check
from tools.tvmaze_tool import tvmaze_show_info
from tools.events_tool import find_local_events
from tools.grounding_tool import verify_claim
from tools.kg_tool import kg_get_topic_context

ALL_TOOLS = [
    web_search,
    rag_search,
    kg_get_topic_context,
    tmdb_fact_check,
    tvmaze_show_info,
    find_local_events,
    verify_claim,
]

__all__ = [
    "ALL_TOOLS",
    "web_search",
    "rag_search",
    "kg_get_topic_context",
    "tmdb_fact_check",
    "tvmaze_show_info",
    "find_local_events",
    "verify_claim",
]
