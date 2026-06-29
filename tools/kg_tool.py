"""
Tool di lettura del Knowledge Graph per l'agente ReAct.
"""
from langchain_core.tools import tool

from kg.kg_manager import KnowledgeGraphManager


@tool
def kg_get_topic_context(topic: str) -> str:
    """
    Restituisce il contesto del Knowledge Graph per un topic:
    post correlati gia' pubblicati e claim sostenuti in passato.
 
    """
    return KnowledgeGraphManager().get_topic_context(topic)
