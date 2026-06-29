"""
RAG retrieval tool: wrappa il K-RAG come @tool selezionabile dall'agente ReAct.
"""
from langchain_core.tools import tool

from rag.retriever import krag_retrieve, format_citations


@tool
def rag_search(topic: str) -> str:
    """
    Cerca passaggi rilevanti nel corpus LOCALE del blog (documenti di
    riferimento su film e serie TV) usando il K-RAG: la query viene espansa
    con i topic correlati del Knowledge Graph, poi i documenti vengono
    filtrati per rilevanza. Da preferire alla web_search per informazioni
    di riferimento stabili (registi, generi, piattaforme).
    """
    docs = krag_retrieve(topic, k=4, grade=True)
    if not docs:
        return "Nessun documento rilevante nel corpus locale."

    citations = format_citations(docs)
    body = "\n\n---\n\n".join(
        f"[fonte: {d.metadata.get('source', '?')}]\n{d.page_content}"
        for d in docs
    )
    sources_line = "\n\nFonti: " + ", ".join(c["source"] for c in citations)
    return body + sources_line
