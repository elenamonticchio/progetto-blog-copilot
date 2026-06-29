"""
Web search via Tavily: recupera informazioni aggiornate da Internet
(notizie, uscite, eventi, recensioni recenti).
"""
import os
from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """
    Cerca informazioni aggiornate sul web (notizie, uscite di film/serie,
    eventi, recensioni). Restituisce i risultati piu' rilevanti formattati
    con titolo, URL e estratto.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Errore: TAVILY_API_KEY non configurata in .env"

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        res = client.search(query=query, max_results=4, search_depth="basic")
    except Exception as e:
        return f"Errore Tavily: {e}"

    results = res.get("results", [])
    if not results:
        return f"Nessun risultato web per '{query}'."

    blocks = []
    for r in results:
        blocks.append(
            f"[{r.get('title', 'senza titolo')}]\n"
            f"url: {r.get('url', '')}\n"
            f"{r.get('content', '')[:500]}"
        )
    return "\n\n---\n\n".join(blocks)
