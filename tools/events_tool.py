"""
Events Finder: cerca eventi cinematografici imminenti via Tavily
e filtra i risultati con un grader LLM per tenere solo articoli
che descrivono eventi concreti nella location richiesta.
"""
import os
from datetime import datetime

from pydantic import BaseModel, Field
from langchain_core.tools import tool

from config.settings import get_llm


class GradeEvent(BaseModel):
    """Valutazione binaria di pertinenza di un articolo a un evento richiesto."""
    binary_score: str = Field(
        description="'yes' se l'articolo descrive davvero un evento del tipo "
                    "richiesto, localizzato nella location richiesta; 'no' altrimenti"
    )
    reason: str = Field(description="Motivazione sintetica, una frase")


GRADE_PROMPT = (
    "Devi decidere se questo articolo descrive davvero un EVENTO di tipo "
    "'{kind}' che si svolge in '{location}'.\n\n"
    "Articolo:\n"
    "Titolo: {title}\n"
    "Contenuto: {content}\n\n"
    "Rispondi 'yes' solo se l'articolo descrive un evento concreto "
    "(festival, anteprima, rassegna, incontro con registi) localizzato "
    "nell'area richiesta.\n\n"
    "Rispondi 'no' se l'articolo:\n"
    "- parla di un evento in un'altra area geografica\n"
    "- è una notizia generica non legata a un evento concreto\n"
    "- menziona la location solo di passaggio o per altri tipi di evento\n"
    "  (es. Biennale d'arte invece che festival del cinema)"
)


def _grade_events(results: list[dict], location: str, kind: str) -> list[tuple[dict, str]]:
    """
    Valuta ciascun risultato con il grader LLM.
    Restituisce solo i promossi, accoppiati alla motivazione del grader.
    """
    grader = get_llm(temperature=0).with_structured_output(GradeEvent)
    promoted: list[tuple[dict, str]] = []

    for r in results:
        prompt = GRADE_PROMPT.format(
            kind=kind,
            location=location,
            title=r.get("title", ""),
            content=(r.get("content") or "")[:3000],
        )

        verdict = grader.invoke(prompt)

        if verdict.binary_score.strip().lower().startswith("yes"):
            promoted.append((r, verdict.reason))

    return promoted


@tool
def find_local_events(location: str = "Roma", kind: str = "festival cinema") -> str:
    """
    Cerca eventi cinematografici e televisivi imminenti (festival, anteprime,
    rassegne, incontri con registi) in una specifica area geografica.
    Restituisce un elenco di eventi con titolo, fonte e descrizione sintetica.

    Args:
        location: città o regione di interesse (es. "Sicilia", "Catania", "Roma")
        kind: tipo di evento (es. "festival cinema", "anteprima film",
              "rassegna cinematografica", "incontro con regista")

    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Errore: TAVILY_API_KEY non configurata in .env"

    year = datetime.now().year
    query = f"prossimi {kind} {location} {year}"

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)

        res = client.search(
            query=query,
            max_results=10,
            search_depth="basic",
            topic="news",
            days=60,
        )
    except Exception as e:
        return f"Errore Tavily: {e}"

    results = res.get("results", [])

    if not results:
        return f"Nessun evento trovato per '{kind}' in '{location}' nell'ultimo periodo."

    promoted = _grade_events(results, location, kind)

    if not promoted:
        return (
            f"Trovati {len(results)} articoli per '{kind}' in '{location}', "
            "ma nessuno descrive eventi realmente pertinenti."
        )

    header = (
        f"Eventi per '{kind}' in '{location}' "
        f"(query: {query}; {len(promoted)}/{len(results)} risultati rilevanti):\n"
    )

    lines = [header]

    for r, reason in promoted[:6]:
        title = r.get("title", "senza titolo")
        url = r.get("url", "")
        published = r.get("published_date", "")
        content = (r.get("content") or "")[:300]

        lines.append(f"• {title}")

        if reason:
            lines.append(f"  rilevanza: {reason}")

        if published:
            lines.append(f"  pubblicato: {published}")

        lines.append(f"  url: {url}")

        if content:
            lines.append(f"  {content}")

        lines.append("")

    return "\n".join(lines)