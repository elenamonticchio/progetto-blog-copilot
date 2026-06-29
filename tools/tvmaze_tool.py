"""
TVmaze tool: database specializzato in serie TV (schedule episodi, canali,
riassunti, date). API gratuita, senza chiave.
Complementa TMDb sul fronte dei calendari TV.
"""
import re
import requests
from datetime import date
from langchain_core.tools import tool

TVMAZE_BASE = "https://api.tvmaze.com"


def _strip_html(text: str) -> str:
    """Rimuove i tag HTML dai summary di TVmaze."""
    return re.sub(r"<[^>]+>", "", text or "").strip()


@tool
def tvmaze_show_info(title: str, with_episodes: bool = True) -> str:
    """
    Cerca una serie TV su TVmaze e restituisce metadati ufficiali:
    canale di messa in onda, giorni e orari di trasmissione, stato
    (in corso/conclusa), rating medio, generi, trama, e
    gli ultimi 3 episodi andati in onda e i prossimi 3 in programma.

    """
    try:
        params = {"q": title}
        if with_episodes:
            params["embed"] = "episodes"
        r = requests.get(f"{TVMAZE_BASE}/singlesearch/shows",
                         params=params, timeout=10)
        if r.status_code == 404:
            return f"Nessuna serie TV trovata su TVmaze per '{title}'."
        r.raise_for_status()
        show = r.json()
    except Exception as e:
        return f"Errore TVmaze: {e}"

    name = show.get("name", "?")
    status = show.get("status", "?")
    premiered = show.get("premiered", "n/d")
    ended = show.get("ended") or "in corso"
    network = (show.get("network") or show.get("webChannel") or {}).get("name", "n/d")
    schedule = show.get("schedule") or {}
    days = ", ".join(schedule.get("days") or []) or "n/d"
    air_time = schedule.get("time") or "n/d"
    rating = (show.get("rating") or {}).get("average")
    genres = ", ".join(show.get("genres") or []) or "n/d"
    summary = _strip_html(show.get("summary"))
    show_url = show.get("url") or f"{TVMAZE_BASE}/shows/{show.get('id', '')}"

    lines = [
        f"Titolo: {name}",
        f"URL: {show_url}", 
        f"Stato: {status}  |  Esordio: {premiered}  |  Fine: {ended}",
        f"Network: {network}  |  Schedule: {days} alle {air_time}",
        f"Generi: {genres}",
        f"Rating medio: {rating if rating is not None else 'n/d'}",
        f"Trama: {summary[:400]}" if summary else "Trama: n/d",
    ]

    if with_episodes:
        episodes = (show.get("_embedded") or {}).get("episodes") or []
        if episodes:
            today = date.today().isoformat()
            past = [e for e in episodes if (e.get("airdate") or "") and (e["airdate"] <= today)]
            future = [e for e in episodes if (e.get("airdate") or "") and (e["airdate"] > today)]

            lines.append(f"\nEpisodi totali: {len(episodes)}")
            if past:
                lines.append("Ultimi andati in onda:")
                for e in past[-3:]:
                    lines.append(
                        f"  S{e.get('season')}E{e.get('number')} - "
                        f"{e.get('name')} ({e.get('airdate')})"
                    )
            if future:
                lines.append("Prossimi in programma:")
                for e in future[:3]:
                    lines.append(
                        f"  S{e.get('season')}E{e.get('number')} - "
                        f"{e.get('name')} ({e.get('airdate')})"
                    )

    return "\n".join(lines)
