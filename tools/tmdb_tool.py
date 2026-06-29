"""
Recupera titoli, date e cast per film e serie TV.
"""

import os
import requests
from langchain_core.tools import tool

TMDB_BASE = "https://api.themoviedb.org/3"


def _fetch_overview_en(api_key: str, kind: str, item_id: int) -> str:
    """Fallback in inglese se l'overview italiana dell'hit della search e' vuota."""
    try:
        r = requests.get(
            f"{TMDB_BASE}/{kind}/{item_id}",
            params={"api_key": api_key, "language": "en-US"},
            timeout=10,
        )
        if r.ok:
            return r.json().get("overview") or ""
    except Exception:
        pass
    return ""


@tool
def tmdb_fact_check(title: str, kind: str = "movie") -> str:
    """
    Recupera i metadati di un film o serie TV su TMDb.

    Args:
        title: titolo da cercare
        kind: 'movie' o 'tv'

    Restituisce un blocco testuale con:
      - Match: 'esatto unico' (caso ideale), 'esatto ambiguo (N candidati...)'
        (piu' opere con lo stesso titolo, es. remake) o 'approssimato'
        (il titolo esatto non esiste, ripiego sul piu' popolare con titolo simile).
      - Titolo ufficiale, data di uscita, voto medio, cast principale, trama.
    """
    api_key = os.getenv("TMDB_API_KEY")

    if not api_key:
        return "Errore: TMDB_API_KEY non configurata in .env"

    if kind not in ("movie", "tv"):
        return "Errore: 'kind' deve essere 'movie' o 'tv'."

    try:
        r = requests.get(
            f"{TMDB_BASE}/search/{kind}",
            params={"api_key": api_key, "query": title, "language": "it-IT"},
            timeout=10,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
    except Exception as e:
        return f"Errore TMDb (search): {e}"

    if not results:
        return f"Nessun risultato TMDb per '{title}' ({kind})."

    hit = results[0]

    item_id = hit.get("id")
    if not item_id:
        return f"Errore: ID mancante per '{title}'."

    name_key = "title" if kind == "movie" else "name"
    date_key = "release_date" if kind == "movie" else "first_air_date"

    cast = []
    try:
        r2 = requests.get(
            f"{TMDB_BASE}/{kind}/{item_id}/credits",
            params={"api_key": api_key},
            timeout=10,
        )
        if r2.ok:
            cast = [c.get("name") for c in r2.json().get("cast", []) if c.get("name")][:5]
    except Exception:
        cast = []

    overview = hit.get("overview") or ""
    if not overview:
        overview = _fetch_overview_en(api_key, kind, item_id)

    official_title = hit.get(name_key) or "n/d"
    date = hit.get(date_key) or "n/d"
    vote_avg = hit.get("vote_average")
    vote_count = hit.get("vote_count", 0)

    vote_str = (
        f"{vote_avg} ({vote_count} voti)"
        if vote_avg is not None
        else "n/d"
    )

    tmdb_url = f"https://www.themoviedb.org/{kind}/{item_id}"

    return (
        f"URL: {tmdb_url}\n"
        f"Titolo ufficiale: {official_title}\n"
        f"Data: {date}\n"
        f"Voto medio: {vote_str}\n"
        f"Cast principale: {', '.join(cast) if cast else 'n/d'}\n"
        f"Trama: {overview[:400] if overview else 'n/d'}"
    )
