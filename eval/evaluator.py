"""
Modulo di valutazione del sistema blog-copilot.

Valuta ogni sessione di generazione su quattro assi:
  1. Source quality  — diversità e tipologia delle fonti usate
  2. Failure cases   — anomalie rilevabili dallo stato del grafo
  3. Qualitative     — qualità editoriale del draft (LLM-as-judge)
  4. LangSmith       — parametri di osservabilità (token, costo, latenza, LLM calls)
"""
from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from config.settings import get_llm


# ══════════════════════════════════════════════════════════════════════════════
# DATACLASS DI REPORT
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SourceQualityReport:
    total_docs: int
    local_docs: int
    web_docs: int
    structured_tool_docs: int
    diversity_ok: bool          
    notes: str


@dataclass
class FailureCase:
    code: str                   
    description: str
    detail: str


@dataclass
class QualityReport:
    score: float                
    notes: str


@dataclass
class LangSmithReport:
    run_id: str | None
    available: bool              
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost: float = 0.0
    latency_seconds: float = 0.0
    llm_calls: int = 0
    notes: str = ""


@dataclass
class EvalReport:
    topic: str
    post_type: str
    sources: SourceQualityReport
    failures: list[FailureCase]
    quality: QualityReport
    langsmith: LangSmithReport | None = None


# ══════════════════════════════════════════════════════════════════════════════
# 1. SOURCE QUALITY
# ══════════════════════════════════════════════════════════════════════════════

_STRUCTURED_TOOLS = {"tmdb_fact_check", "tvmaze_show_info", "find_local_events"}


def assess_source_quality(state: dict) -> SourceQualityReport:
    """
    Classifica i retrieved_docs in: locali (corpus RAG), web, tool strutturati.
    Valuta la diversità delle fonti.
    """
    docs = state.get("retrieved_docs", [])

    local_docs = 0
    web_docs = 0
    structured_tool_docs = 0

    for d in docs:
        meta = d.get("metadata") or {}
        if meta.get("type") == "local":
            local_docs += 1
        elif meta.get("tool") in _STRUCTURED_TOOLS:
            structured_tool_docs += 1
        else:
            web_docs += 1

    distinct_types = sum([local_docs > 0, web_docs > 0, structured_tool_docs > 0])
    diversity_ok = distinct_types >= 2

    notes_parts = []
    if local_docs == 0:
        notes_parts.append("nessuna fonte locale (corpus RAG non contribuisce)")
    if web_docs == 0:
        notes_parts.append("nessuna fonte web")
    if not diversity_ok:
        notes_parts.append("bassa diversità — un solo tipo di fonte")
    if structured_tool_docs == 0:
        notes_parts.append("nessun tool strutturato (TMDB/TVMaze/eventi) usato")

    return SourceQualityReport(
        total_docs=len(docs),
        local_docs=local_docs,
        web_docs=web_docs,
        structured_tool_docs=structured_tool_docs,
        diversity_ok=diversity_ok,
        notes="; ".join(notes_parts) if notes_parts else "OK",
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2. FAILURE CASES
# ══════════════════════════════════════════════════════════════════════════════

def detect_failure_cases(state: dict) -> list[FailureCase]:
    """
    Identifica anomalie rilevabili dallo stato del grafo dopo l'esecuzione.

    Failure cases implementati:
      FC-1  Self-RAG over-filtering (rag_search chiamato ma ha restituito zero doc)
      FC-2  Fonti insufficienti (< 2 retrieved_docs)
      FC-3  TVMaze non chiamato per una serie TV (topic_kind == tv_series)
      FC-4  Nessuna ricerca web durante il research
    """
    from graph.state import MAX_ITERATIONS

    failures: list[FailureCase] = []
    offset = state.get("tool_outputs_offset", 0)
    tool_outputs = (state.get("tool_outputs") or [])[offset:]
    tools_called = [to.get("tool", "") for to in tool_outputs]
    docs = state.get("retrieved_docs", [])
    topic_kind = state.get("topic_kind", "")

    # FC-1: Self-RAG over-filter — rag_search chiamato ma non ha prodotto documenti locali
    if "rag_search" in tools_called:
        rag_outputs = [to for to in tool_outputs if to.get("tool") == "rag_search"]
        all_empty = all(
            "nessun documento" in (to.get("output") or "").lower()
            for to in rag_outputs
        )
        if all_empty:
            failures.append(FailureCase(
                code="FC-1",
                description="Self-RAG over-filtering",
                detail="rag_search chiamato ma il grader Self-RAG ha scartato tutti i documenti — corpus locale ignorato.",
            ))

    # FC-2: Fonti insufficienti
    if len(docs) < 2:
        failures.append(FailureCase(
            code="FC-2",
            description="Fonti insufficienti",
            detail=f"Solo {len(docs)} documento/i selezionato/i dopo verify_and_select — post poco supportato.",
        ))

    # FC-3: TVMaze non chiamato per una serie TV
    if topic_kind == "tv_series" and "tvmaze_show_info" not in tools_called:
        failures.append(FailureCase(
            code="FC-3",
            description="TVMaze non chiamato per serie TV",
            detail=(
                f"topic_kind='{topic_kind}' ma tvmaze_show_info non è stato invocato. "
                "Per le serie TV è il tool preferito per dati su episodi e stagioni."
            ),
        ))

    # FC-4: Nessuna ricerca web
    if "web_search" not in tools_called:
        failures.append(FailureCase(
            code="FC-4",
            description="Nessuna ricerca web effettuata",
            detail="web_search non invocato durante il research — contenuto basato solo su fonti locali o tool strutturati.",
        ))

    return failures


# ══════════════════════════════════════════════════════════════════════════════
# 3. QUALITATIVE (LLM-as-judge)
# ══════════════════════════════════════════════════════════════════════════════

class _QualityVerdict(BaseModel):
    score: float = Field(ge=0, le=10, description="Punteggio qualità editoriale 0-10")
    notes: str = Field(description="Motivazione sintetica, max 2 frasi")


def evaluate_qualitative(state: dict) -> QualityReport:
    """
    LLM-as-judge sulla qualità editoriale del draft generato.
    Valuta: specificità, tono, struttura, uso citazioni, pertinenza al topic.
    """
    draft = state.get("current_draft", "")
    topic = state.get("current_topic", "?")
    post_type = state.get("post_type", "?")

    if not draft:
        return QualityReport(score=0.0, notes="nessun draft disponibile")

    judge = get_llm(temperature=0).with_structured_output(_QualityVerdict)
    prompt = (
        f"Valuta la qualità editoriale di questo post di un blog di cinema e serie TV.\n"
        f"Tipo: '{post_type}'  |  Topic: '{topic}'\n\n"
        f"POST:\n{draft[:3000]}\n\n"
        "Criteri (0–10):\n"
        "- Specificità e accuratezza delle informazioni (dati concreti, non solo affermazioni generiche)\n"
        "- Tono appropriato per un blog informativo su cinema/serie TV\n"
        "- Struttura chiara e leggibilità\n"
        "- Uso corretto delle citazioni (fonte indicata dopo ogni affermazione chiave)\n"
        "- Pertinenza e originalità rispetto al topic dichiarato\n\n"
        "Fornisci un punteggio da 0 a 10 e una nota sintetica in italiano (max 2 frasi)."
    )
    try:
        result = judge.invoke(prompt)
        return QualityReport(score=result.score, notes=result.notes)
    except Exception as e:
        return QualityReport(score=0.0, notes=f"errore valutazione: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_full_eval(state: dict) -> EvalReport:
    """Esegue i tre assi di valutazione su uno stato completato."""
    return EvalReport(
        topic=state.get("current_topic", "?"),
        post_type=state.get("post_type", "?"),
        sources=assess_source_quality(state),
        failures=detect_failure_cases(state),
        quality=evaluate_qualitative(state),
    )
