"""
Nodi del grafo (architettura modulare - separati dal cablaggio in graph/builder.py).
"""
import json
import random
import re
from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.types import interrupt

from graph.state import AgentState, PlannedPost, MAX_ITERATIONS, MAX_GROUNDING_RETRIES
from config.settings import get_llm
from config.domain import DOMAIN_TOPICS
from kg.kg_manager import KnowledgeGraphManager
from rag.retriever import format_citations

from tools import (
    web_search,
    rag_search,
    find_local_events,
    tvmaze_show_info,
    tmdb_fact_check,
    verify_claim,
    kg_get_topic_context,
)


class _InterestScore(BaseModel):
    score: float = Field(ge=0, le=10, description="Punteggio da 0 (scarso) a 10 (eccellente)")
    reason: str = Field(description="Motivazione sintetica, max una frase")


_SCORE_PROMPT = (
    "Valuta quanto questa risorsa e' interessante e di qualita' per un post "
    "di blog su film e serie TV, relativamente al topic '{topic}'.\n\n"
    "Risorsa:\n{content}\n\n"
    "Criteri: novita' informativa, autorevolezza della fonte, rilevanza al "
    "topic, originalita' del taglio.\n\n"
    "Scala di riferimento (usala per calibrare):\n"
    "  0-2  Irrilevante, spam, contenuto vuoto o fuori tema\n"
    "  3-4  Generico, poco specifico, fonte di bassa qualita'\n"
    "  5-6  Pertinente ma senza dati nuovi o angolazione originale\n"
    "  7-8  Informativo, fonte affidabile, aggiunge valore al post\n"
    "  9-10 Eccellente: dati esclusivi, fonte autorevole, taglio unico\n\n"
    "Punteggio da 0 a 10."
)


def _score_interestingness(topic: str, content: str) -> _InterestScore:
    scorer = get_llm(temperature=0).with_structured_output(_InterestScore)
    prompt = _SCORE_PROMPT.format(topic=topic, content=content[:2000])
    return scorer.invoke(prompt)


def _parse_claim_verdict(claim: str, tool_output: str) -> dict:
    """Parsa l'output di verify_claim in {claim, grounded, explanation}."""
    output = tool_output.strip()
    if "|" in output:
        verdict_str, explanation = output.split("|", 1)
    else:
        verdict_str, explanation = output, ""
    verdict_str = verdict_str.strip().upper()
    # PARTIAL conta come grounded 
    grounded = verdict_str in ("GROUNDED", "PARTIAL")
    return {"claim": claim, "grounded": grounded, "explanation": explanation.strip()}

def _doc_to_dict(doc) -> dict:
    def _sanitize(v):
        if type(v).__module__ == "numpy" or (hasattr(v, "item") and callable(v.item)):
            return v.item()
        return v

    metadata = {k: _sanitize(v) for k, v in doc.metadata.items()}
    return {"page_content": doc.page_content, "metadata": metadata}

_RESEARCH_TOOLS = [
    kg_get_topic_context,
    web_search,
    rag_search,
    find_local_events,
    tvmaze_show_info,
    tmdb_fact_check,
]
_RESEARCH_TOOLS_BY_NAME = {t.name: t for t in _RESEARCH_TOOLS}

_GROUNDING_TOOLS = [verify_claim]

_MAX_REACT_STEPS = 4

_W = 62 

def _hdr(label: str) -> None:
    fill = "─" * max(0, _W - len(label) - 5)
    print(f"\n─── {label} {fill}")

_RESEARCH_SYSTEM = """Sei l'agente di ricerca di un blog su film e serie TV.
Il tuo compito è raccogliere materiale di alta qualità per un post di tipo '{post_type}' sul topic '{topic}'.
Il classificatore ha identificato questo topic come appartenente alla categoria: '{topic_kind}'.

STRUMENTI A TUA DISPOSIZIONE:
- kg_get_topic_context: Interroga il Knowledge Graph per recuperare i post precedenti correlati a questo topic e i claim già sostenuti in passato. Chiamalo SEMPRE come primo passo: ti permette di capire cosa il blog ha già scritto, evitare ripetizioni e creare collegamenti espliciti tra post.
- rag_search: Cerca nel corpus locale del blog (linea editoriale, persone del cinema come registi e attori, concetti astratti, storia del cinema).
- web_search: Cerca nel web in tempo reale (news recenti, biografie di registi e attori, tendenze, gossip, recensioni esterne). Utile per qualsiasi topic che richiede informazioni aggiornate.
- tmdb_fact_check: Ottiene metadati ufficiali (cast, data uscita, voto medio). Utile se ti mancano dati tecnici verificati su un film o una serie TV specifica. Non produce risultati utili per persone, generi o eventi.
- tvmaze_show_info: Fornisce episodi, stagioni, stato di produzione e palinsesti aggiornati di una serie TV. Utile se hai già i metadati base (cast, voti) ma ti mancano ancora informazioni su quante stagioni ci sono, se è in corso o conclusa, o dettagli sugli episodi.
- find_local_events: Fornisce date, programma, giuria e logistica di festival o eventi cinematografici in una specifica area. Utile se stai scrivendo di un evento o festival e le ricerche web non ti hanno ancora dato dati strutturati (orari, location, programma ufficiale).

PROCESSO DI RAGIONAMENTO:
Prima di ogni chiamata a un tool scrivi un Thought che risponda a:
  - Cosa ho trovato finora?
  - Cosa mi manca ancora per questo post?
  - Perché questo tool è il più adatto adesso?
Non chiamare tool senza prima aver generato un Thought.

Principi:
- Inizia sempre con kg_get_topic_context per capire il contesto editoriale del blog prima di cercare fonti esterne.
- Scegli il tool in base alle informazioni che ti mancano, non in base a una regola fissa sul topic.
- Dopo ogni risultato ricevuto, ragiona su cosa hai trovato e cosa manca ancora.
- Se l'output di un tool è vuoto o irrilevante, non ripeterlo: valuta cosa può offrire un altro tool.
- Raccogli almeno 2 fonti distinte (oltre al KG) prima di fermarti.
- Non usare la tua conoscenza interna: hai bisogno di fonti citabili e verificabili.
- Fermati quando hai materiale sufficiente — non raccogliere più del necessario."""


class TopicKind(BaseModel):
    """Classificazione del topic per indirizzare la selezione dei tool."""
    kind: Literal["movie", "tv_series", "person", "concept", "event"] = Field(
        description=(
            "Categoria del topic. Esempi:\n"
            "- 'movie': titolo di un film specifico (es. 'Oppenheimer', 'Dune', 'Inception')\n"
            "- 'tv_series': titolo di una serie TV (es. 'House of the Dragon', 'The Last of Us')\n"
            "- 'person': una persona (regista, attore, sceneggiatore — es. 'Christopher Nolan', 'Greta Gerwig')\n"
            "- 'concept': genere, tema, piattaforma, tecnica (es. 'fantascienza', 'Netflix', 'IMAX', 'colonne sonore')\n"
            "- 'event': festival, anteprima, rassegna (es. 'festival cinema Sicilia', 'Venezia 2026')"
        )
    )
    confidence: float = Field(ge=0, le=1, description="Confidenza (0-1)")
    rationale: str = Field(description="Motivazione sintetica, max una frase")


def classify_topic(topic: str) -> dict:
    classifier = get_llm(temperature=0).with_structured_output(TopicKind)
    prompt = (
        "Classifica il seguente topic di un blog su film e serie TV "
        "nella categoria piu' appropriata.\n\n"
        f"Topic: '{topic}'"
    )
    try:
        out = classifier.invoke(prompt)
        return {"kind": out.kind, "confidence": out.confidence, "rationale": out.rationale}
    except Exception as e:
        return {"kind": "concept", "confidence": 0.0, "rationale": f"errore classificatore: {e}"}


def _extract_json_array(text: str):
    """Estrae il primo array JSON da una risposta LLM."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError("Nessun array JSON nella risposta")
    return json.loads(match.group(0))


# ====================================================================== #
# 1. SUGGEST TOPICS                                                        #
# ====================================================================== #

def suggest_topics(state: AgentState) -> dict:
    _hdr("SUGGEST TOPICS")
    kg = KnowledgeGraphManager()

    gaps = kg.get_coverage_gaps(DOMAIN_TOPICS)
    organic_gaps = kg.get_uncovered_related_topics()
    recent_topics = [p.get("topic") for p in kg.get_recent_posts(5) if p.get("topic")]

    n = state.get("posts_per_session", 1)
    pool_size = n + 2   

    all_gaps_covered = not gaps
    candidate_topics = list(DOMAIN_TOPICS)
    random.shuffle(candidate_topics)

    organic_hint = (
        f"Topic emersi da post precedenti ma mai approfonditi come argomento principale "
        f"(gap organici del blog, ottimi candidati): {organic_gaps[-10:]}\n"
        if organic_gaps else ""
    )

    if not all_gaps_covered:
        gap_list = list(gaps)
        random.shuffle(gap_list)
        topic_context = (
            f"Topic del dominio NON ancora trattati (priorità): {gap_list}\n"
            f"{organic_hint}"
            f"Puoi anche proporre topic nuovi non presenti in lista se sono particolarmente "
            f"rilevanti o attuali per il blog (film usciti di recente, notizie del settore, ecc.)."
        )
    else:
        topic_context = (
            f"Tutti i topic del dominio predefinito sono già stati trattati.\n"
            f"{organic_hint}"
            f"Universo topic di riferimento: {candidate_topics}\n"
            f"Sei libero di proporre topic completamente nuovi — film recenti, serie in uscita, "
            f"tendenze del momento, registi emergenti — purché coerenti con il blog."
        )

    prompt = (
        "Sei l'assistente editoriale di un blog su film e serie TV.\n"
        f"{topic_context}\n"
        f"Topic già trattati di recente, da NON ripetere: {recent_topics}\n\n"
        f"Proponi esattamente {pool_size} topic su cui vale la pena scrivere.\n"
        f"I topic devono essere DIVERSI tra loro per garantire varietà.\n"
        "Per ciascuno indica: topic (l'argomento), reason (perché è rilevante o attuale ora).\n"
        "Rispondi SOLO con un array JSON, senza testo extra ne backtick. Esempio:\n"
        '[{"topic": "...", "reason": "..."}]'
    )

    response = get_llm(temperature=0.7).invoke(prompt)

    try:
        suggestions = _extract_json_array(response.content)
    except (ValueError, json.JSONDecodeError):
        suggestions = [
            {"topic": t, "reason": "gap di copertura nel KG"}
            for t in candidate_topics[:pool_size]
        ]

    mode = "tutti coperti" if all_gaps_covered else f"{len(gaps)} gap nel KG"
    organic_info = f", {len(organic_gaps)} gap organici da CORRELATO_A" if organic_gaps else ""
    print(f"  {mode}{organic_info}  →  pool di {len(suggestions)} candidati per il planner")
    return {
        "suggested_topics": suggestions,
        "messages": [response],
        "reasoning_trace": [{
            "thought": (
                f"Genero un pool di {pool_size} topic candidati ({mode}{organic_info}). "
                f"Il planner sceglierà i {n} e deciderà il formato di ciascuno."
            ),
            "action": "kg.get_coverage_gaps + kg.get_uncovered_related_topics + kg.get_recent_posts + llm_ideation",
            "observation": f"{len(suggestions)} topic candidati generati",
        }],
    }


# ====================================================================== #
# 2. PLANNER                                                               #
# ====================================================================== #

class _OrderedPost(BaseModel):
    topic: str = Field(description="Topic del post, invariato rispetto al suggerimento")
    post_type: str = Field(description="Tipo: review, how-to, news, events")
    justification: str = Field(
        description=(
            "Motivazione editoriale di questo post IN QUESTA POSIZIONE: "
            "perché va pubblicato ora e perché in questo ordine rispetto agli altri."
        )
    )
    priority: int = Field(description="Posizione nella sequenza (1 = primo da pubblicare)")


class _ExcludedTopic(BaseModel):
    topic: str = Field(description="Topic non selezionato")
    reason: str = Field(description="Perché è stato escluso dal piano di questa sessione")


class _EditorialPlan(BaseModel):
    strategy: str = Field(
        description=(
            "Strategia editoriale complessiva in 2-3 frasi: "
            "perché questi topic, perché questo ordine, come garantiscono "
            "diversità e copertura del dominio."
        )
    )
    posts: list[_OrderedPost] = Field(description="Post selezionati, ordinati per priorità")
    excluded: list[_ExcludedTopic] = Field(
        description="Topic del pool NON selezionati per questa sessione, con motivazione"
    )


def _assign_publishing_dates(plan: list[dict], days_between: int) -> list[dict]:
    """
    Assegna una publishing_date a ciascun post del piano.
    Parte dalla data successiva all'ultimo post già pianificato nel KG
    (o da oggi se non ce ne sono), con passo `days_between` giorni.
    """
    from datetime import date, timedelta

    kg = KnowledgeGraphManager()
    last_date_str = kg.get_last_scheduled_date()

    if last_date_str:
        last_date = date.fromisoformat(last_date_str)
        candidate_start = last_date + timedelta(days=days_between)
        start_date = max(candidate_start, date.today())
        if candidate_start < date.today():
            print(
                f"\n  Nota: l'ultimo post era pianificato per {last_date_str} "
                f"(nel passato). Pianificazione dal {start_date} (oggi)."
            )
        else:
            print(
                f"\n  Ultimo post pianificato: {last_date_str}. "
                f"I nuovi post partiranno dal {start_date}."
            )
    else:
        start_date = date.today()
        print(f"\n  Nessun post pianificato in precedenza. Pianificazione dal {start_date}.")

    for i, p in enumerate(plan):
        p["publishing_date"] = (start_date + timedelta(days=i * days_between)).isoformat()

    return plan


def planner(state: AgentState) -> dict:
    """
    Costruisce un piano editoriale ordinato tramite LLM:
    - seleziona esattamente posts_per_session post dal pool di suggerimenti
    - decide l'ordine ottimale e giustifica ogni scelta
    - motiva anche le esclusioni (topic nel pool non selezionati)
    - classifica il topic del primo post per il routing dei tool
    """
    _hdr("PLANNER")
    suggestions = state.get("suggested_topics", [])
    n = state.get("posts_per_session", 1)

    if not suggestions:
        plan: list[PlannedPost] = [{
            "topic": "Topic di esempio",
            "post_type": "review",
            "justification": "fallback: nessun suggerimento disponibile",
            "priority": 1,
        }]
        strategy = "Piano di fallback: nessun suggerimento disponibile."
    else:
        suggestions_text = "\n".join(
            f"{i+1}. topic='{s.get('topic')}' | motivo={s.get('reason', '')}"
            for i, s in enumerate(suggestions)
        )
        planner_llm = get_llm(temperature=0.3).with_structured_output(_EditorialPlan)
        prompt = (
            "Sei il direttore editoriale di un blog su film e serie TV.\n\n"
            f"Hai ricevuto un pool di {len(suggestions)} topic candidati:\n"
            f"{suggestions_text}\n\n"
            f"Devi pianificare ESATTAMENTE {n} post da preparare in questa sessione.\n\n"
            "Il tuo compito:\n"
            f"1. SELEZIONA i {n} topic più adatti tra i candidati (non sei obbligato "
            "a includerli tutti — scegli quelli che garantiscono varietà, attualità "
            "e copertura del dominio).\n"
            "2. Per ogni topic selezionato, decidi il FORMATO del post corrispondente (post_type: review, how-to, "
            "news, events) in base alla natura del topic. "
            "I formati DEVONO essere tutti diversi tra loro — è vietato avere due post "
            "dello stesso tipo nella stessa sessione.\n"
            "3. Decidi l'ORDINE OTTIMALE di pubblicazione (considera attualità, "
            "varietà di formato, progressione logica per il lettore).\n"
            "4. Per ogni post selezionato, assegna una MOTIVAZIONE editoriale che spieghi "
            "perché è stato scelto, quale formato ha e perché in quella posizione.\n"
            "5. Per ogni topic NON selezionato, spiega brevemente perché è stato escluso.\n"
            "6. Descrivi la STRATEGIA complessiva della sessione.\n\n"
            f"IMPORTANTE: restituisci esattamente {n} post nel campo 'posts', "
            f"tutti con post_type diverso."
        )
        excluded_list: list = []
        try:
            result = planner_llm.invoke(prompt)
            strategy = result.strategy
            posts_sorted = sorted(result.posts, key=lambda x: x.priority)[:n]
            plan = [
                {
                    "topic": p.topic,
                    "post_type": p.post_type,
                    "justification": p.justification,
                    "priority": p.priority,
                }
                for p in posts_sorted
            ]
            excluded_list = result.excluded if result.excluded else []

            _ALL_TYPES = ["review", "how-to", "news", "events"]
            used_types: set[str] = set()
            for p in plan:
                if p["post_type"] in used_types:
                    replacement = next(
                        (t for t in _ALL_TYPES if t not in used_types), _ALL_TYPES[len(used_types) % 4]
                    )
                    print(f"  [fix] formato duplicato '{p['post_type']}' → '{replacement}' per '{p['topic']}'")
                    p["post_type"] = replacement
                used_types.add(p["post_type"])
        except Exception as e:
            print(f"  [errore LLM] {e} — fallback ordine originale")
            strategy = "Ordine originale dei suggerimenti (fallback da errore LLM)."
            plan = [
                {
                    "topic": s.get("topic", ""),
                    "post_type": "news",
                    "justification": s.get("reason", ""),
                    "priority": i + 1,
                }
                for i, s in enumerate(suggestions[:n])
            ]

    days_between = state.get("days_between_posts", 0)
    plan = _assign_publishing_dates(plan, days_between)

    strategy_short = strategy[:200].replace("\n", " ")
    print(f"  Strategia: {strategy_short}")
    print()
    for p in plan:
        just = p['justification'][:120].replace("\n", " ")
        date_tag = f"  [{p['publishing_date']}]" if p.get("publishing_date") else ""
        print(f"  ✓  [{p['priority']}] {p['topic']}  →  {p['post_type']}{date_tag}")
        print(f"     {just}")
    if excluded_list:
        print()
        for e in excluded_list:
            reason = e.reason[:100].replace("\n", " ")
            print(f"  ✗  {e.topic}  →  {reason}")

    current_topic = plan[0]["topic"]
    current_post_type = plan[0]["post_type"]

    classification = classify_topic(current_topic)
    topic_kind = classification["kind"]
    print(f"\n  → topic corrente: {current_topic}  |  kind={topic_kind}  (conf {classification['confidence']:.1f})")

    return {
        "editorial_plan": plan,
        "current_post_index": 0,
        "current_topic": current_topic,
        "post_type": current_post_type,
        "topic_kind": topic_kind,
        "reasoning_trace": [{
            "thought": "Ordino i topic suggeriti e costruisco una strategia editoriale motivata.",
            "action": "llm.with_structured_output(_EditorialPlan) + classify_topic + _assign_publishing_dates",
            "observation": (
                f"Strategia: {strategy[:200]} | "
                "Piano: " + str([
                    str(p["priority"]) + ". " + p["topic"]
                    + (f" [{p['publishing_date']}]" if p.get("publishing_date") else "")
                    for p in plan
                ])
            ),
        }],
    }


# ====================================================================== #
# 3. RESEARCH                                                              #
# ====================================================================== #

def research(state: AgentState) -> dict:
    topic = state["current_topic"]
    post_type = state["post_type"]
    topic_kind = state.get("topic_kind") or "concept"  # fallback conservativo
    _hdr(f"RESEARCH  {topic} | {post_type} | {topic_kind}")

    llm = get_llm(temperature=0.2).bind_tools(_RESEARCH_TOOLS, parallel_tool_calls=False)

    from datetime import date
    today = date.today().strftime("%B %Y") 
    human_content = (
        f"Raccogli materiale per un post '{post_type}' (topic_kind={topic_kind}) "
        f"sul topic: '{topic}'.\n"
        f"Data odierna: {today}. Quando cerchi notizie o tendenze recenti, usa questa data "
        f"come riferimento e includi l'anno corrente nelle query di ricerca.\n"
        f"Inizia chiamando kg_get_topic_context('{topic}') per capire il contesto editoriale "
        f"del blog prima di procedere con le ricerche esterne.\n"
    )

    messages = [
        SystemMessage(content=_RESEARCH_SYSTEM.format(
            post_type=post_type, topic=topic, topic_kind=topic_kind
        )),
        HumanMessage(content=human_content),
    ]

    reasoning_trace: list[dict] = []
    tool_outputs: list[dict] = []

    for step in range(_MAX_REACT_STEPS):
        response = llm.invoke(messages)
        messages.append(response)

        tool_calls = getattr(response, "tool_calls", None) or []
        # Rimuove eventuali prefissi "Thought[...]: " che il modello antepone al testo
        raw_thought = (response.content or "").strip()
        thought_text = re.sub(r"^Thought[^:]*:\s*", "", raw_thought, flags=re.IGNORECASE).strip()
        thought_text = thought_text.strip('"').strip()

        first_line = next((l.strip() for l in thought_text.splitlines() if l.strip()), "")
        thought_short = (first_line[:120] + "…") if len(first_line) > 120 else first_line

        print(f"\n  step {step + 1}")
        if thought_short:
            print(f"  Thought : {thought_short}")
        else:
            print(f"  Thought : (nessun pensiero prodotto)")

        if not tool_calls:
            reasoning_trace.append({
                "thought": thought_text[:300] or "Materiale sufficiente, chiudo la ricerca.",
                "action": "no_action",
                "observation": thought_text[:300],
            })
     
            if len(tool_outputs) < 2 and step < _MAX_REACT_STEPS - 1:
                continue
            break

        tc = tool_calls[0]  # sequential: sempre un tool per volta
        tool_name = tc["name"]
        tool_args = tc.get("args", {})
        main_arg = next(iter(tool_args.values()), "") if tool_args else ""
        print(f"  Action  : {tool_name}({main_arg!r})")

        t = _RESEARCH_TOOLS_BY_NAME.get(tool_name)
        if t is None:
            obs = f"Tool '{tool_name}' non disponibile."
        else:
            try:
                obs = t.invoke(tool_args)
            except Exception as e:
                obs = f"Errore esecuzione {tool_name}: {e}"

        messages.append(ToolMessage(content=obs, tool_call_id=tc["id"]))

        obs_preview = (obs or "")[:160].replace("\n", " ")
        print(f"  Obs     : {obs_preview}...")

        reasoning_trace.append({
            "thought": thought_text[:300] or f"Chiamo {tool_name} per raccogliere materiale sul topic '{topic}'.",
            "action": f"{tool_name}({tool_args})",
            "observation": (obs or "")[:500],
        })
        tool_outputs.append({
            "tool": tool_name,
            "args": str(tool_args),
            "output": (obs or "")[:1000],
        })

    print(f"\n  → {len(tool_outputs)} tool eseguiti")

    # Estrae kg_context dall'output del tool, per alimentare il nodo draft.
    kg_context = ""
    for to in tool_outputs:
        if to.get("tool") == "kg_get_topic_context":
            kg_context = to.get("output", "")
            break

    if not tool_outputs:
        print("  [fallback] nessun tool chiamato — forzo web_search")
        try:
            fallback_query = f"{topic} {post_type}"
            obs = web_search.invoke({"query": fallback_query})
            tool_outputs.append({
                "tool": "web_search",
                "args": str({"query": fallback_query}),
                "output": (obs or "")[:1000],
            })
            reasoning_trace.append({
                "thought": "Nessun tool chiamato nel loop ReAct. Forzo web_search per garantire fonti esterne.",
                "action": f"web_search(query='{fallback_query}')",
                "observation": (obs or "")[:500],
            })
        except Exception as e:
            print(f"  [fallback] web_search fallito: {e}")

    return {
        "kg_context": kg_context,
        "retrieved_docs": [],
        "tool_outputs": tool_outputs,
        "reasoning_trace": reasoning_trace,
    }


# ====================================================================== #
# 4. VERIFY_AND_SELECT                                                     #
# ====================================================================== #

def verify_and_select(state: AgentState) -> dict:
    topic = state["current_topic"]
    _hdr(f"VERIFY & SELECT  {topic}")

    new_tool_outputs: list[dict] = []
    new_reasoning: list[dict] = []

    tool_outputs = state.get("tool_outputs", []) or []
    all_docs = []

    offset = state.get("tool_outputs_offset", 0)
    for to in tool_outputs[offset:]:
        content = to.get("output", "") or ""
        if not content:
            continue

        tool_name = to.get("tool", "online")

        if tool_name == "kg_get_topic_context":
            continue
        
        from langchain_core.documents import Document

        if tool_name == "web_search":
            # Formato: [titolo]\nurl: https://...\ncontent\n\n---\n\n[titolo]...
            for result in content.split("\n\n---\n\n"):
                if not result.strip():
                    continue
                url = "url_sconosciuto"
                for line in result.split("\n"):
                    if line.strip().startswith("url:"):
                        url = line.strip().replace("url:", "").strip()
                        break
                result_content = "\n".join(
                    line for i, line in enumerate(result.split("\n"))
                    if i > 0 and not line.strip().startswith("url:")
                ).strip()
                if result_content:
                    all_docs.append(Document(
                        page_content=result_content,
                        metadata={"source": url, "type": "online", "tool": tool_name},
                    ))

        elif tool_name == "rag_search":
            # Formato: [fonte: X]\ncontent\n\n---\n\n[fonte: Y]\ncontent\nFonti: X, Y
            for section in content.split("\n\n---\n\n"):
                lines = section.strip().split("\n")
                if not lines:
                    continue
                source = "rag_search"
                first = lines[0].strip()
                if first.startswith("[fonte:") and first.endswith("]"):
                    source = first[7:-1].strip()
                section_content = "\n".join(
                    line for line in lines[1:]
                    if not line.startswith("Fonti:")
                ).strip()
                if section_content:
                    all_docs.append(Document(
                        page_content=section_content,
                        metadata={"source": source, "type": "local", "tool": tool_name},
                    ))

        else:
            # tmdb_fact_check, tvmaze_show_info, find_local_events
            url_match = re.search(r'(?:url|URL):\s*(https?://[^\s]+)', content)
            source_val = url_match.group(1) if url_match else tool_name
            all_docs.append(Document(
                page_content=content,
                metadata={"source": source_val, "type": "online", "tool": tool_name},
            ))

    rag_from_tool = sum(1 for d in all_docs if d.metadata.get("tool") == "rag_search")
    web_and_other = len(all_docs) - rag_from_tool
    print(f"  {rag_from_tool} locali  +  {web_and_other} online  =  {len(all_docs)} documenti")

    kept_docs = []

    if not all_docs:
        print("  (nessun documento da filtrare)")
        new_reasoning.append({
            "thought": "Nessun documento da filtrare.",
            "action": "no_action",
            "observation": "all_docs vuoto",
        })
    else:
        _MIN_SCORE = 5.0

        scored: list[tuple[float, object, str]] = []
        for doc in all_docs:
            try:
                result = _score_interestingness(topic=topic, content=doc.page_content)
                score = result.score
                raw = f"score={score:.1f}/10  motivo: {result.reason}"
            except Exception as e:
                raw = f"errore scorer: {e}"
                score = 0.0
            scored.append((score, doc, raw))

        scored.sort(key=lambda x: x[0], reverse=True)

        above = [(s, d, r) for s, d, r in scored if s >= _MIN_SCORE]
        kept_docs = [doc for _, doc, _ in above[:5]]

        if not kept_docs and scored:
            kept_docs = [scored[0][1]]
            print("  [fallback] nessun doc sopra soglia — tenuto il migliore disponibile")

        print()
        for score, doc, _ in scored:
            source = doc.metadata.get("source", "?")
            doc_type = doc.metadata.get("type", "?")
            from urllib.parse import urlparse
            label = urlparse(source).netloc or source
            content_preview = doc.page_content[:80].replace("\n", " ")
            kept = "✓" if score >= _MIN_SCORE else "✗"
            print(f"  {kept} [{score:.1f}]  {label}  ({doc_type})")
            print(f"         {content_preview}...")

        n_scartati = sum(1 for s, _, _ in scored if s < _MIN_SCORE)
        print()
        print(f"  → {len(kept_docs)} tenuti  |  {n_scartati} scartati (score < {_MIN_SCORE})")

        new_tool_outputs.append({
            "tool": "score_interestingness",
            "args": f"{len(all_docs)} doc (locali+online) su '{topic}'",
            "output": (
                f"{len(kept_docs)} tenuti su {len(all_docs)} (soglia {_MIN_SCORE})"
            ),
        })
        new_reasoning.append({
            "thought": "Filtro tutti i documenti per qualita': tengo solo quelli con score>=5 (pertinenti).",
            "action": f"score_interestingness x {len(all_docs)}",
            "observation": f"{len(kept_docs)} tenuti su {len(all_docs)} (soglia {_MIN_SCORE})",
        })

    new_citations = format_citations(kept_docs)

    return {
        "retrieved_docs": [_doc_to_dict(d) for d in kept_docs],
        "citations": new_citations,
        "tool_outputs": new_tool_outputs,
        "reasoning_trace": new_reasoning,
    }


# ====================================================================== #
# 5. DRAFT                                                                 #
# ====================================================================== #

_RELEVANT_TOOLS_FOR_DRAFT = {
    "tmdb_fact_check", "web_search", "find_local_events", "tvmaze_show_info"
}


class DraftPost(BaseModel):
    """Bozza completa di un post del blog (structured output)."""
    title: str = Field(
        description="Titolo accattivante del post, 6-12 parole, in italiano."
    )
    body: str = Field(
        description=(
            "Corpo del post in markdown, ~500-800 parole, in italiano. "
            "Inserisci citazioni inline tra parentesi quadre subito dopo le affermazioni "
            "che provengono dai documenti, usando il nome file della fonte: es. "
            "'Nolan gira spesso in IMAX 70mm [christopher_nolan.md].' "
            "Concludi con una sezione '## Fonti' che elenca tutti i file citati."
        )
    )
    key_claims: list[str] = Field(
        description=(
            "Da 1 a 5 affermazioni fattuali tratte ESCLUSIVAMENTE dai documenti forniti. "
            "Regola fondamentale: ogni claim deve essere rintracciabile in una delle fonti — "
            "se non riesci a citare da quale documento proviene, NON includerlo. "
            "È preferibile avere 2 claim solidi che 5 vaghi.\n"
            "BUONI (specifici, da fonte): "
            "'Hans Zimmer ha composto la colonna sonora di Inception (2010) [christopher_nolan.md].' "
            "'Denis Villeneuve ha scelto Hans Zimmer per Dune (2021) [denis_villeneuve.md].'\n"
            "DA EVITARE (opinioni, generalizzazioni, conoscenza interna LLM): "
            "'La colonna sonora è essenziale per il successo di un film.' "
            "'La musica deve integrarsi con la narrazione.' "
            "'Un tema distintivo è fondamentale.' "
            "Questi sono giudizi editoriali, non fatti verificabili da una fonte."
        )
    )


def _format_docs_for_prompt(docs: list) -> str:
    if not docs:
        return "(nessun documento selezionato)"
    parts = []
    for d in docs:
        if isinstance(d, dict):
            source = d.get("metadata", {}).get("source", "?")
            content = d.get("page_content", "")[:1500]
        else:
            source = d.metadata.get("source", "?")
            content = d.page_content[:1500]
        parts.append(f"[fonte: {source}]\n{content}")
    return "\n\n".join(parts)


_DRAFT_PROMPT = """Sei il copywriter editoriale di un blog di film e serie TV.
Scrivi un post di tipo '{post_type}' sul topic '{topic}'.

CONTESTO DAL KNOWLEDGE GRAPH (usa questo per garantire coerenza e fare collegamenti espliciti):
{kg_context}

DOCUMENTI SELEZIONATI (cita queste fonti nel post):
{docs_with_sources}

FONTI DISPONIBILI (cita SOLO queste, non inventarne altre):
{available_sources}

ISTRUZIONI:
- Tono informato e accessibile, ~500-800 parole.
- Se il CONTESTO DAL KNOWLEDGE GRAPH contiene post o claim correlati al topic, fai
  riferimento esplicito a quei collegamenti nel testo (es. "Come già esplorato nel nostro
  articolo su X..." oppure "Questo si collega al tema Y che abbiamo trattato in...").
  Il post deve mostrare chiaramente che il ragionamento è informato dal Knowledge Graph.
- Inserisci citazioni inline [nome_fonte] dopo le affermazioni che provengono dai documenti.
- Cita SOLO le fonti elencate in "FONTI DISPONIBILI". Non usare [web_search] o altri nomi
  se non compaiono in quella lista.
- Concludi con una sezione "## Fonti" che elenca tutte le fonti citate.
- Estrai 3-5 key_claims brevi e dichiarativi (in italiano) sui fatti sostenuti nel post.

Genera title, body, key_claims."""


def draft(state: AgentState) -> dict:
    topic = state["current_topic"]
    post_type = state["post_type"]
    kg_context = state.get("kg_context") or "(nessun post precedente correlato)"
    retrieved_docs = state.get("retrieved_docs") or []

    _hdr(f"DRAFT  {post_type} / {topic}")

    tool_outputs_state = state.get("tool_outputs", []) or []
    doc_sources = list({
        (d.get("metadata", {}).get("source") if isinstance(d, dict) else d.metadata.get("source"))
        for d in retrieved_docs
        if (d.get("metadata", {}).get("source") if isinstance(d, dict) else d.metadata.get("source"))
    })
    tool_sources = list({
        to.get("tool") for to in tool_outputs_state
        if to.get("tool") in _RELEVANT_TOOLS_FOR_DRAFT
    })
    available_sources = doc_sources + tool_sources
    available_sources_str = (
        ", ".join(f"[{s}]" for s in available_sources)
        if available_sources else "(nessuna fonte disponibile)"
    )

    user_msg = _DRAFT_PROMPT.format(
        post_type=post_type,
        topic=topic,
        kg_context=kg_context[:1500],
        docs_with_sources=_format_docs_for_prompt(retrieved_docs),
        available_sources=available_sources_str,
    )

    user_feedback = state.get("user_feedback", "")
    if user_feedback:
        user_msg += (
            f"\n\nFEEDBACK DELL'UTENTE (incorpora queste correzioni nella nuova versione):\n"
            f"{user_feedback}"
        )

    grounding_feedback = state.get("grounding_feedback", [])
    if grounding_feedback:
        feedback_lines = "\n".join(
            f"  - \"{f.get('claim', '')[:80]}\": {f.get('explanation', 'non grounded')[:120]}"
            for f in grounding_feedback
        )
        user_msg += (
            "\n\nFEEDBACK DI VERIFICA AUTOMATICA — i seguenti claim non sono supportati "
            "dalle fonti disponibili. Rimuovili o sostituiscili con affermazioni verificabili:\n"
            + feedback_lines
        )

    writer = get_llm(temperature=0.5).with_structured_output(DraftPost)
    try:
        out = writer.invoke(user_msg)
        title = out.title
        body = out.body
        claims = out.key_claims
    except Exception as e:
        print(f"  [errore structured output] {e}")
        title = f"Post su {topic}"
        body = f"(errore nella generazione automatica: {e})"
        claims = []

    print(f"  Titolo : \"{title}\"")
    print(f"  Misura : {len(body)} char  |  {len(claims)} key_claims")

    return {
        "current_title": title,
        "current_draft": body,
        "key_claims": claims,
        "reasoning_trace": [{
            "thought": (
                f"Scrivo un post '{post_type}' su '{topic}' basandomi sulle "
                f"informazioni filtrate da verify_and_select: kg_context e "
                f"{len(retrieved_docs)} documenti (locali + online) selezionati."
            ),
            "action": "llm.with_structured_output(DraftPost)",
            "observation": f"Titolo: '{title}' | body {len(body)} char | {len(claims)} key_claims",
        }],
    }


# ====================================================================== #
# 5b. VERIFY_GROUNDING                                                   #
# ====================================================================== #

def verify_grounding(state: AgentState) -> dict:
    """
    Agente ReAct con tool-calling: verifica ogni key_claim contro le fonti recuperate
    chiamando il tool verify_claim per ciascuna affermazione.
    Se più del 40% dei claim non è supportato, restituisce grounding_feedback
    per far riscrivere il draft. Massimo MAX_GROUNDING_RETRIES tentativi.
    """
    claims  = state.get("key_claims", [])
    docs    = state.get("retrieved_docs", [])
    retries = state.get("grounding_retries", 0)

    _hdr(f"GROUNDING  {len(claims)} claim  /  {len(docs)} fonti")

    if not claims:
        print("  (nessun claim — skip)")
        return {"grounding_feedback": [], "grounding_passed": True}

    docs_text = "\n\n---\n\n".join(
        f"[DOC {i+1} — {d.get('metadata', {}).get('source', '?')}]\n"
        f"{d.get('page_content', '')[:1500]}"
        for i, d in enumerate(docs)
    ) or "(nessuna fonte disponibile)"

    claims_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))

    llm = get_llm(temperature=0).bind_tools(_GROUNDING_TOOLS, parallel_tool_calls=False)

    system = (
        "Sei un fact-checker rigoroso per un blog di cinema e serie TV.\n"
        "Il tuo compito è verificare ogni claim della lista usando lo strumento verify_claim.\n"
        "Per ogni claim: scegli il documento più rilevante come evidenza e chiama "
        "verify_claim(claim=..., evidence=...).\n"
        "Devi verificare TUTTI i claim nella lista, uno alla volta."
    )
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=(
            f"CLAIM DA VERIFICARE ({len(claims)} totali):\n{claims_text}\n\n"
            f"DOCUMENTI DISPONIBILI:\n{docs_text}"
        )),
    ]

    verdicts: list[dict] = []
    verified_claims: set[str] = set()
    max_steps = len(claims) + 3

    for step in range(max_steps):
        response = llm.invoke(messages)
        messages.append(response)

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            break

        tc = tool_calls[0]
        tool_name = tc["name"]
        tool_args = tc.get("args", {})
        claim_arg = tool_args.get("claim", "")

        if tool_name != "verify_claim":
            obs = f"Tool '{tool_name}' non disponibile."
        else:
            try:
                obs = verify_claim.invoke(tool_args)
            except Exception as e:
                obs = f"NOT_GROUNDED|Errore durante la verifica: {e}"

        messages.append(ToolMessage(content=obs, tool_call_id=tc["id"]))

        verdict_label = obs.split("|")[0].strip().upper() if "|" in obs else obs.strip().upper()
        print(f"  step {step + 1}  {tool_name} → {verdict_label}  \"{claim_arg[:60]}\"")

        if tool_name == "verify_claim" and claim_arg not in verified_claims:
            verdicts.append(_parse_claim_verdict(claim_arg, obs))
            verified_claims.add(claim_arg)

        if len(verdicts) >= len(claims):
            break

    # Fallback per claim non raggiunti dall'agente.
    verified_prefixes = {v["claim"][:40] for v in verdicts}
    for c in claims:
        if c[:40] not in verified_prefixes:
            verdicts.append({"claim": c, "grounded": True, "explanation": "non verificato"})

    unsupported = [v for v in verdicts if not v["grounded"]]
    total       = len(verdicts)
    passed      = (total - len(unsupported)) / total >= 0.6 if total else True

    supported_count = total - len(unsupported)
    if passed:
        print(f"  ✓ {supported_count}/{total} supportati")
    else:
        print(f"  ✗ {supported_count}/{total} supportati  (tentativo {retries + 1}/{MAX_GROUNDING_RETRIES + 1})")

    for v in unsupported:
        print(f"  ✗ \"{v['claim'][:90]}\"")
        print(f"    → {v.get('explanation', '')[:80]}")

    if passed:
        return {"grounding_feedback": unsupported, "grounding_passed": True}
    else:
        return {
            "grounding_feedback": unsupported,
            "grounding_passed": False,
            "grounding_retries": retries + 1,
        }


# ====================================================================== #
# ESTRAZIONE TOPIC CORRELATI                                             #
# ====================================================================== #

class _RelatedTopicsList(BaseModel):
    """Topic, persone o opere correlate al post, usati per creare CORRELATO_A nel KG."""
    topics: list[str] = Field(
        description=(
            "Nomi brevi e canonici (max 5). Usa il nome ufficiale senza aggiunte: "
            "'Greta Gerwig' non 'Greta Gerwig (regista)', 'Inception' non 'Inception (2010)'."
        )
    )


def _extract_related_topics(state: dict) -> list[str]:
    """
    Estrae entità correlate dal draft approvato per popolare le relazioni CORRELATO_A nel KG.
    Queste relazioni vengono usate in due modi:
      1. K-RAG: expand_query_for_rag() le usa per arricchire le query di retrieval future.
      2. Topic suggestion: get_uncovered_related_topics() le usa per proporre argomenti
         menzionati nel blog ma mai trattati come topic principale.
    """
    draft = state.get("current_draft", "")
    topic = state.get("current_topic", "")
    if not draft:
        return []
    extractor = get_llm(temperature=0).with_structured_output(_RelatedTopicsList)
    prompt = (
        f"Dal seguente post di blog sul topic '{topic}', estrai al massimo 5 nomi "
        f"di film, serie TV, registi o generi che vengono menzionati nel testo.\n"
        f"Questi verranno salvati nel Knowledge Graph come topic correlati: "
        f"serviranno come candidati per post futuri non ancora scritti.\n\n"
        f"Regole:\n"
        f"- Scegli solo topic pertinenti al dominio di un blog di cinema e serie TV "
        f"(titoli, registi, generi, piattaforme, formati audiovisivi...).\n"
        f"- Ogni topic deve poter diventare il soggetto di un post futuro sul blog.\n"
        f"- Escludi personaggi, eventi storici, persone reali non legate al cinema, "
        f"temi narrativi generici e argomenti troppo vaghi o "
        f"non specifici del mondo audiovisivo.\n\n"
        f"Post:\n{draft}"
    )
    try:
        result = extractor.invoke(prompt)
        return [t for t in result.topics if t.lower() != topic.lower()]
    except Exception as e:
        print(f"   [update_kg] estrazione related_topics fallita: {e}")
        return []


# ====================================================================== #
# NODI HUMAN-IN-THE-LOOP E AGGIORNAMENTO KG                               #
# ====================================================================== #

def human_review(state: AgentState) -> dict:
    """Sospende il grafo e attende la decisione umana via interrupt."""
    iteration = state.get("iteration_count", 0) + 1
    _hdr(f"HUMAN REVIEW  iterazione {iteration}")

    decision = interrupt({
        "title": state.get("current_title", ""),
        "draft": state.get("current_draft", ""),
        "key_claims": state.get("key_claims", []),
        "citations": [c.get("source") for c in state.get("citations", [])],
        "iteration": iteration,
        "unsupported_claims": state.get("grounding_feedback", []),
    })

    action = decision.get("action", "reject")
    feedback = decision.get("feedback", "")

    if action == "approve":
        print("  → approvato")
        return {
            "user_status": "approved",
            "user_feedback": "",
            "approved": True,
            "iteration_count": iteration,
        }
    elif action == "modify":
        print(f"  → modifica: {feedback[:80]}")
        return {
            "user_status": "modify",
            "user_feedback": feedback,
            "approved": False,
            "iteration_count": iteration,
            "grounding_retries": 0,
            "grounding_feedback": [],
            "grounding_passed": True,
        }
    else:  
        print("  → rigettato — rigenero la bozza")
        return {
            "user_status": "rejected",
            "user_feedback": "",
            "approved": False,
            "iteration_count": iteration,
            "grounding_retries": 0,
            "grounding_feedback": [],
            "grounding_passed": True,
        }


def update_kg(state: AgentState) -> dict:
    """Aggiorna incrementalmente il KG dopo approvazione umana.

    Estrae i topic correlati dal draft per creare relazioni CORRELATO_A,
    necessarie a expand_query_for_rag() nelle esecuzioni future (K-RAG).
    """
    title    = state.get("current_title", "")
    topic    = state.get("current_topic", "")
    post_type = state.get("post_type", "")
    raw_claims = state.get("key_claims", [])
    sources  = [c.get("source", "") for c in state.get("citations", []) if c.get("source")]

    # Rimuove l'attribuzione alla fonte dai claim non supportati
    ungrounded_texts = {
        v.get("claim", "").strip()
        for v in state.get("grounding_feedback", [])
    }
    claims = []
    for c in raw_claims:
        if any(c.strip().startswith(u[:40]) for u in ungrounded_texts if u):
            clean = re.sub(r"\s*\[[^\]]+\]\s*$", "", c).strip()
            claims.append(clean)
        else:
            claims.append(c)


    idx = state.get("current_post_index", 0)
    plan = state.get("editorial_plan", [])
    planned_date = plan[idx].get("publishing_date", "") if idx < len(plan) else ""

    related_topics = _extract_related_topics(state)
    _hdr("UPDATE KG")
    print(f"  \"{title}\"")
    n_stripped = sum(1 for orig, saved in zip(raw_claims, claims) if orig != saved)
    date_info = f"  |  data: {planned_date}" if planned_date else ""
    print(f"  {len(claims)} claim  |  {len(sources)} fonti  |  correlato a: {related_topics}{date_info}")
    if n_stripped:
        print(f"  ({n_stripped} claim salvati senza fonte — non verificati dalle fonti recuperate)")
    kg = KnowledgeGraphManager()
    kg.add_approved_post(
        title=title,
        topic=topic,
        post_type=post_type,
        claims=claims,
        sources=sources,
        related_topics=related_topics if related_topics else None,
        planned_date=planned_date or None,
    )

    next_idx = state.get("current_post_index", 0) + 1
    return {
        "current_post_index": next_idx,
        "reasoning_trace": [{
            "thought": "Post approvato: lo registro nel KG con le sue relazioni.",
            "action": "kg.add_approved_post",
            "observation": (
                f"KG aggiornato: '{title}' ({post_type}) — "
                f"{len(claims)} claim, {len(sources)} fonti, "
                f"CORRELATO_A → {related_topics}"
            ),
        }],
    }


def advance_post(state: AgentState) -> dict:
    """Carica il post successivo dal piano editoriale e resetta lo stato per-post."""
    plan = state.get("editorial_plan", [])
    idx = state.get("current_post_index", 0)   

    next_post = plan[idx]
    topic = next_post["topic"]
    post_type = next_post["post_type"]
    pub_date = next_post.get("publishing_date", "")

    total = min(len(plan), state.get("posts_per_session", 1))
    date_tag = f"  [{pub_date}]" if pub_date else ""
    _hdr(f"ADVANCE  post {idx + 1}/{total}  →  {topic} ({post_type}){date_tag}")

    classification = classify_topic(topic)
    topic_kind = classification["kind"]
    print(f"  kind={topic_kind}  (conf {classification['confidence']:.1f})  —  {classification['rationale']}")

    return {
        "current_topic": topic,
        "post_type": post_type,
        "topic_kind": topic_kind,
        # Reset campi per-post
        "iteration_count": 0,
        "user_feedback": "",
        "user_status": "",
        "approved": False,
        "retrieved_docs": [],
        "citations": [],
        "current_draft": "",
        "current_title": "",
        "key_claims": [],
        "kg_context": "",
        "grounding_feedback": [],
        "grounding_retries": 0,
        "grounding_passed": True,
        "tool_outputs_offset": len(state.get("tool_outputs", [])),
        "reasoning_trace": [{
            "thought": f"Avanzo al post {idx + 1}/{total} del piano: '{topic}' ({post_type}).",
            "action": "advance_post + classify_topic",
            "observation": f"topic_kind={topic_kind} (conf={classification['confidence']:.1f})",
        }],
    }


# ====================================================================== #
# ROUTING CONDIZIONALE (dopo il grounding, la revisione umana e kg update) #
# ====================================================================== #

def route_after_grounding(state: AgentState) -> str:
    """Grounding ok → human_review. Fallito e retry disponibile → draft. Esauriti → human_review."""
    passed  = state.get("grounding_passed", True)
    retries = state.get("grounding_retries", 0)

    if passed:
        return "human_review"
    if retries <= MAX_GROUNDING_RETRIES:
        feedback = state.get("grounding_feedback", [])
        print(f"  → retry draft  ({len(feedback)} claim da correggere)")
        return "draft"
    print(f"  → max retry raggiunto — procedo a human_review")
    return "human_review"


def route_after_review(state: AgentState) -> str:
    """approvato -> update_kg ; rigettato -> draft ; troppi tentativi -> end."""
    if state.get("approved"):
        return "update_kg"
    if state.get("iteration_count", 0) >= MAX_ITERATIONS:
        print("   (raggiunto MAX_ITERATIONS: mi fermo)")
        return "end"
    return "draft"


def route_after_kg(state: AgentState) -> str:
    """Ci sono altri post nel piano da preparare? -> advance_post, altrimenti end."""
    idx = state.get("current_post_index", 0)   # già incrementato da update_kg
    plan = state.get("editorial_plan", [])
    n = state.get("posts_per_session", 1)

    if idx < len(plan) and idx < n:
        print(f"  → altro post da preparare ({idx + 1}/{min(len(plan), n)})")
        return "advance_post"
    print(f"  → piano completato  ({idx}/{min(len(plan), n)} post)")
    return "end"
