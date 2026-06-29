"""
Entry point: costruisce lo stato iniziale, esegue il grafo e gestisce
il loop human-in-the-loop tramite interrupt/resume di LangGraph.
"""
import uuid
from datetime import date as _date

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from graph.builder import graph
from graph.state import MAX_ITERATIONS
import config.settings  


def _check_todays_posts() -> None:
    """Controlla se ci sono post pianificati per oggi e mostra un promemoria."""
    today = _date.today().isoformat()
    try:
        from kg.kg_manager import KnowledgeGraphManager
        kg = KnowledgeGraphManager()
        posts = kg.get_posts_scheduled_for_date(today)
        if posts:
            W = 62
            print(f"\n{'═' * W}")
            print(f"  PROMEMORIA — oggi ({today}) dovresti pubblicare:")
            for p in posts:
                print(f"  > \"{p['title']}\"  [{p.get('post_type', '')}]")
                print(f"    Topic: {p['topic']}")
            print(f"{'═' * W}")
    except Exception:
        pass 


def _ask_session_params() -> tuple[int, int]:
    """Chiede quanti post pianificare e ogni quanti giorni pubblicarli."""
    print("Quanti post vuoi pianificare? (default 1)")
    while True:
        raw = input("> ").strip()
        if raw == "":
            n = 1
            break
        try:
            n = int(raw)
            if n >= 1:
                break
        except ValueError:
            pass
        print("Inserisci un numero intero maggiore di 0.")

    print("\nOgni quanti giorni vuoi pubblicare un post? (default 7)")
    while True:
        raw = input("> ").strip()
        if raw == "":
            days = 7
            break
        try:
            days = int(raw)
            if days >= 1:
                break
        except ValueError:
            pass
        print("Inserisci un numero intero >= 1.")

    return n, days


def initial_state(user_input: str, posts_per_session: int = 1, days_between_posts: int = 0) -> dict:
    """Stato di partenza con tutti i campi inizializzati."""
    return {
        "messages": [HumanMessage(content=user_input)],
        "user_input": user_input,
        "suggested_topics": [],
        "editorial_plan": [],
        "current_post_index": 0,
        "post_type": "",
        "current_topic": "",
        "topic_kind": "",
        "kg_context": "",
        "retrieved_docs": [],
        "tool_outputs": [],
        "reasoning_trace": [],
        "current_title": "",
        "current_draft": "",
        "citations": [],
        "key_claims": [],
        "user_status": "",
        "user_feedback": "",
        "approved": False,
        "iteration_count": 0,
        "grounding_feedback": [],
        "grounding_retries": 0,
        "grounding_passed": True,
        "posts_per_session": posts_per_session,
        "days_between_posts": days_between_posts,
    }


def _show_draft(data: dict) -> None:
    """Stampa titolo, bozza, claim e fonti in modo leggibile."""
    iteration = data.get("iteration", 1)
    print(f"\n{'=' * 70}")
    print(f"  REVISIONE BOZZA  (iterazione {iteration})")
    print(f"{'=' * 70}")
    print(f"\nTITOLO:  {data.get('title', '')}\n")
    print(data.get("draft", ""))
    claims = data.get("key_claims", [])
    if claims:
        print("\nKEY CLAIMS:")
        for c in claims:
            print(f"  • {c}")

    unsupported = data.get("unsupported_claims", [])
    if unsupported:
        print(f"\n⚠  CLAIM NON SUPPORTATI DALLE FONTI ({len(unsupported)}):")
        for u in unsupported:
            print(f"  ✗ {u.get('claim', '')[:80]}")
            print(f"    → {u.get('explanation', '')[:100]}")
    citations = [s for s in data.get("citations", []) if s]
    if citations:
        print(f"\nFONTI: {', '.join(citations)}")
    print(f"{'=' * 70}")


def _get_user_decision() -> dict:
    """Chiede all'utente di approvare, modificare o rigettare la bozza."""
    print("\nCosa vuoi fare?")
    print("  [A] Approva  — pubblica e aggiorna il Knowledge Graph")
    print("  [M] Modifica — fornisci feedback per migliorare la bozza")
    print("  [R] Rigetta  — rigenera la bozza da capo")
    while True:
        scelta = input("\n> ").strip().lower()
        if scelta == "a":
            return {"action": "approve", "feedback": ""}
        elif scelta == "m":
            feedback = input("Descrivi le modifiche da apportare:\n> ").strip()
            return {"action": "modify", "feedback": feedback}
        elif scelta == "r":
            return {"action": "reject", "feedback": ""}
        else:
            print("Inserisci A, M o R.")


if __name__ == "__main__":
    _check_todays_posts()

    n, days = _ask_session_params()
    state = initial_state(
        "Voglio gestire il mio blog su film e serie TV",
        posts_per_session=n,
        days_between_posts=days,
    )
    # thread_id univoco per ogni sessione; necessario per il checkpointer
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    W = 62
    print("\n" + "═" * W)
    print(f"  BLOG COPILOT  —  {n} post richiesti")
    if days > 0:
        print(f"  Cadenza: ogni {days} giorni")
    print("═" * W)
    graph.invoke(state, config=config)

    while True:
        current = graph.get_state(config)
        if not current.next:
            break  

        interrupt_data = current.tasks[0].interrupts[0].value
        _show_draft(interrupt_data)

        if interrupt_data.get("iteration", 1) >= MAX_ITERATIONS:
            print(
                f"\n*** ULTIMA ITERAZIONE ({MAX_ITERATIONS}/{MAX_ITERATIONS}) ***\n"
                "Se non approvi ora, il processo termina senza aggiornare il Knowledge Graph."
            )

        decision = _get_user_decision()

        graph.invoke(Command(resume=decision), config=config)

    # Riepilogo finale
    final = graph.get_state(config).values
    posts_done = final.get("current_post_index", 0)
    W = 62
    print(f"\n{'═' * W}")
    print(f"  SESSIONE COMPLETATA  —  {posts_done} post preparati")
    print(f"{'═' * W}")
    print(f"  Stato  : {final.get('user_status')}")
    print(f"  Topic  : {final.get('current_topic')}")
    print(f"  Titolo : {final.get('current_title')}")

    reasoning = final.get("reasoning_trace", [])
    if reasoning:
        print(f"\n  TRACE  ({len(reasoning)} passi)")
        print(f"  {'─' * 56}")
        for i, step in enumerate(reasoning, 1):
            thought = step.get("thought", "")[:160]
            action = step.get("action", "—")
            obs = step.get("observation", "")[:130]
            if thought:
                print(f"\n  [{i:2d}]  Thought: {thought}")
            if action:
                print(f"        Action: {action}")
            if obs and action != "no_action":
                print(f"        Observation: {obs}")
