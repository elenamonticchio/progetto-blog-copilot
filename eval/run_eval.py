"""
Runner di valutazione: esegue 4 scenari predefiniti attraverso il pipeline
research → verify_and_select → draft → verify_grounding, poi valuta i risultati.

Ogni scenario viene wrappato con @traceable per creare un parent run su LangSmith.
Dopo l'esecuzione, i parametri di osservabilità (token, costo, latenza, LLM calls)
vengono estratti dal run LangSmith corrispondente.

Uso:
  python -m eval.run_eval
"""
from __future__ import annotations

import os
import time

from eval.evaluator import EvalReport, LangSmithReport, run_full_eval

# ══════════════════════════════════════════════════════════════════════════════
# SCENARI
# ══════════════════════════════════════════════════════════════════════════════

SCENARIOS = [
    {"topic": "Oppenheimer",                  "topic_kind": "movie",    "post_type": "review"},
    {"topic": "fantascienza",                 "topic_kind": "concept",  "post_type": "how-to"},
    {"topic": "festival cinema Venezia 2026", "topic_kind": "event",    "post_type": "events"},
    {"topic": "Adolescence",                  "topic_kind": "tv_series","post_type": "review"},
]

# ══════════════════════════════════════════════════════════════════════════════
# STATE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def _build_initial_state(topic: str, topic_kind: str, post_type: str) -> dict:
    return {
        "messages": [],
        "user_input": topic,
        "suggested_topics": [],
        "editorial_plan": [],
        "current_post_index": 0,
        "post_type": post_type,
        "current_topic": topic,
        "topic_kind": topic_kind,
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
        "grounding_passed": False,
        "posts_per_session": 1,
        "days_between_posts": 0,
        "tool_outputs_offset": 0,
    }


def _merge(state: dict, updates: dict) -> dict:
    """
    Fonde gli aggiornamenti restituiti da un nodo nello stato corrente,
    replicando i reducer di LangGraph:
      - tool_outputs, reasoning_trace  → operator.add (append)
      - messages                       → add_messages (append)
      - tutti gli altri campi          → overwrite
    """
    result = dict(state)
    for k, v in updates.items():
        if k in ("tool_outputs", "reasoning_trace"):
            result[k] = result.get(k, []) + (v or [])
        elif k == "messages":
            result[k] = list(result.get(k, [])) + list(v or [])
        else:
            result[k] = v
    return result


# ══════════════════════════════════════════════════════════════════════════════
# LANGSMITH 
# ══════════════════════════════════════════════════════════════════════════════

_LANGSMITH_WAIT_SECONDS = 4 


def _fetch_langsmith_stats(run_id: str) -> LangSmithReport:
    """
    Interroga LangSmith per estrarre i parametri di osservabilità del run.
    """
    api_key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
    if not api_key:
        return LangSmithReport(run_id=run_id, available=False, notes="LANGSMITH_API_KEY non configurata")

    try:
        from langsmith import Client
        client = Client()

        print(f"  [LangSmith] attendo upload trace ({_LANGSMITH_WAIT_SECONDS}s)...")
        time.sleep(_LANGSMITH_WAIT_SECONDS)

        run = client.read_run(run_id)

        llm_calls = sum(
            1 for _ in client.list_runs(
                trace_id=run_id,
                run_type="llm",
                is_root=False,
            )
        )

        latency = 0.0
        if run.start_time and run.end_time:
            latency = (run.end_time - run.start_time).total_seconds()

        return LangSmithReport(
            run_id=run_id,
            available=True,
            total_tokens=run.total_tokens or 0,
            prompt_tokens=run.prompt_tokens or 0,
            completion_tokens=run.completion_tokens or 0,
            total_cost=run.total_cost or 0.0,
            latency_seconds=latency,
            llm_calls=llm_calls,
            notes="",
        )

    except Exception as e:
        return LangSmithReport(run_id=run_id, available=False, notes=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# ESECUZIONE SCENARIO
# ══════════════════════════════════════════════════════════════════════════════

def run_scenario(topic: str, topic_kind: str, post_type: str) -> EvalReport:
    """
    Esegue research → verify_and_select → draft → verify_grounding per un
    singolo scenario, wrappato in un @traceable LangSmith per il parent run.
    """
    from agents.nodes import research, verify_and_select, draft, verify_grounding
    from langsmith.run_helpers import traceable, get_current_run_tree

    run_id_holder: list[str | None] = [None]

    @traceable(name=f"eval — {topic} [{post_type}]", run_type="chain", tags=["eval"])
    def _run_traced() -> dict:
        rt = get_current_run_tree()
        if rt:
            run_id_holder[0] = str(rt.id)

        state = _build_initial_state(topic, topic_kind, post_type)

        print(f"  [1/4] research...")
        state = _merge(state, research(state))

        print(f"  [2/4] verify_and_select...")
        state = _merge(state, verify_and_select(state))

        print(f"  [3/4] draft...")
        state = _merge(state, draft(state))

        print(f"  [4/4] verify_grounding...")
        state = _merge(state, verify_grounding(state))

        return state

    final_state = _run_traced()
    report = run_full_eval(final_state)

    # Fetch osservabilità LangSmith 
    if run_id_holder[0]:
        print(f"  [LangSmith] run_id: {run_id_holder[0]}")
        report.langsmith = _fetch_langsmith_stats(run_id_holder[0])
    else:
        report.langsmith = LangSmithReport(run_id=None, available=False, notes="run_id non disponibile")

    return report


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

_W = 66


def _print_report(report: EvalReport) -> None:
    print(f"\n{'═' * _W}")
    print(f"  REPORT — {report.topic}  [{report.post_type}]")
    print(f"{'═' * _W}")

    # Sources
    s = report.sources
    print(f"\n  FONTI  {s.total_docs} doc totali  "
          f"(locali: {s.local_docs}  web: {s.web_docs}  tool: {s.structured_tool_docs})")
    diversity = "OK" if s.diversity_ok else "BASSA"
    print(f"    diversità: {diversity}  —  {s.notes}")

    # Failure cases
    print(f"\n  FAILURE CASES  ", end="")
    if report.failures:
        print(f"({len(report.failures)} rilevati)")
        for fc in report.failures:
            print(f"    [{fc.code}] {fc.description}")
            print(f"           {fc.detail[:100]}")
    else:
        print("nessuno rilevato ✓")

    # Quality
    q = report.quality
    print(f"\n  QUALITÀ  {q.score:.1f}/10")
    print(f"    {q.notes[:130]}")

    # LangSmith
    ls = report.langsmith
    if ls and ls.available:
        print(f"\n  LANGSMITH  run_id: {ls.run_id}")
        print(f"    token:    {ls.total_tokens:>6}  "
              f"(prompt: {ls.prompt_tokens}  completion: {ls.completion_tokens})")
        print(f"    costo:   ${ls.total_cost:.4f}")
        print(f"    latenza:  {ls.latency_seconds:.1f}s")
        print(f"    LLM calls: {ls.llm_calls}")
    elif ls:
        print(f"\n  LANGSMITH  non disponibile — {ls.notes}")

    print()


def _print_summary(reports: list[EvalReport]) -> None:
    print(f"\n{'═' * _W}")
    print(f"  RIEPILOGO  {len(reports)} scenari")
    print(f"{'─' * _W}")
    print(f"  {'Topic':<32}  {'Tipo':<10}  {'Qual':>5}  {'Token':>7}  {'Costo':>7}  FC")
    print(f"  {'─'*32}  {'─'*10}  {'─'*5}  {'─'*7}  {'─'*7}  {'─'*2}")
    for r in reports:
        ls = r.langsmith
        tok = str(ls.total_tokens) if ls and ls.available else "n/d"
        cost = f"${ls.total_cost:.4f}" if ls and ls.available else "n/d"
        print(
            f"  {r.topic:<32}  {r.post_type:<10}  "
            f"{r.quality.score:>4.1f}  "
            f"{tok:>7}  {cost:>7}  "
            f"{len(r.failures)}"
        )
    print(f"{'═' * _W}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    reports: list[EvalReport] = []
    for s in SCENARIOS:
        label = f"{s['topic']}  [{s['post_type']}]"
        print(f"\n{'─' * _W}")
        print(f"  Scenario: {label}")
        print(f"{'─' * _W}")
        report = run_scenario(
            topic=s["topic"],
            topic_kind=s["topic_kind"],
            post_type=s["post_type"],
        )
        reports.append(report)
        _print_report(report)

    _print_summary(reports)


if __name__ == "__main__":
    main()
