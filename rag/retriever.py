"""
Logica di recupero K-RAG — pipeline completa:
  1. Espansione query con il Knowledge Graph (K-RAG)
  2. Hybrid retrieval: BM25 (keyword) + Chroma con soglia (semantico)
  3. Reranking con FlashrankRerank (cross-encoder leggero)
  4. Self-RAG grading: filtro binario di rilevanza con LLM

"""
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from config.settings import get_llm
from kg.kg_manager import KnowledgeGraphManager
from rag.vectorstore import (
    get_retriever,
    get_retriever_no_threshold,
    load_chunks_for_bm25,
)

_CANDIDATE_K = 8
_RERANK_TOP_N = 4


class GradeDocuments(BaseModel):
    """Valutazione binaria di rilevanza di un documento (self-RAG)."""
    binary_score: str = Field(description="'yes' se rilevante, 'no' altrimenti")


GRADE_PROMPT = (
    "Sei un valutatore della rilevanza di un documento per la scrittura di un articolo di blog.\n\n"
    "Documento:\n{context}\n\n"
    "Argomento dell'articolo: {topic}\n\n"
    "Rispondi 'yes' se il documento contiene informazioni utili per scrivere un articolo su questo argomento "
    "(incluso contesto, approfondimenti correlati o materiale di supporto), 'no' se è del tutto irrilevante."
)


# ------------------------------------------------------------------ #
# 1. HYBRID RETRIEVER (BM25 + Chroma con soglia)                      #
# ------------------------------------------------------------------ #

def _build_hybrid_retriever(k: int):
    """
    Combina BM25 (ricerca lessicale/keyword) e Chroma con soglia (ricerca semantica)
    tramite EnsembleRetriever.

    BM25 gestisce efficacemente nomi propri e titoli esatti;
    Chroma cattura la similarità semantica.
    """
    from langchain_community.retrievers import BM25Retriever
    from langchain_classic.retrievers import EnsembleRetriever

    chunks = load_chunks_for_bm25()

    if not chunks:
        return get_retriever_no_threshold(k=k)

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = k

    chroma = get_retriever(k=k)  

    return EnsembleRetriever(
        retrievers=[bm25, chroma],
        weights=[0.4, 0.6],  
    )


# ------------------------------------------------------------------ #
# 2. RERANKING (FlashrankRerank)                                       #
# ------------------------------------------------------------------ #

def _rerank(docs: list[Document], query: str, top_n: int) -> list[Document]:
    """
    Riordina i documenti candidati con FlashrankRerank e restituisce i top_n più rilevanti.
    """
    if not docs:
        return docs
    try:
        from langchain_community.document_compressors import FlashrankRerank
        reranker = FlashrankRerank(top_n=top_n)
        reranked = reranker.compress_documents(docs, query)
        print(f"   [K-RAG] reranking: {len(docs)} candidati → top {len(reranked)}")
        return list(reranked)
    except Exception as e:
        print(f"   [K-RAG] reranking non disponibile ({e}), uso ordine originale")
        return docs[:top_n]


# ------------------------------------------------------------------ #
# 3. PIPELINE PRINCIPALE                                               #
# ------------------------------------------------------------------ #

def krag_retrieve(
    topic: str,
    k: int = 4,
    grade: bool = True,
    trace_collector: list | None = None,
) -> list[Document]:
    """
    Pipeline K-RAG completa:
      1. Espande la query con topic correlati e claim passati dal KG
      2. Hybrid retrieval su _CANDIDATE_K documenti
      3. Fallback senza soglia se la hybrid restituisce troppo poco
      4. Reranking con FlashrankRerank → top _RERANK_TOP_N
      5. Self-RAG grading: scarta i documenti non rilevanti

    """
    kg = KnowledgeGraphManager()
    expanded_query = kg.expand_query_for_rag(topic)
    print(f"   [K-RAG] query espansa: {expanded_query}")

    # --- 1. Hybrid retrieval ---
    retriever = _build_hybrid_retriever(k=_CANDIDATE_K)
    docs = retriever.invoke(expanded_query)
    print(f"   [K-RAG] hybrid retrieval: {len(docs)} candidati")

    # --- Fallback: se la soglia di similarità ha scartato tutto ---
    if not docs:
        print("   [K-RAG] nessun risultato sopra soglia, fallback senza threshold")
        fallback = get_retriever_no_threshold(k=_CANDIDATE_K)
        docs = fallback.invoke(expanded_query)

    if not docs:
        return []

    # --- 2. Reranking ---
    docs = _rerank(docs, expanded_query, top_n=_RERANK_TOP_N)

    if not grade or not docs:
        return docs

    # --- 3. Self-RAG grading  ---
    grader = get_llm(temperature=0).with_structured_output(GradeDocuments)
    relevant: list[Document] = []
    for doc in docs:
        prompt = GRADE_PROMPT.format(context=doc.page_content[:1500], topic=topic)
        verdict = grader.invoke(prompt)
        if verdict.binary_score.strip().lower() == "yes":
            relevant.append(doc)

    if not relevant:
        msg = f"Self-RAG grading ha scartato tutti i {len(docs)} documenti — nessun doc locale disponibile"
        print(f"   [K-RAG] {msg}")
        if trace_collector is not None:
            trace_collector.append({
                "thought": "Il grader ha scartato tutti i documenti recuperati.",
                "action": "self_rag_grading_all_rejected",
                "observation": msg,
            })
        return []
    kept = relevant

    print(f"   [K-RAG] dopo grading: {len(kept)} documenti finali")
    return kept


def format_citations(docs: list[Document]) -> list[dict]:
    """Estrae le fonti uniche dai metadati dei documenti recuperati."""
    seen, citations = set(), []
    for doc in docs:
        source = doc.metadata.get("source", "sconosciuta")
        if source not in seen:
            seen.add(source)
            citations.append({
                "title": doc.metadata.get("title", source),
                "url": doc.metadata.get("url", ""),
                "source": source,
            })
    return citations
