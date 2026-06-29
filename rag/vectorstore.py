"""
Costruzione e caricamento del vectorstore (RAG).

Strategia di chunking: Document-aware + Recursive fallback.
  1. MarkdownHeaderTextSplitter — divide sui titoli # e ##, conservando
     il testo del titolo sia nel contenuto che nei metadati del chunk.
     Ogni sezione (## Stile e tecnica, ## Filmografia...) diventa un chunk
     semanticamente coerente.
  2. RecursiveCharacterTextSplitter — suddivide ulteriormente le sezioni
     che superano CHUNK_SIZE, rispettando i confini di paragrafo e frase.

Uso:
  python -m rag.vectorstore      # costruisce l'indice (una tantum)
"""
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma

from config.settings import VECTORSTORE_DIR, get_embeddings

SEED_DIR = Path(__file__).resolve().parent.parent / "data" / "seed_corpus"
COLLECTION = "film_tv_blog"

SCORE_THRESHOLD = 0.3

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

_HEADERS_TO_SPLIT_ON = [
    ("#", "section_h1"),
    ("##", "section_h2"),
]


# ------------------------------------------------------------------ #
# CHUNKING                                                             #
# ------------------------------------------------------------------ #

def _load_seed_documents() -> list[Document]:
    """Carica ogni file .md del seed corpus come un Document con metadati."""
    documents = []
    for path in sorted(SEED_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        documents.append(Document(
            page_content=text,
            metadata={"source": path.name, "title": path.stem.replace("_", " ")},
        ))
    return documents


def _split_document_aware(doc: Document) -> list[Document]:
    """
    Document-aware chunking in due passi:
      1. MarkdownHeaderTextSplitter: divide sui titoli # / ##.
         strip_headers=False → il titolo resta nel testo del chunk
         (es. "## Stile e tecnica\nNolan è..."), e viene anche salvato
         nei metadati come section_h1 / section_h2.
      2. RecursiveCharacterTextSplitter: suddivide le sezioni che
         superano CHUNK_SIZE, rispettando paragrafi e frasi.

    I metadati finali di ogni chunk includono:
      source, title + section_h1, section_h2.
    """
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )
    header_chunks = md_splitter.split_text(doc.page_content)

    for chunk in header_chunks:
        chunk.metadata = {**doc.metadata, **chunk.metadata}

    recursive = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return recursive.split_documents(header_chunks)


def _chunk_all_documents(documents: list[Document]) -> list[Document]:
    """Applica _split_document_aware a tutti i documenti e restituisce i chunk."""
    chunks = []
    for doc in documents:
        chunks.extend(_split_document_aware(doc))
    return chunks


def load_chunks_for_bm25() -> list[Document]:
    """Carica e splitta il seed corpus per il BM25Retriever."""
    documents = _load_seed_documents()
    if not documents:
        return []
    return _chunk_all_documents(documents)


# ------------------------------------------------------------------ #
# INDEX BUILD / LOAD                                                   #
# ------------------------------------------------------------------ #

def build_index():
    """Carica il seed, fa chunking document-aware e salva gli embedding su Chroma."""
    documents = _load_seed_documents()
    if not documents:
        raise RuntimeError(
            f"Nessun documento in {SEED_DIR}. Aggiungi file .md al seed corpus."
        )

    chunks = _chunk_all_documents(documents)

    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=COLLECTION,
        persist_directory=str(VECTORSTORE_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"[RAG] Indicizzati {len(chunks)} chunk da {len(documents)} documenti "
          f"in {VECTORSTORE_DIR}")
    for c in chunks:
        h1 = c.metadata.get("section_h1", "")
        h2 = c.metadata.get("section_h2", "")
        section = f"{h1} > {h2}" if h2 else h1
        print(f"  [{c.metadata['source']}] {section!r:40s} {len(c.page_content)} chars")


def _load_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(VECTORSTORE_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def get_retriever(k: int = 4):
    """Retriever denso con filtro per soglia di similarità coseno (>= SCORE_THRESHOLD)."""
    return _load_vectorstore().as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": SCORE_THRESHOLD},
    )


def get_retriever_no_threshold(k: int = 4):
    """Retriever denso senza soglia — fallback quando la soglia non restituisce risultati."""
    return _load_vectorstore().as_retriever(search_kwargs={"k": k})


if __name__ == "__main__":
    build_index()
