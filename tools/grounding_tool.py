"""
Claim grounding verifier: verifica se un'affermazione è supportata da un testo di evidenza.

Usa xlm-roberta-base fine-tunato con strategia ibrida: truncation sul testo intero per il label,
poi ricerca post-hoc della frase più rilevante come spiegazione estrattiva.

Output format: "GROUNDED|spiegazione" | "PARTIAL|spiegazione" | "NOT_GROUNDED|spiegazione"
"""
import re
from pathlib import Path

from langchain_core.tools import tool

# ------------------------------------------------------------------ #
# Configurazione                                                       #
# ------------------------------------------------------------------ #
_HF_MODEL_ID = "elenamonticchio/verify-claim-xlm-roberta-config-search"
_BASE_MODEL  = "xlm-roberta-base"
_MAX_LENGTH  = 256
_LABEL_MAP   = {0: "GROUNDED", 1: "PARTIAL", 2: "NOT_GROUNDED"}

_ft_model      = None
_ft_tokenizer  = None
_ft_ready      = False  


# ------------------------------------------------------------------ #
# Caricamento del modello fine-tunato                            #
# ------------------------------------------------------------------ #
def _load_finetuned() -> None:
    """
    Carica xlm-roberta fine-tunato da HuggingFace Hub (_HF_MODEL_ID).
    """
    global _ft_model, _ft_tokenizer, _ft_ready
    if _ft_ready:
        return
    _ft_ready = True

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        print(f"[verify_claim] carico modello da HF Hub ({_HF_MODEL_ID}) ...")
        _ft_tokenizer = AutoTokenizer.from_pretrained(_BASE_MODEL)
        _ft_model     = AutoModelForSequenceClassification.from_pretrained(_HF_MODEL_ID)
        _ft_model.eval()
        print("[verify_claim] modello pronto")
    except Exception as e:
        print(f"[verify_claim] errore caricamento modello: {e}")
        _ft_model     = None
        _ft_tokenizer = None


# ------------------------------------------------------------------ #
# Inferenza con il modello fine-tunato (strategia ibrida)             #
# ------------------------------------------------------------------ #
def _split_sentences(text: str, min_len: int = 15) -> list[str]:
    """Divide un testo in frasi, filtrando frammenti troppo corti."""
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in raw if len(s.strip()) >= min_len]


def _classify_pair(claim: str, text: str) -> tuple[int, list]:
    """Classifica una singola coppia (claim, text); restituisce (pred_id, probs)."""
    import torch

    device = next(_ft_model.parameters()).device
    enc    = _ft_tokenizer(
        claim, text,
        max_length=_MAX_LENGTH, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    with torch.no_grad():
        logits = _ft_model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
        ).logits
    probs = logits.softmax(dim=-1)[0].cpu()
    return probs.argmax().item(), probs.tolist()


def _predict_finetuned(claim: str, evidence: str) -> str:
    """
    Strategia ibrida (truncation + explanation post-hoc):
      - predizione sul testo intero (truncation) per accuracy massima
      - explanation: frase dell'evidence con confidenza più alta per il label predetto
    """
    pred_id, _ = _classify_pair(claim, evidence)

    sentences = _split_sentences(evidence) or [evidence[:300]]
    best_sentence, best_conf = sentences[0], 0.0
    for sent in sentences:
        _, probs = _classify_pair(claim, sent)
        if probs[pred_id] > best_conf:
            best_conf     = probs[pred_id]
            best_sentence = sent

    verdict = _LABEL_MAP[pred_id]
    if pred_id == 0:
        explanation = f"Supportato da: \"{best_sentence[:110]}\""
    elif pred_id == 1:
        explanation = f"Parzialmente suggerito da: \"{best_sentence[:100]}\""
    else:
        explanation = f"Contraddetto da: \"{best_sentence[:100]}\""

    return f"{verdict}|{explanation}"


# ------------------------------------------------------------------ #
# Tool LangChain                                                       #
# ------------------------------------------------------------------ #
@tool
def verify_claim(claim: str, evidence: str) -> str:
    """
    Verifica se un claim è supportato da un testo di evidenza.
    Restituisce GROUNDED, PARTIAL o NOT_GROUNDED con una spiegazione breve.
    """
    evidence = re.sub(r'^\[DOC \d+[^\]]*\]\s*', '', evidence.strip())
    _load_finetuned()
    if _ft_model is not None:
        return _predict_finetuned(claim, evidence)
    raise RuntimeError("[verify_claim] modello fine-tunato non trovato")
