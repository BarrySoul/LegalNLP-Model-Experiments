#Souleymane Barry
#Maël Fauquette
import re
import math
from collections import Counter
from typing import Dict, List, Tuple

def _normalize(text: str) -> str:
    """Normalise le texte : minuscules et espaces multiples réduits."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _tokenize(text: str) -> List[str]:
    """Découpe le texte en tokens sur les espaces après normalisation.

    On conserve la ponctuation attachée pour rester simple et sans dépendances.
    """
    text = _normalize(text)
    return text.split()

def _ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Renvoie la liste des n-grammes à partir de la liste de tokens."""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def rouge_n_f1(pred: str, ref: str, n: int) -> float:
    """Calcule le F1 de ROUGE‑n entre prédiction et référence.

    On considère un seul n-gramme (n) et on utilise la moyenne harmonique
    précision/recall pour obtenir le score F1.
    """
    ptoks = _tokenize(pred)
    rtoks = _tokenize(ref)

    p_ngr = Counter(_ngrams(ptoks, n))
    r_ngr = Counter(_ngrams(rtoks, n))
    if not p_ngr or not r_ngr:
        return 0.0

    overlap = sum((p_ngr & r_ngr).values())
    p_total = sum(p_ngr.values())
    r_total = sum(r_ngr.values())

    precision = overlap / p_total if p_total else 0.0
    recall = overlap / r_total if r_total else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def bleu4(pred: str, ref: str, max_n: int = 4, smooth: float = 1.0) -> float:
    """Calcul d'un BLEU-4 simplifié et sans dépendances externes.

    - précision modifiée pour n-grammes jusqu'à max_n
    - lissage add-k pour éviter log(0)
    - pénalité de brièveté
    """
    ptoks = _tokenize(pred)
    rtoks = _tokenize(ref)

    if len(ptoks) == 0:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        p_ngr = Counter(_ngrams(ptoks, n))
        r_ngr = Counter(_ngrams(rtoks, n))
        if not p_ngr:
            precisions.append(0.0)
            continue
        overlap = sum((p_ngr & r_ngr).values())
        total = sum(p_ngr.values())
        # lissage add-k : on ajoute 'smooth' faux n-grammes corrects et 'smooth' au total
        precisions.append((overlap + smooth) / (total + smooth))

    # moyenne géométrique des précisions
    log_p = sum(math.log(p) for p in precisions if p > 0) / max_n
    geo_mean = math.exp(log_p)

    # pénalité de brièveté : si la prédiction est plus courte que la référence, on pénalise
    pred_len = len(ptoks)
    ref_len = len(rtoks)
    if pred_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - (ref_len / max(pred_len, 1)))

    return bp * geo_mean

def compute_metrics(preds: List[str], refs: List[str]) -> Dict[str, float]:
    """Retourne les métriques ROUGE-1, ROUGE-2 et BLEU moyennées sur un lot.

    Les entrées doivent être des listes de même longueur.
    """
    assert len(preds) == len(refs)
    r1 = [rouge_n_f1(p, r, 1) for p, r in zip(preds, refs)]
    r2 = [rouge_n_f1(p, r, 2) for p, r in zip(preds, refs)]
    b = [bleu4(p, r) for p, r in zip(preds, refs)]
    return {
        "rouge1_f1": sum(r1) / len(r1) if r1 else 0.0,
        "rouge2_f1": sum(r2) / len(r2) if r2 else 0.0,
        "bleu": sum(b) / len(b) if b else 0.0,
    }