"""Retrieve the k most relevant train examples for a given case.

Ranking signal (no embeddings needed - the templates are near-duplicates within a
modality x body_part bucket):

    same (modality, body_part)          strong bonus
    template_content similarity         primary (matching template => matching field
                                        structure and house rephrasing conventions)
    study_description similarity        tie-breaker
    similar dictation length            mild bonus (comparable abnormality load)
"""
from __future__ import annotations

from difflib import SequenceMatcher
from typing import List

from res_score import normalize


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def _norm_join(s: str) -> str:
    return " ".join(normalize(s))


def score_pair(case: dict, cand: dict) -> float:
    s = 0.0
    if case["modality"] == cand["modality"]:
        s += 2.0
    if case["body_part"] == cand["body_part"]:
        s += 3.0
    tmpl_sim = _ratio(_norm_join(case["template_content"]), _norm_join(cand["template_content"]))
    s += 6.0 * tmpl_sim
    if tmpl_sim > 0.98:
        s += 4.0                       # essentially the same template
    s += 1.5 * _ratio(case["study_description"].lower(), cand["study_description"].lower())
    la, lb = len(case["dictation"]), len(cand["dictation"])
    s += 0.5 * (min(la, lb) / max(la, lb, 1))
    return s


def retrieve(case: dict, pool: List[dict], k: int = 4) -> List[dict]:
    ranked = sorted(pool, key=lambda c: score_pair(case, c), reverse=True)
    return ranked[:k]
