"""
Local re-implementation of the competition's Radiology Edit Score (RES).

Spec: ../EVALUATION.md  (verbatim from the Kaggle Evaluation tab).

This is a best-effort reconstruction. Constants that the host did not fully specify
are collected in `Config` and must be calibrated against the first real leaderboard
result. The goal is high *rank correlation* with the hidden scorer, not identity.

Usage:
    from res_score import score_case, leaderboard_res
    r = score_case(pred_report, ref_report, template_content)   # -> dict, r["res"]
    lb = leaderboard_res([(pred, ref, tmpl), ...])              # -> float (mean RES)
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------------------
# Token classification
# --------------------------------------------------------------------------------------

NEGATION = {
    "no", "not", "without", "negative", "absent", "none", "nor", "neither", "non",
    "unremarkable", "normal", "clear", "patent", "intact", "unchanged",
}
LATERALITY = {
    "left", "right", "bilateral", "bilaterally", "unilateral", "lt", "rt", "bilat",
    "midline", "central", "medial", "lateral", "proximal", "distal", "superior",
    "inferior", "anterior", "posterior",
}
SEVERITY = {
    "mild", "mildly", "moderate", "moderately", "severe", "severely", "minimal",
    "minimally", "marked", "markedly", "trace", "small", "large", "tiny", "extensive",
    "significant", "gross", "subtle", "prominent", "borderline", "slight", "slightly",
    "grade", "low", "high",
}
ACUITY = {
    "acute", "chronic", "subacute", "old", "new", "healing", "healed", "interval",
    "age", "aged", "remote", "recent", "progressive", "stable",
}
UNITS = {"mm", "cm", "ml", "cc", "mmhg", "hu", "cm2", "mm2", "mm3", "cm3"}

CRITICAL_WORDS = NEGATION | LATERALITY | SEVERITY | ACUITY | UNITS

FUNCTION_WORDS = {
    "the", "a", "an", "and", "or", "of", "with", "is", "are", "was", "were", "be",
    "been", "being", "to", "in", "on", "at", "as", "by", "for", "that", "this",
    "these", "those", "there", "it", "its", "from", "than", "then", "which", "has",
    "have", "had", "do", "does", "did", "into", "within", "also", "but", "if",
}

_NUM_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")


@dataclass
class Config:
    w_critical: float = 4.0
    w_content: float = 2.0
    w_function: float = 0.25
    # substitution cost for two unequal tokens
    sub_mode: str = "max"            # "max" | "avg" | "sum"
    # field weight when reference field differs from template
    w_field_changed: float = 3.0
    w_field_same: float = 1.0
    # reference label not present in template_content -> treat as changed
    new_ref_field_changed: bool = True
    # penalty book-keeping for our labels not in reference / stray unlabelled content
    w_unexpected_field: float = 1.0
    # RES weighting
    w_findings: float = 0.65
    w_impression: float = 0.35


DEFAULT = Config()


def token_weight(tok: str, cfg: Config = DEFAULT) -> float:
    if _NUM_RE.match(tok):
        return cfg.w_critical
    if tok in CRITICAL_WORDS:
        return cfg.w_critical
    if tok in FUNCTION_WORDS:
        return cfg.w_function
    return cfg.w_content


# --------------------------------------------------------------------------------------
# Normalization  ->  token list
# --------------------------------------------------------------------------------------

_LIST_MARKER_RE = re.compile(r"^[ \t]*(?:\d+[.)]|[-*•])[ \t]+", re.MULTILINE)
_UNIT_SUBS = [
    (re.compile(r"\bmillimet(?:er|re)s?\b"), "mm"),
    (re.compile(r"\bcentimet(?:er|re)s?\b"), "cm"),
    (re.compile(r"\bmillilit(?:er|re)s?\b"), "ml"),
]
_LETTER_LETTER_HYPHEN = re.compile(r"(?<=[a-z])-(?=[a-z])")
_LETTER_NUM = re.compile(r"(?<=[a-z])(?=\d)")
_NUM_LETTER = re.compile(r"(?<=\d)(?=[a-z])")
_KEEP_SIGNED = re.compile(r"(?:(?<=\s)|^)([+-])(?=\d)")
_PUNCT = re.compile(r"[^\w\s]")


def normalize(text: str) -> List[str]:
    if not text:
        return []
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = _LIST_MARKER_RE.sub("", text)
    text = text.replace("\n", " ")
    for rx, rep in _UNIT_SUBS:
        text = rx.sub(rep, text)
    text = _LETTER_LETTER_HYPHEN.sub("", text)          # air-space -> airspace
    text = _LETTER_NUM.sub(" ", text)                   # 5mm -> 5 mm
    text = _NUM_LETTER.sub(" ", text)
    # protect signed measurements with a sentinel, strip other punctuation, restore
    text = _KEEP_SIGNED.sub(lambda m: "\x00" if m.group(1) == "-" else "\x01", text)
    text = _PUNCT.sub(" ", text)
    text = text.replace("\x00", "-").replace("\x01", "+")
    toks = text.split()
    out = []
    for t in toks:
        if _NUM_RE.match(t):
            out.append(t)
        else:
            t = t.strip("+-")
            if t:
                out.append(t)
    return out


# --------------------------------------------------------------------------------------
# Report parsing  ->  {label: text}, impression, unlabelled bucket
# --------------------------------------------------------------------------------------

# a "label" line: short phrase (any case) then a colon at line start.
# templates use both "BONES:" and "Bones:" / "Hip Joints:"; references always UPPERCASE them.
_LABEL_RE = re.compile(r"^[ \t]*([A-Za-z][A-Za-z0-9 /,()&'\-]{0,40}):[ \t]*(\S.*|)$")


def _is_label(m) -> bool:
    """Guard against sentence lines that happen to contain an early colon."""
    lbl = m.group(1).strip()
    return "." not in lbl and len(lbl.split()) <= 7


@dataclass
class ParsedReport:
    fields: "Dict[str, str]" = field(default_factory=dict)   # label -> text (in order)
    order: List[str] = field(default_factory=list)
    unlabelled: str = ""                                     # stray FINDINGS content
    impression: str = ""


def _split_sections(report: str) -> Tuple[str, str]:
    """Return (findings_block, impression_block)."""
    lines = report.splitlines()
    fi = ii = None
    for i, ln in enumerate(lines):
        s = ln.strip().upper()
        if fi is None and s.startswith("FINDINGS:"):
            fi = i
        elif s.startswith("IMPRESSION:"):
            ii = i
            break
    if fi is None:
        fi = -1
    findings = "\n".join(lines[fi + 1: ii if ii is not None else len(lines)])
    # keep any text after "FINDINGS:" on the same line
    if fi >= 0:
        head = lines[fi].split(":", 1)[1] if ":" in lines[fi] else ""
        findings = (head + "\n" + findings).strip("\n")
    impression = ""
    if ii is not None:
        imp_head = lines[ii].split(":", 1)[1] if ":" in lines[ii] else ""
        impression = (imp_head + "\n" + "\n".join(lines[ii + 1:])).strip()
    return findings, impression


def parse_report(report: str) -> ParsedReport:
    findings_block, impression = _split_sections(report or "")
    pr = ParsedReport(impression=impression)
    current = None
    for raw in findings_block.splitlines():
        m = _LABEL_RE.match(raw)
        if m and _is_label(m):
            label = re.sub(r"\s+", " ", m.group(1).strip()).upper()
            current = label
            if label not in pr.fields:
                pr.fields[label] = m.group(2).strip()
                pr.order.append(label)
            else:  # duplicate label - append
                pr.fields[label] += " " + m.group(2).strip()
            continue
        if raw.strip() == "":
            current = None            # blank line ends a field; trailing text -> unlabelled
            continue
        if current is None:
            pr.unlabelled += (" " if pr.unlabelled else "") + raw.strip()
        else:
            pr.fields[current] += (" " if pr.fields[current] else "") + raw.strip()
    return pr


# --------------------------------------------------------------------------------------
# Weighted ordered Levenshtein
# --------------------------------------------------------------------------------------

def _weights(toks: List[str], cfg: Config) -> List[float]:
    return [token_weight(t, cfg) for t in toks]


def weighted_word_edit(ref: List[str], sub: List[str], cfg: Config = DEFAULT) -> float:
    """Return per-field score in [0, 1]  (weighted edit cost / max total weight, capped)."""
    rw = _weights(ref, cfg)
    sw = _weights(sub, cfg)
    tot_ref, tot_sub = sum(rw), sum(sw)
    denom = max(tot_ref, tot_sub)
    if denom == 0:
        return 0.0
    n, m = len(ref), len(sub)
    # dp over rows
    prev = [0.0] * (m + 1)
    for j in range(1, m + 1):
        prev[j] = prev[j - 1] + sw[j - 1]
    for i in range(1, n + 1):
        cur = [0.0] * (m + 1)
        cur[0] = prev[0] + rw[i - 1]
        ri = ref[i - 1]
        for j in range(1, m + 1):
            if ri == sub[j - 1]:
                sub_cost = 0.0
            elif cfg.sub_mode == "avg":
                sub_cost = (rw[i - 1] + sw[j - 1]) / 2
            elif cfg.sub_mode == "sum":
                sub_cost = rw[i - 1] + sw[j - 1]
            else:
                sub_cost = max(rw[i - 1], sw[j - 1])
            cur[j] = min(
                prev[j] + rw[i - 1],        # delete ref token
                cur[j - 1] + sw[j - 1],     # insert sub token
                prev[j - 1] + sub_cost,     # substitute
            )
        prev = cur
    return min(prev[m] / denom, 1.0)


# --------------------------------------------------------------------------------------
# Case scoring
# --------------------------------------------------------------------------------------

def _norm_key(text: str) -> Tuple[str, ...]:
    return tuple(normalize(text))


def score_findings(pred: ParsedReport, ref: ParsedReport, tmpl: ParsedReport,
                   cfg: Config = DEFAULT) -> Tuple[float, list]:
    num = 0.0
    den = 0.0
    detail = []
    ref_labels = set(ref.order)

    for label in ref.order:
        ref_txt = ref.fields.get(label, "")
        tmpl_txt = tmpl.fields.get(label, None)
        if tmpl_txt is None:
            fw = cfg.w_field_changed if cfg.new_ref_field_changed else cfg.w_field_same
        else:
            fw = cfg.w_field_same if _norm_key(ref_txt) == _norm_key(tmpl_txt) else cfg.w_field_changed
        pred_txt = pred.fields.get(label, "")
        s = weighted_word_edit(normalize(ref_txt), normalize(pred_txt), cfg)
        num += fw * s
        den += fw
        detail.append((label, round(fw, 2), round(s, 4), label not in pred.fields))

    # reference unlabelled content -> pseudo-field, weight = changed (it is never in template)
    if _norm_key(ref.unlabelled):
        fw = cfg.w_field_changed
        s = weighted_word_edit(normalize(ref.unlabelled), normalize(pred.unlabelled), cfg)
        num += fw * s
        den += fw
        detail.append(("__UNLABELLED__", fw, round(s, 4), False))
    elif _norm_key(pred.unlabelled):
        num += cfg.w_unexpected_field * 1.0
        den += cfg.w_unexpected_field
        detail.append(("__UNLABELLED_EXTRA__", cfg.w_unexpected_field, 1.0, False))

    # our labels not expected by reference -> extra-content penalty
    for label in pred.order:
        if label not in ref_labels and _norm_key(pred.fields.get(label, "")):
            num += cfg.w_unexpected_field * 1.0
            den += cfg.w_unexpected_field
            detail.append((f"__EXTRA__:{label}", cfg.w_unexpected_field, 1.0, False))

    F = num / den if den else 0.0
    return F, detail


def score_case(pred_report: str, ref_report: str, template_content: str,
               cfg: Config = DEFAULT) -> dict:
    pred = parse_report(pred_report)
    ref = parse_report(ref_report)
    tmpl = parse_report(template_content)

    F, detail = score_findings(pred, ref, tmpl, cfg)
    I = weighted_word_edit(normalize(ref.impression), normalize(pred.impression), cfg)
    res = cfg.w_findings * F + cfg.w_impression * I
    return {"res": res, "F": F, "I": I, "fields": detail}


def leaderboard_res(triples, cfg: Config = DEFAULT) -> float:
    if not triples:
        return 0.0
    return sum(score_case(p, r, t, cfg)["res"] for p, r, t in triples) / len(triples)


# --------------------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    import csv
    import os
    import statistics

    here = os.path.dirname(__file__)
    train = os.path.join(here, "..", "data", "train.csv")
    rows = list(csv.DictReader(open(train, encoding="utf-8-sig")))

    # 1. reference vs itself -> 0.0
    zero = [score_case(r["report"], r["report"], r["template_content"])["res"] for r in rows[:50]]
    print(f"identity check: max RES over 50 = {max(zero):.4f}  (expect 0.0)")

    # 2. baseline B0: predict template verbatim
    b0 = [score_case(r["template_content"], r["report"], r["template_content"])["res"] for r in rows]
    print(f"B0 (emit template)        mean RES = {statistics.mean(b0):.4f}  median = {statistics.median(b0):.4f}")

    # by modality
    from collections import defaultdict
    bucket = defaultdict(list)
    for r, s in zip(rows, b0):
        bucket[r["modality"]].append(s)
    for k in sorted(bucket):
        print(f"    {k:5s} n={len(bucket[k]):3d}  B0 mean RES = {statistics.mean(bucket[k]):.4f}")
