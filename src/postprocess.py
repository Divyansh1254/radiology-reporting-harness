"""Deterministic post-processing: force the model output onto the template's exact skeleton.

Guarantees, regardless of what the model returned:
  - starts with `FINDINGS:`, contains `IMPRESSION:`
  - every template field label reproduced verbatim, in template order
  - template blank-line layout preserved
  - a field the model did not fill falls back to the template's normal sentence
  - the model's unlabelled trailing findings (if any) kept just before IMPRESSION
"""
from __future__ import annotations

import re

from res_score import _LABEL_RE, _is_label, parse_report


def _norm_label(lbl: str) -> str:
    return re.sub(r"\s+", " ", lbl.strip()).upper()


def reskeleton(template_content: str, model_report: str) -> str:
    tmpl_lines = template_content.splitlines()
    mp = parse_report(model_report)
    model_fields = {_norm_label(k): v.strip() for k, v in mp.fields.items()}

    # locate FINDINGS: / IMPRESSION: in the template
    f_idx = i_idx = None
    for i, ln in enumerate(tmpl_lines):
        s = ln.strip().upper()
        if f_idx is None and s.startswith("FINDINGS:"):
            f_idx = i
        elif s.startswith("IMPRESSION:"):
            i_idx = i
            break
    if f_idx is None:
        # template has no explicit FINDINGS header - just trust the model output
        return model_report.strip()

    out = ["FINDINGS:"]
    body_end = i_idx if i_idx is not None else len(tmpl_lines)
    for ln in tmpl_lines[f_idx + 1: body_end]:
        m = _LABEL_RE.match(ln)
        if m and _is_label(m):
            label = _norm_label(m.group(1))          # references always UPPERCASE labels
            tmpl_text = m.group(2).strip()
            new_text = model_fields.get(label, tmpl_text)
            out.append(f"{label}: {new_text}".rstrip())
        else:
            out.append(ln.rstrip())            # blank lines / stray template lines

    # model's unlabelled trailing findings, if any
    if mp.unlabelled.strip():
        if out and out[-1].strip():
            out.append("")
        out.append(mp.unlabelled.strip())

    # IMPRESSION
    imp = mp.impression.strip()
    if not imp and i_idx is not None:
        imp = "\n".join(tmpl_lines[i_idx + 1:]).strip()
    out += ["", "IMPRESSION:", imp]

    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text
