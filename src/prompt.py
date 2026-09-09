"""Prompt construction: system guide + retrieved few-shot exemplars + target case."""
from __future__ import annotations

import os
from typing import List

HERE = os.path.dirname(__file__)
HOUSE_STYLE = open(os.path.join(HERE, "house_style.md"), encoding="utf-8").read()

SYSTEM = f"""You are a radiology report editor. You are given a normal report TEMPLATE and a \
radiologist's telegraphic DICTATION. Produce the final report by merging the dictation's findings \
into the template.

You are scored by an edit-distance metric (RES) that rewards reproducing the reference report's \
exact wording, field routing and word order, and heavily penalises invented content, omitted \
findings, wrong-field routing and unnecessary rewriting of normal text.

{HOUSE_STYLE}

Output ONLY the final report, starting with `FINDINGS:` and containing `IMPRESSION:`. No preamble, \
no commentary, no code fences."""

_RULES = (
    "Reproduce every template field label in order (UPPERCASE). Copy every field the dictation "
    "does not mention, unchanged. Merge each dictated finding into the field whose anatomy it "
    "names. Omit nothing from the dictation; add nothing not in the dictation or template. Do not "
    "copy anything from the examples."
)


def _case_block(r: dict, with_report: bool) -> str:
    parts = [
        f"MODALITY: {r['modality']}   BODY PART: {r['body_part']}   STUDY: {r['study_description']}",
        f"PATIENT: {r['patient_age_band']} {r['patient_sex']}",
        "",
        "TEMPLATE:",
        r["template_content"].strip(),
        "",
        "DICTATION:",
        r["dictation"].strip(),
    ]
    block = "\n".join(parts)
    block += "\n\nFINAL REPORT:\n" + (r["report"].strip() if with_report else "")
    return block


def build_messages(case: dict, exemplars: List[dict]):
    """Return (system, messages) for client.messages.create."""
    chunks = [
        "## HOUSE-STYLE EXAMPLES (different patients — for wording and routing only, never copy "
        "their findings)\n"
    ]
    for i, ex in enumerate(exemplars, 1):
        chunks.append(f"----- EXAMPLE {i} -----\n{_case_block(ex, with_report=True)}")
    chunks.append(
        "## YOUR CASE\n\n" + _case_block(case, with_report=False).rstrip()
        + "\n\nWrite the FINAL REPORT now. " + _RULES
    )
    return SYSTEM, [{"role": "user", "content": "\n\n".join(chunks)}]
