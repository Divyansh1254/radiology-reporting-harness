"""End-to-end: case (+ retrieval pool) -> final report string."""
from __future__ import annotations

from typing import List

from exemplars import retrieve
from llm import complete
from postprocess import reskeleton
from prompt import build_messages


def generate_report(case: dict, pool: List[dict], *, k: int = 4,
                    model: str | None = None, effort: str | None = None,
                    postprocess: bool = True, use_cache: bool = True) -> dict:
    exemplars = retrieve(case, pool, k=k)
    system, messages = build_messages(case, exemplars)
    rec = complete(system, messages, model=model, effort=effort, use_cache=use_cache)
    raw = rec["text"]
    report = reskeleton(case["template_content"], raw) if postprocess else raw
    return {
        "case_id": case["case_id"],
        "report": report,
        "raw": raw,
        "exemplar_ids": [e["case_id"] for e in exemplars],
        "usage": rec["usage"],
        "model": rec["model"],
        "refusal": rec.get("refusal", False),
    }
