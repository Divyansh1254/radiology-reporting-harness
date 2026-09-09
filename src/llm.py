"""Thin Anthropic wrapper with on-disk response caching (keyed by model + prompt hash)."""
from __future__ import annotations

import hashlib
import json
import os
import time

HERE = os.path.dirname(__file__)
CACHE_DIR = os.path.join(HERE, "..", "experiments", ".llm_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

MODEL = os.environ.get("RRH_MODEL", "claude-sonnet-5")
EFFORT = os.environ.get("RRH_EFFORT", "medium")   # low | medium | high | xhigh | max

_client = None


def _get_client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic(max_retries=4)
    return _client


def _key(model: str, system: str, messages: list, effort: str) -> str:
    blob = json.dumps({"m": model, "s": system, "u": messages, "e": effort},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def complete(system: str, messages: list, *, model: str | None = None,
             effort: str | None = None, max_tokens: int = 4000,
             use_cache: bool = True) -> dict:
    model = model or MODEL
    effort = effort or EFFORT
    k = _key(model, system, messages, effort)
    path = os.path.join(CACHE_DIR, k + ".json")
    if use_cache and os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))

    client = _get_client()
    t0 = time.time()
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        thinking={"type": "adaptive"},
        output_config={"effort": effort},
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=messages,
    )
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    rec = {
        "text": text,
        "model": model,
        "effort": effort,
        "stop_reason": resp.stop_reason,
        "usage": {
            "input": resp.usage.input_tokens,
            "output": resp.usage.output_tokens,
            "cache_read": getattr(resp.usage, "cache_read_input_tokens", 0),
            "cache_write": getattr(resp.usage, "cache_creation_input_tokens", 0),
        },
        "latency_s": round(time.time() - t0, 2),
        "request_id": resp._request_id,
    }
    if resp.stop_reason == "refusal":
        rec["refusal"] = True
    json.dump(rec, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return rec


# rough USD cost per 1M tokens (first-party API, 2026-06)
PRICE = {
    "claude-opus-5":  (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def usd(usage: dict, model: str) -> float:
    pin, pout = PRICE.get(model, (2.0, 10.0))
    inp = usage["input"] + usage["cache_read"] * 0.1 + usage["cache_write"] * 1.25
    return inp / 1e6 * pin + usage["output"] / 1e6 * pout
