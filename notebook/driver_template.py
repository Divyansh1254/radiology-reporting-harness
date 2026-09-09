# ==== CELL ====
# Radiology Reporting Harness — retrieval + few-shot + skeleton pipeline (open model, Kaggle GPU)
#
# HOW TO USE
#   1. + Add Input -> Competitions -> the radiology reporting harness competition
#   2. Accelerator (right panel): GPU T4 x2
#   3. Internet: ON  (first run downloads the model from Hugging Face)
#   4. Set MODE in the CONFIG cell.  "cv" = measure local RES;  "submit" = write submission.csv
#   5. Run -> "Restart & Run All"  (NOT plain "Run All" — a stale kernel keeps dead
#      model copies in GPU memory and the load cell then OOMs)
#   6. Save Version the moment it finishes.  Outputs -> /kaggle/working/
#
# Backend: transformers + bitsandbytes 4-bit (matches Kaggle's CUDA 12 image; vLLM's current
# build needs CUDA 13). No pip -U — that breaks Kaggle's pinned deps.

# ==== CELL ====
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch, subprocess, sys
print("torch", torch.__version__, "| cuda", torch.version.cuda,
      "| gpu", torch.cuda.is_available(), torch.cuda.device_count())
assert torch.cuda.is_available(), "No GPU — set Accelerator to GPU T4 x2 and restart the session"
try:
    import bitsandbytes  # noqa
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "bitsandbytes"], check=True)
    import bitsandbytes  # noqa
print("bitsandbytes", bitsandbytes.__version__)

# ==== CELL ====
# ---------------- CONFIG ----------------
MODE       = "cv"        # "cv" | "submit"
CV_FOLD    = 0
CV_LIMIT   = 100         # cap CV cases (None = whole fold)
K          = 2           # few-shot exemplars per case

MODEL      = "Qwen/Qwen2.5-14B-Instruct"    # fallback if it OOMs at load: "Qwen/Qwen2.5-7B-Instruct"
LOAD_4BIT  = True
MAX_INPUT_TOKENS = 9000
MAX_NEW_TOKENS   = 1500   # long MRI reports need this; 1024 truncated them
BATCH_SIZE = 4           # length-bucketed; generation cell auto-halves any batch that OOMs
DEVICE_MAP = "auto"      # spread the 7B across both T4s for KV-cache headroom

WORK       = "/kaggle/working"   # DATA_DIR auto-detected under /kaggle/input below
# ---------------------------------------

# ==== CELL ====
import json, statistics, re, glob, os
from collections import defaultdict
import pandas as pd

_cands = glob.glob("/kaggle/input/**/train.csv", recursive=True)
assert _cands, "train.csv not found under /kaggle/input — use '+ Add Input' to attach the competition data"
DATA_DIR = os.path.dirname(_cands[0])
print("DATA_DIR =", DATA_DIR, "->", sorted(os.listdir(DATA_DIR)))

train = pd.read_csv(f"{DATA_DIR}/train.csv").fillna("").to_dict("records")
test  = pd.read_csv(f"{DATA_DIR}/test.csv").fillna("").to_dict("records")
print(f"train {len(train)}  test {len(test)}")

def make_folds(rows, k=5):
    strata = defaultdict(list)
    for r in rows:
        strata[(r["modality"], r["body_part"])].append(r["case_id"])
    fold, n = {}, 0
    for key in sorted(strata):
        for cid in sorted(strata[key]):
            fold[cid] = n % k
            n += 1
    return fold

folds = make_folds(train, 5)

if MODE == "cv":
    targets = [r for r in train if folds[r["case_id"]] == CV_FOLD]
    if CV_LIMIT:
        targets = targets[:CV_LIMIT]
    def pool_for(c):
        return [r for r in train if folds[r["case_id"]] != folds[c["case_id"]]]
else:
    targets = test
    def pool_for(c):
        return train
print(f"MODE={MODE}  targets={len(targets)}")

# ==== CELL ====
# Build prompts (retrieve() / build_messages() come from the inlined pipeline cells above)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

def make_prompt(case, k):
    ex = retrieve(case, pool_for(case), k=k)
    system, messages = build_messages(case, ex)
    chat = [{"role": "system", "content": system}] + messages
    text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    return text, [e["case_id"] for e in ex]

prompts, meta = [], []
for c in targets:
    p, exids = make_prompt(c, K)
    prompts.append(p)
    meta.append({"case_id": c["case_id"], "exemplar_ids": exids})

ntok = [len(tok(p).input_ids) for p in prompts]
print(f"prompt tokens: median {int(statistics.median(ntok))}  max {max(ntok)}  (cap {MAX_INPUT_TOKENS})")

# ==== CELL ====
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import gc

for _n in ("model", "out", "enc"):          # free any leftovers from an earlier run
    if _n in dir():
        del globals()[_n]
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    for _d in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(_d)
        print(f"  cuda:{_d}  free {free/1e9:.1f} / {total/1e9:.1f} GB")

kw = dict(torch_dtype=torch.float16, device_map=DEVICE_MAP)
if LOAD_4BIT:
    kw["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)

model = AutoModelForCausalLM.from_pretrained(MODEL, **kw).eval()
print("loaded", MODEL, "| device map", set(model.hf_device_map.values()) if hasattr(model, "hf_device_map") else "n/a")

# ==== CELL ====
import time, gc

def _decode(batch, max_new):
    enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
              max_length=MAX_INPUT_TOKENS).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             num_beams=1, pad_token_id=tok.pad_token_id)
    n_in = enc.input_ids.shape[1]
    res = [tok.decode(out[j][n_in:], skip_special_tokens=True).strip() for j in range(len(batch))]
    del enc, out
    return res

def gen_batch(batch, max_new=MAX_NEW_TOKENS):
    """Greedy-decode a list of prompts; recursively halve on CUDA OOM."""
    try:
        return _decode(batch, max_new)
    except torch.cuda.OutOfMemoryError:
        gc.collect(); torch.cuda.empty_cache()
        if len(batch) == 1:
            return None            # caller handles (salvage pass)
        mid = len(batch) // 2
        print(f"   OOM on batch of {len(batch)} -> splitting")
        left = gen_batch(batch[:mid], max_new); right = gen_batch(batch[mid:], max_new)
        return [x for x in (left or [None] * mid)] + [x for x in (right or [None] * (len(batch) - mid))]

def looks_bad(r):
    return (r is None) or (len(r.strip()) < 40) or ("IMPRESSION" not in r.upper())

# length-bucket so each batch has similar padding (less OOM, faster, cleaner output)
order = sorted(range(len(prompts)), key=lambda i: len(tok(prompts[i]).input_ids))
raw = [None] * len(prompts)
t0 = time.time()
for b in range(0, len(order), BATCH_SIZE):
    idx = order[b:b + BATCH_SIZE]
    res = gen_batch([prompts[i] for i in idx])
    for i, r in zip(idx, res):
        raw[i] = r
    done = min(b + BATCH_SIZE, len(order))
    print(f"  {done}/{len(order)}   {(time.time()-t0)/done:.1f}s/case")
print(f"main pass done in {(time.time()-t0)/60:.1f} min")

# salvage: retry bad/empty cases individually with fewer, shorter exemplars
bad = [i for i in range(len(raw)) if looks_bad(raw[i])]
print(f"salvage pass: {len(bad)} cases")
for i in bad:
    for kk in (2, 1):
        p, _ = make_prompt(targets[i], kk)
        r = gen_batch([p])
        r = r[0] if r else None
        if not looks_bad(r):
            raw[i] = r
            break
    if looks_bad(raw[i]):
        raw[i] = raw[i] or ""      # give up -> skeleton falls back to template
print(f"still bad after salvage: {sum(looks_bad(raw[i]) for i in range(len(raw)))}")
raw = ["" if r is None else r for r in raw]
assert len(raw) == len(prompts)

# ==== CELL ====
preds = [reskeleton(c["template_content"], r) for c, r in zip(targets, raw)]

if MODE == "cv":
    b0  = [score_case(c["template_content"], c["report"], c["template_content"])["res"] for c in targets]
    sc  = [score_case(p, c["report"], c["template_content"]) for c, p in zip(targets, preds)]
    res = [s["res"] for s in sc]

    print(f"\n=== CV fold {CV_FOLD}  n={len(res)}  model={MODEL} ===")
    print(f"B0 (emit template)  mean RES = {statistics.mean(b0):.4f}")
    print(f"pipeline            mean RES = {statistics.mean(res):.4f}   median = {statistics.median(res):.4f}")
    print(f"                    F = {statistics.mean(s['F'] for s in sc):.3f}   I = {statistics.mean(s['I'] for s in sc):.3f}")
    bucket = defaultdict(list)
    for c, s in zip(targets, sc):
        bucket[c["modality"]].append(s["res"])
    for m in sorted(bucket):
        print(f"    {m:5s} n={len(bucket[m]):3d}  mean RES = {statistics.mean(bucket[m]):.4f}")

    out = []
    for c, p, r, s, m in zip(targets, preds, raw, sc, meta):
        out.append({"case_id": c["case_id"], "modality": c["modality"], "body_part": c["body_part"],
                    "res": s["res"], "F": s["F"], "I": s["I"],
                    "dictation": c["dictation"], "template_content": c["template_content"],
                    "gold": c["report"], "pred": p, "raw": r, "exemplar_ids": m["exemplar_ids"]})
    out.sort(key=lambda x: -x["res"])
    json.dump(out, open(f"{WORK}/cv_results.json", "w"), ensure_ascii=False, indent=1)
    with open(f"{WORK}/cv_summary.txt", "w") as f:
        f.write(f"fold {CV_FOLD} n={len(res)} model={MODEL}\nB0 {statistics.mean(b0):.4f}\n"
                f"pipeline {statistics.mean(res):.4f} median {statistics.median(res):.4f}\n\nworst 15:\n")
        for x in out[:15]:
            f.write(f"  {x['res']:.3f}  {x['modality']:5s} {x['body_part']}\n")
    print(f"\nwrote {WORK}/cv_results.json  and  {WORK}/cv_summary.txt")

    # ---- inline diagnostics (screenshot this — no download needed) ----
    print("\n" + "=" * 70 + "\nWORST 6 CASES\n" + "=" * 70)
    for x in out[:6]:
        print(f"\n### RES {x['res']:.3f}  ({x['modality']} {x['body_part']})  F={x['F']:.2f} I={x['I']:.2f}")
        print("- DICTATION:", " ".join(x["dictation"].split())[:300])
        print("- GOLD:\n" + x["gold"][:700])
        print("- PRED:\n" + x["pred"][:700])
    print("\n" + "=" * 70 + "\nMEDIAN-DIFFICULTY CASE (for style check)\n" + "=" * 70)
    mid = out[len(out) // 2]
    print(f"RES {mid['res']:.3f}  ({mid['modality']} {mid['body_part']})")
    print("- DICTATION:", " ".join(mid["dictation"].split())[:300])
    print("- GOLD:\n" + mid["gold"][:700])
    print("- PRED:\n" + mid["pred"][:700])
else:
    sub = pd.DataFrame({"case_id": [c["case_id"] for c in targets], "report": preds})
    assert sub["case_id"].is_unique and len(sub) == len(test), "submission shape wrong"
    sub.to_csv(f"{WORK}/submission.csv", index=False)
    json.dump(meta, open(f"{WORK}/submission_meta.json", "w"), indent=1)
    print(f"wrote {WORK}/submission.csv  rows={len(sub)}")
    print(f"report chars: median {int(sub.report.str.len().median())}  max {int(sub.report.str.len().max())}")
    print("--- example ---\n" + sub.iloc[0].report[:700])
