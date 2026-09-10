# Radiology Reporting Harness

Turn a radiologist's **telegraphic dictation** plus a **normal report template** into a
polished, correctly-structured radiology report — evaluated by an edit-distance metric that
rewards putting each finding in the right place, in the house wording, and inventing nothing.

Built for a private, skills-based ML hiring challenge (host: Natoe.ai). This repo is the
pipeline, the local evaluation harness, and a write-up of the approach. **It contains no
competition data** — the challenge data is not redistributable, so the demo and examples here
are synthetic and hand-authored to match the task's structure.

**▶ Interactive demo:** _Render static site — URL added after deploy._ &nbsp;
The demo is a single self-contained page (`public/index.html`): case explorer, live RES
scoring, word-diff against the reference.

---

## The task

Text → text. Each case gives you:

| field | meaning |
|---|---|
| `modality`, `body_part`, `study_description` | routing metadata (CT / MRI / XR / US, region, exam) |
| `template_content` | a **normal** structured report — the starting point, with fixed section labels (`LUNGS:`, `BONES:`, `VERTEBRAE:` …) |
| `dictation` | the radiologist's raw shorthand — typos, abbreviations, fragments, unordered ("degen changes si joint rt side and rt hip") |

You produce the final `report`: the dictation's findings merged into the template's sections,
in house style, with a concise `IMPRESSION`.

```
                TEMPLATE                          DICTATION
  FINDINGS:                              "mild lower lumbar facet
  VERTEBRAE: Normal density and           degenerative changes. mild
    alignment. No fracture.               disc space narrowing L4-L5"
  DISC SPACES: Preserved.
  SOFT TISSUES: Unremarkable.                        │
  IMPRESSION: No acute abnormality.                  ▼
                                    ┌─────────────────────────────┐
                                    │  retrieval → generate →     │
                                    │  skeleton post-processor    │
                                    └─────────────────────────────┘
                                                    │
                                                    ▼
  FINDINGS:
  VERTEBRAE: Mild lower lumbar facet arthropathy is present. Normal vertebral body
    heights and alignment. No fracture or osseous lesion.
  DISC SPACES: There is mild disc space narrowing at L4-L5 and L5-S1.
  SOFT TISSUES: Unremarkable.
  IMPRESSION:
  1. Mild degenerative changes of the lower lumbar spine, most pronounced at L4-L5 and L5-S1.
```

## The metric (RES — Radiology Edit Score, lower is better)

`RES = 0.65·F + 0.35·I` — a **field-aware, weighted word-level edit distance** between the
prediction and the reference, after normalization (lowercasing, unit spelling, list-marker
removal, letter–number splitting).

- **`F` (FINDINGS)** aligns fields **by label**. A field the reference changed from the
  template carries **3×** the weight of an untouched field, so the score concentrates on the
  1–3 sections the dictation actually touched. A finding under the **wrong label** is
  penalised twice (missing where expected + extra where it isn't). A dropped label scores the
  maximum.
- **`I` (IMPRESSION)** is compared as one block.
- Token weights: **negation / laterality / severity / acuity / numbers / units = 4**,
  other content words = 2, function words = 0.25. **Word order matters.**

`src/res_score.py` is a from-scratch reimplementation of this metric from the public spec
(identity check: predicting the reference scores exactly `0.0`). It's what every design
decision was measured against.

## Approach — a metric-aware pipeline

```
case
 ├─ 1. retrieval        k nearest training examples by template-similarity (exact-template
 │                      matches strongly boosted) then modality/body-part/study — they teach
 │                      the per-template house wording, not just generic style
 ├─ 2. generation       instruction model, greedy: house-style system guide + full
 │                      (template, dictation, report) exemplars + this case → draft report
 └─ 3. skeleton         reskeleton(): force the draft onto the template's exact skeleton —
    post-processor      every label reproduced verbatim, UPPERCASE, in template order;
                        template blank-line layout; a field the model didn't fill falls back
                        to the template sentence; stray trailing findings kept before IMPRESSION
```

Why this shape:

- **The metric rewards structure the model drifts on.** Templates use both `BONES:` and
  `Bones:`; the reference always UPPERCASEs. Models rename, reorder, and drop labels. The
  deterministic post-processor recovers all of that for free — fixing it was worth ~0.05 RES.
- **Retrieval by template, not by embedding.** Within a modality×body-part the templates are
  near-duplicates; matching the *template* gives the model worked examples of exactly how each
  field of *this* template transforms.
- **No fine-tuning needed to be competitive**, though it's the clear next step (below).

## Results

Local 5-fold CV (`src/res_score.py`), and the competition public leaderboard:

| system | RES | notes |
|---|---|---|
| emit the template unchanged (baseline) | **0.63** | what "do nothing" scores |
| retrieval + few-shot + skeleton, Qwen2.5-7B (4-bit) | 0.40 | |
| **+ label fix + routing/fidelity prompt, Qwen2.5-14B (4-bit), k=2** | **0.33** local · **0.41** public LB | even across CT / MRI / XR / US |

Run entirely at **zero cost** — generation on a free Kaggle T4 GPU with `Qwen2.5-14B-Instruct`
in 4-bit (`notebook/rrh.ipynb`). The pipeline is provider-agnostic (`src/llm.py` also targets
the Anthropic API); a frontier model in place of the 14B would very likely close much of the
gap to the leaders.

### What I'd do next (deadline hit first)

- **Verification pass** — a second call that checks every weight-4 token (negation, laterality,
  severity, measurements) is present and correctly placed, and that no dictated finding was
  dropped. The residual errors are mostly routing + omissions, exactly what this catches.
- **QLoRA fine-tune** on the training pairs — narrow, templated, supervised task; `peft` +
  `transformers.Trainer` scaffold is in git history. Started, ran out of runway.
- **Self-consistency** — 3 samples, pick the medoid by pairwise RES.

## Repo layout

```
src/
  res_score.py     from-scratch RES metric (field-aware weighted word edit distance)
  exemplars.py     template-similarity k-NN retrieval
  prompt.py        few-shot prompt builder  +  house_style.md
  postprocess.py   reskeleton() — forces the draft onto the template skeleton
  pipeline.py      retrieval → generate → post-process
  llm.py           Anthropic client wrapper (cached); Kaggle notebook uses a local model
  run_cv.py        5-fold cross-validation harness
  splits.py        stratified fold assignment
notebook/
  rrh.ipynb        self-contained Kaggle notebook (built from src/ by build_notebook.py)
  KAGGLE.md        how to run it on a free GPU
demo/
  app.py           Streamlit demo (synthetic cases; retrieval + scorer + post-processor run live)
  build_demo_data.py   authors the synthetic cases and scores them with the real metric
  build_page.py    inlines the scored cases into public/index.html
public/index.html  the static interactive demo (deployed via render.yaml)
render.yaml        Render static-site blueprint
PLAN.md            design notes — the metric analysis that drove the pipeline
```

## Run it

```bash
pip install -r requirements.txt

# the local metric
python -c "from src.res_score import score_case; print('scorer imports OK')"

# the static demo — just open it
open public/index.html                       # or double-click

# or the Streamlit demo (retrieval + scorer + post-processor run live)
python demo/build_demo_data.py && python demo/build_page.py
streamlit run demo/app.py
```

**Deploy the static demo (Render):** New → Blueprint → pick this repo (reads `render.yaml`),
or New → Static Site with build command empty and publish directory `public`.

The Kaggle notebook (`notebook/rrh.ipynb`) needs the competition dataset attached and a GPU;
see `notebook/KAGGLE.md`.

## License

MIT — see `LICENSE`. Synthetic demo content only; not for clinical use.
