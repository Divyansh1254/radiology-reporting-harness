# Design notes

The metric analysis that drove the pipeline. Kept as a record of the engineering reasoning.

## The task

Text → text. Given `template_content` (a normal structured report) + `dictation` (telegraphic
radiologist findings) + routing metadata, produce the final `report`: the dictation's findings
merged into the template's sections, in house style, with a concise `IMPRESSION`.

No images. Private skills-based hiring challenge; the data is not redistributable.

## What RES rewards (drives every design choice)

- `RES = 0.65·F + 0.35·I`. FINDINGS routing carries roughly 2× the IMPRESSION.
- Fields the **reference** changed from the template = weight 3; untouched = weight 1. The score
  concentrates in the 1–3 sections the dictation actually touched.
- Wrong-label routing = double penalty (missing where expected + extra where it isn't).
  **Routing is the highest-leverage decision.**
- Missing a template field label = maximum score for that field → **always emit every label, in order.**
- Weight-4 tokens: negation, laterality, severity, acuity, numbers, units — get these exactly
  right and in the right place.
- Word order is preserved in the distance → match the reference's phrasing, not just its content.
- List markers are stripped before scoring → IMPRESSION numbering is cosmetic.
- Unexpected fields / stray unlabelled content are penalised → only add a trailer paragraph when
  the house style clearly does.
- Templates use both `BONES:` and `Bones:`; **the reference always UPPERCASEs labels.** The
  scorer and the post-processor both have to normalise this.

## Pipeline

```
case
 ├─ 1. retrieval        k train exemplars ranked by template-similarity (exact-template match
 │                      strongly boosted), then modality / body-part / study
 ├─ 2. generation       instruction model, greedy: house-style guide + full exemplar triples
 │                      + this template + dictation → draft report
 └─ 3. post-process     reskeleton(): draft forced onto the template skeleton — every label
                        verbatim & UPPERCASE & in order, template blank-line layout, unfilled
                        field → template sentence, stray trailing findings kept before IMPRESSION
```

## What moved the score

| change | local RES | note |
|---|---|---|
| emit template unchanged | 0.63 | baseline |
| retrieval + few-shot + skeleton, Qwen2.5-7B 4-bit | 0.40 | |
| fix Title-Case label handling in scorer + post-processor | −0.05 | scorer was discarding correct output on ~12% of cases |
| routing/fidelity prompt rewrite + Qwen2.5-14B, k=2 | **0.33** | public LB 0.41; even across CT / MRI / XR / US |
| "always emit residual-normal impression line" | reverted | +0.02 — model appended a redundant line the reference often omits |
| snap near-template fields to template verbatim | reverted | no gain — the reference rephrases untouched fields more than expected |

## Failure modes at 0.33 (from the CV error analysis)

- **Routing** — degenerative / arthroplasty / sacroiliac findings placed under SOFT TISSUES.
- **Omissions** — a finding in a long, messy dictation dropped entirely.
- **Exemplar leakage** — the model occasionally copies a normal sentence from a few-shot example.
- **IMPRESSION phrasing** — too terse ("Degenerative changes.") vs the localised reference
  ("Mild degenerative changes of the pelvis.").

## What I'd do next

- **Verification pass** — a second call that checks every weight-4 token and every dictated
  finding is present and correctly routed; regenerate/patch on a flag.
- **QLoRA fine-tune** on the training pairs (`peft` + `transformers.Trainer` scaffold in git
  history) — narrow supervised task, should beat few-shot on routing and house phrasing.
- **Self-consistency** — 3 samples, medoid by pairwise RES.
- **Frontier model** in place of the local 14B — the residual errors are capability-bound.

## Constraints honoured

- Solo work. Data not redistributed, not used for clinical decisions, no re-identification.
- Fully automated pipeline — the post-processor uses only generic rules, no per-case edits.
- No API keys or secrets in the notebook or any committed file (env vars only).
- The Kaggle notebook regenerates the submitted CSV end to end.
