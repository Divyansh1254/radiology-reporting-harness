# Running the pipeline on Kaggle (zero-cost, open model)

The pipeline: retrieve 3 similar train reports → few-shot prompt an open instruction model
(`Qwen2.5-14B-Instruct-AWQ`) → force the output onto the template skeleton → score / submit.

## One-time setup

1. **kaggle.com → the competition → `Code` tab → `New Notebook`.**
2. Right panel:
   - **Accelerator → `GPU T4 x2`**  (not P100 — the AWQ kernels need a Turing-or-newer GPU).
   - **Internet → `On`**  (first run downloads the ~9 GB model from Hugging Face).
3. **`Input` → `+ Add Input` → `Competitions` → `radiology-reporting-harness`.**
   Confirm the path in the panel is `/kaggle/input/radiology-reporting-harness`; if it differs,
   edit `DATA_DIR` in the CONFIG cell.
4. **`File` → `Import Notebook` → upload `notebook/rrh.ipynb`** (from this repo).

## CV run (measure local RES)

- CONFIG cell: `MODE = "cv"`, `CV_LIMIT = 120`.
- `Run All`. First run ≈ 20–35 min (vLLM install + model download + inference); later runs faster.
- When it finishes, the last cell prints:
  ```
  B0 (emit template)  mean RES = 0.66xx
  pipeline            mean RES = 0.xxxx   median = ...
      XRAY / MRI / CT / USG breakdown
  ```
- **Send me:** the printed block, and download **`/kaggle/working/cv_results.json`**
  (Output tab → download) and attach it here. It has per-case `res`, `pred`, `raw`, `gold`,
  sorted worst-first — that's what I use to tune the prompt/retrieval/post-processor.

## Iteration

I edit `src/`, rebuild `rrh.ipynb`, send it back. You: `File → Import Notebook` (replace), `Run All`.
Expect 2–3 rounds.

## Submission run

- CONFIG cell: `MODE = "submit"`.
- `Run All`. Downloads `/kaggle/working/submission.csv` (132 rows, `case_id,report`).
- Competition page → **`Submit Predictions`** → upload `submission.csv`.
- Keep this notebook **private** — it is the "private notebook containing the complete pipeline"
  the rules require. Disclose in a text cell: model id (`Qwen/Qwen2.5-14B-Instruct-AWQ`),
  `temperature=0`, `k=3`, vLLM version, and that no external API is used.
- 5 submissions/day max. Select 2 for final scoring near the deadline.

## If something breaks

| Symptom | Fix |
|---|---|
| `pip install vllm` fails / version conflict | paste me the error; we pin a version |
| CUDA OOM on model load | CONFIG: `TENSOR_PARALLEL = 2`, or `MAX_MODEL_LEN = 12288`, or `GPU_MEM_UTIL = 0.85` |
| `awq` quantization error on T4 | try `MODEL = "Qwen/Qwen2.5-7B-Instruct"`, `QUANTIZATION = None` |
| model download blocked | Internet is Off — turn it On in the right panel |
| very slow (>1 hr) | `CV_LIMIT = 60` while tuning; full test run is only 132 cases |
