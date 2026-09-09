"""Streamlit demo for the Radiology Reporting Harness.

Runs the deterministic parts of the pipeline live — template-similarity retrieval, the
skeleton post-processor, and the RES metric — on synthetic cases (no competition data).
The generation step needs a GPU model (see notebook/rrh.ipynb); its output is precomputed
here so the pipeline and the scorer can be shown end to end.

    streamlit run demo/app.py
"""
import difflib
import json
import os
import sys

import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from exemplars import retrieve            # noqa: E402
from postprocess import reskeleton        # noqa: E402
from res_score import score_case          # noqa: E402

CASES = json.load(open(os.path.join(os.path.dirname(__file__), "data", "cases.json")))
for c in CASES:                            # retrieve() expects these key names
    c["template_content"] = c["template"]
    c["study_description"] = c["study"]

st.set_page_config(page_title="Radiology Reporting Harness", layout="wide")
st.title("Radiology Reporting Harness")
st.caption("Merge a radiologist's telegraphic **dictation** into a normal report **template**, "
           "scored by an edit-distance metric (RES, lower is better). "
           "Synthetic examples — no competition data.")

tab_demo, tab_metric, tab_try = st.tabs(["Pipeline", "The RES metric", "Try your own"])


def diff_html(a: str, b: str) -> str:
    sm = difflib.SequenceMatcher(None, a.split(), b.split(), autojunk=False)
    out = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        seg = " ".join(b.split()[j1:j2])
        if op == "equal":
            out.append(seg)
        elif op in ("replace", "insert"):
            out.append(f"<mark style='background:#c9f7d4'>{seg}</mark>")
    return " ".join(out)


with tab_demo:
    ids = [c["id"] for c in CASES]
    cid = st.sidebar.selectbox("Case", ids)
    case = next(c for c in CASES if c["id"] == cid)
    st.sidebar.markdown(
        f"**{case['modality']} · {case['body_part']}**\n\n{case['study']}\n\n"
        f"RES **{case['res']:.3f}**  ·  baseline (emit template) {case['res_baseline']:.2f}\n\n"
        f"F {case['F']:.2f} · I {case['I']:.2f}")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Template (starting point)")
        st.code(case["template"], language="text")
        st.subheader("Dictation (to merge in)")
        st.info(case["dictation"])
    with c2:
        st.subheader("Retrieved exemplars (few-shot)")
        pool = [c for c in CASES if c["id"] != cid]
        for ex in retrieve(case, pool, k=2):
            st.markdown(f"`{ex['id']}` — {ex['modality']} {ex['body_part']}")
        st.subheader("Pipeline output")
        st.code(case["prediction"], language="text")

    st.divider()
    st.subheader("Output vs reference  ·  green = content the pipeline added correctly")
    st.markdown(
        f"<div style='font-family:ui-monospace,monospace;white-space:pre-wrap;font-size:13px'>"
        f"{diff_html(case['template'], case['reference'])}</div>", unsafe_allow_html=True)

    st.divider()
    st.subheader("Per-field RES (lower is better)")
    rows = [{"field": f["label"], "weight (1=untouched, 3=changed)": f["weight"],
             "word-edit score": round(f["score"], 3)} for f in case["fields"]]
    st.dataframe(rows, use_container_width=True, hide_index=True)


with tab_metric:
    st.markdown("""
`RES = 0.65 · F + 0.35 · I` — a **field-aware weighted word-level edit distance** vs the
reference, after normalization (lowercase, unit spelling, list markers removed, `5mm`→`5 mm`).

- **F (FINDINGS)** aligns sections **by label**. A field the reference *changed* from the
  template weighs **3×** an untouched one — so the score lives in the sections the dictation
  touched. A finding under the **wrong label** is penalised twice. A dropped label scores the max.
- **I (IMPRESSION)** — compared as one block.
- Token weights: **negation / laterality / severity / acuity / numbers / units = 4**,
  other content = 2, function words (`the`, `and`, `of`) = 0.25. **Word order counts.**

`src/res_score.py` reimplements this from the public spec — predicting the reference scores
exactly `0.0`.
""")
    st.subheader("Live score — edit the prediction and watch RES move")
    case = CASES[0]
    ed = st.text_area("prediction", case["prediction"], height=260)
    sc = score_case(reskeleton(case["template"], ed), case["reference"], case["template"])
    m1, m2, m3 = st.columns(3)
    m1.metric("RES", f"{sc['res']:.3f}")
    m2.metric("F (findings)", f"{sc['F']:.3f}")
    m3.metric("I (impression)", f"{sc['I']:.3f}")
    with st.expander("reference"):
        st.code(case["reference"], language="text")


with tab_try:
    st.markdown("Paste a normal **template** and a **dictation**. This runs retrieval + the "
                "skeleton post-processor live; the generation step needs the GPU notebook, so "
                "here the draft is just the dictation dropped in as unlabelled text to show "
                "how the post-processor re-forms it.")
    t = st.text_area("template", CASES[1]["template"], height=200)
    d = st.text_area("dictation", "degenerative changes of the right hip. no fracture", height=80)
    if st.button("Run"):
        fake = {"template_content": t, "modality": "", "body_part": "",
                "study_description": "", "dictation": d}
        st.write("**Nearest exemplars:**",
                 ", ".join(e["id"] for e in retrieve(fake, CASES, k=2)))
        st.code(reskeleton(t, "FINDINGS:\n" + d + "\n\nIMPRESSION:\n" + d), language="text")
