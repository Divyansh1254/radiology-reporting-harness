"""Run the pipeline over a CV subset and report local RES.

    python run_cv.py --fold 0 --limit 40 --k 4
    python run_cv.py --all                       # every train row via its held-out fold
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict

from data import EXP, load_folds, load_train
from llm import usd
from pipeline import generate_report
from res_score import score_case

OUT = os.path.join(EXP, "cv")
os.makedirs(OUT, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--no-postprocess", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--effort", default=None)
    ap.add_argument("--tag", default="b2")
    args = ap.parse_args()

    rows = load_train()
    folds = load_folds()
    by_id = {r["case_id"]: r for r in rows}

    if args.all:
        targets = rows
    else:
        targets = [r for r in rows if folds.get(r["case_id"]) == args.fold][: args.limit]

    results = []
    cost = 0.0
    for n, case in enumerate(targets, 1):
        pool = [r for r in rows if folds.get(r["case_id"]) != folds.get(case["case_id"])]
        out = generate_report(case, pool, k=args.k, model=args.model, effort=args.effort,
                              postprocess=not args.no_postprocess)
        sc = score_case(out["report"], case["report"], case["template_content"])
        cost += usd(out["usage"], out["model"])
        results.append({
            "case_id": case["case_id"], "modality": case["modality"],
            "body_part": case["body_part"], "res": sc["res"], "F": sc["F"], "I": sc["I"],
            "refusal": out["refusal"], "report": out["report"], "raw": out["raw"],
        })
        if n % 10 == 0 or n == len(targets):
            run = statistics.mean(x["res"] for x in results)
            print(f"  [{n}/{len(targets)}]  running mean RES = {run:.4f}   ~${cost:.2f}")

    res = [x["res"] for x in results]
    print(f"\n=== {args.tag}  n={len(res)}  mean RES = {statistics.mean(res):.4f}  "
          f"median = {statistics.median(res):.4f}  (F={statistics.mean(x['F'] for x in results):.3f} "
          f"I={statistics.mean(x['I'] for x in results):.3f})  est ${cost:.2f} ===")
    bucket = defaultdict(list)
    for x in results:
        bucket[x["modality"]].append(x["res"])
    for mkey in sorted(bucket):
        print(f"    {mkey:5s} n={len(bucket[mkey]):3d}  mean RES = {statistics.mean(bucket[mkey]):.4f}")

    path = os.path.join(OUT, f"{args.tag}_fold{'all' if args.all else args.fold}.json")
    json.dump(results, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
