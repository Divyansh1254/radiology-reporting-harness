"""Generate reports for every test case and write a submission CSV.

    python make_submission.py --k 4 --tag b2
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import statistics

from data import load_test, load_train
from llm import usd
from pipeline import generate_report

HERE = os.path.dirname(__file__)
SUB = os.path.join(HERE, "..", "submissions")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--model", default=None)
    ap.add_argument("--effort", default=None)
    ap.add_argument("--no-postprocess", action="store_true")
    ap.add_argument("--tag", default="sub")
    args = ap.parse_args()

    train = load_train()
    test = load_test()
    date = dt.date.today().isoformat()
    rows_out = []
    meta = []
    cost = 0.0
    for n, case in enumerate(test, 1):
        out = generate_report(case, train, k=args.k, model=args.model, effort=args.effort,
                              postprocess=not args.no_postprocess)
        rows_out.append({"case_id": case["case_id"], "report": out["report"]})
        cost += usd(out["usage"], out["model"])
        meta.append({"case_id": case["case_id"], "exemplar_ids": out["exemplar_ids"],
                     "refusal": out["refusal"], "chars": len(out["report"])})
        if n % 20 == 0 or n == len(test):
            print(f"  [{n}/{len(test)}]  ~${cost:.2f}")

    assert len(rows_out) == len(test)
    assert len({r["case_id"] for r in rows_out}) == len(test)

    base = f"{date}_{args.tag}"
    csv_path = os.path.join(SUB, base + ".csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "report"])
        w.writeheader()
        w.writerows(rows_out)
    json.dump({"tag": args.tag, "k": args.k, "model": args.model, "effort": args.effort,
               "est_cost_usd": round(cost, 2), "n": len(rows_out), "cases": meta},
              open(os.path.join(SUB, base + ".meta.json"), "w", encoding="utf-8"), indent=1)
    print(f"\nwrote {csv_path}")
    print(f"report length chars: median {statistics.median(len(r['report']) for r in rows_out):.0f}  "
          f"est cost ${cost:.2f}  refusals {sum(m['refusal'] for m in meta)}")


if __name__ == "__main__":
    main()
