"""Deterministic 5-fold CV split of train.csv, stratified by modality x body_part.

Writes experiments/folds.json : {case_id: fold_index (0..4)}.
"""
import csv
import json
import os
from collections import defaultdict

HERE = os.path.dirname(__file__)
TRAIN = os.path.join(HERE, "..", "data", "train.csv")
OUT = os.path.join(HERE, "..", "experiments", "folds.json")
K = 5


def main() -> None:
    rows = list(csv.DictReader(open(TRAIN, encoding="utf-8-sig")))
    strata = defaultdict(list)
    for r in rows:
        strata[(r["modality"], r["body_part"])].append(r["case_id"])

    fold = {}
    # round-robin with a global running offset so stratum remainders spread evenly
    n = 0
    for key in sorted(strata):
        for cid in sorted(strata[key]):
            fold[cid] = n % K
            n += 1

    json.dump(fold, open(OUT, "w"), indent=0)
    sizes = [sum(1 for v in fold.values() if v == k) for k in range(K)]
    print(f"wrote {OUT}: {len(fold)} cases, fold sizes {sizes}")


if __name__ == "__main__":
    main()
