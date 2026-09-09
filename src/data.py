"""Shared data loading + fold access."""
from __future__ import annotations

import csv
import json
import os
from typing import Dict, List

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
EXP = os.path.join(HERE, "..", "experiments")

INPUT_COLS = ["modality", "body_part", "study_description", "patient_age_band",
              "patient_sex", "template_content", "dictation"]


def _read(path: str) -> List[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_train() -> List[dict]:
    return _read(os.path.join(DATA, "train.csv"))


def load_test() -> List[dict]:
    return _read(os.path.join(DATA, "test.csv"))


def load_folds() -> Dict[str, int]:
    return json.load(open(os.path.join(EXP, "folds.json")))


def fold_split(fold: int):
    """Return (train_rows, val_rows) for a CV fold index."""
    rows = load_train()
    folds = load_folds()
    tr = [r for r in rows if folds.get(r["case_id"]) != fold]
    va = [r for r in rows if folds.get(r["case_id"]) == fold]
    return tr, va
