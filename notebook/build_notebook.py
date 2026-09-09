"""Assemble a self-contained Kaggle notebook from the src/ modules + driver_template.py.

    python notebook/build_notebook.py   ->   notebook/rrh.ipynb

src/ stays the single source of truth; intra-project imports are stripped when inlined and
house_style.md is embedded into the prompt module.
"""
import json
import os
import re

HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, "..", "src")

INTERNAL = r"^(from|import)\s+(res_score|exemplars|prompt|postprocess|data|llm)\b.*$"


def load(mod):
    return open(os.path.join(SRC, mod), encoding="utf-8").read()


def strip_internal(code):
    lines = []
    for l in code.splitlines():
        if re.match(r'^if __name__ == ["\']__main__["\']:', l):
            break                      # drop module self-test blocks
        if re.match(INTERNAL, l):
            continue
        lines.append(l)
    return "\n".join(lines)


def code_cell(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.splitlines(keepends=True)}


def md_cell(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def main():
    house = load("house_style.md")
    prompt_py = load("prompt.py")
    # replace the file read with an inline literal
    prompt_py = prompt_py.replace(
        'HERE = os.path.dirname(__file__)\n'
        'HOUSE_STYLE = open(os.path.join(HERE, "house_style.md"), encoding="utf-8").read()',
        'HOUSE_STYLE = ' + repr(house))
    prompt_py = strip_internal(prompt_py).replace("import os\n", "")

    cells = [md_cell(
        "# Radiology Reporting Harness — pipeline notebook\n\n"
        "Self-contained. Built from the project `src/` modules by `notebook/build_notebook.py` — "
        "edit there, not here. Scorer, retrieval, prompt and post-processor are identical to the "
        "locally-tested versions.")]

    cells.append(code_cell("# ===== scorer (res_score.py) =====\n" + strip_internal(load("res_score.py"))))
    cells.append(code_cell("# ===== prompt + house style (prompt.py) =====\n" + prompt_py))
    cells.append(code_cell("# ===== retrieval (exemplars.py) =====\n" + strip_internal(load("exemplars.py"))))
    cells.append(code_cell("# ===== post-processor (postprocess.py) =====\n" + strip_internal(load("postprocess.py"))))

    # driver: split on markers
    driver = open(os.path.join(HERE, "driver_template.py"), encoding="utf-8").read()
    for i, chunk in enumerate(driver.split("# ==== CELL ====")):
        chunk = chunk.strip("\n")
        if not chunk.strip():
            continue
        cells.append(code_cell(chunk))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = os.path.join(HERE, "rrh.ipynb")
    json.dump(nb, open(out, "w", encoding="utf-8"), indent=1)
    print(f"wrote {out}  ({len(cells)} cells)")


if __name__ == "__main__":
    main()
