"""Inline demo/data/cases.json into public/index.html (the deployed static demo).

Run after build_demo_data.py if the synthetic cases change:

    python demo/build_demo_data.py && python demo/build_page.py
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PAGE = os.path.join(ROOT, "public", "index.html")

cases = json.load(open(os.path.join(HERE, "data", "cases.json")))
for c in cases:
    for k in ("model_raw", "template_content", "study_description"):
        c.pop(k, None)

html = open(PAGE, encoding="utf-8").read()
payload = json.dumps(cases, separators=(",", ":"))
html, n = re.subn(
    r'(<script id="data" type="application/json">).*?(</script>)',
    lambda m: m.group(1) + payload + m.group(2),
    html, count=1, flags=re.S)
assert n == 1, "data script tag not found in public/index.html"

open(PAGE, "w", encoding="utf-8", newline="\n").write(html)
print(f"inlined {len(cases)} cases into {os.path.relpath(PAGE, ROOT)}")
