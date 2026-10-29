"""Trace every number in the paragraph-built TLDRs back to the filing, by value.

A number counts as traced when the same value, at the precision the TLDR shows, appears in:
  - for a paragraph's sentence: that paragraph, else (tier 2) anywhere in its section;
  - for a section's merged TLDR: its own sentences and its subsections' TLDRs, else (tier 2)
    anywhere in the section's filing text, subsections included.
Values are compared across units, so "$16.3B" matches "16,286" in a table given in millions and
"4.45 million" matches "4,450,000". Years and small counts (under 10) are skipped.

    python3 scratch/proto/check_numbers2.py            # summary
    python3 scratch/proto/check_numbers2.py --list     # every untraced number with its sentence
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
S1 = json.load(open(os.path.join(HERE, "..", "..", "reader", "s1.json")))
R = json.load(open(os.path.join(HERE, "paratldr.json")))
NODES = {}


def index(n):
    NODES[n["id"]] = n
    for c in n["children"]:
        index(c)


index(S1["root"])
UNITS = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12, "b": 1e9, "m": 1e6}
NUM = re.compile(r"(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s*(%|thousand|million|billion|trillion|B\b|M\b)?", re.I)


def claims(text):
    """Numbers stated in a TLDR: (shown text, value, tolerance)."""
    out = []
    for m in NUM.finditer(text):
        whole, frac, unit = m.group(1), m.group(2) or "", (m.group(3) or "").lower()
        v = float(whole.replace(",", "") + frac)
        if unit != "%" and not frac and (v < 10 or 1900 <= v <= 2100):
            continue   # small counts and years
        scale = UNITS.get(unit, 1)
        tol = 0.5 * 10 ** -len(frac[1:]) * scale if frac else 0.5 * scale if scale > 1 else 0
        out.append((m.group(0).strip(), v * scale, tol, unit == "%"))
    return out


def values(text):
    """Every value a source text could mean: as written, and scaled as a table in thousands/millions/billions would be."""
    text = text.replace(" ,", ",").replace(", ", ",") if False else text
    vals, pcts = set(), set()
    for m in NUM.finditer(text.replace(" , ", "")):
        v = float(m.group(1).replace(",", "") + (m.group(2) or ""))
        unit = (m.group(3) or "").lower()
        if unit == "%":
            pcts.add(v)
            continue
        if unit in UNITS:
            vals.add(v * UNITS[unit])
        else:
            vals.update({v, v * 1e3, v * 1e6, v * 1e9})
        pcts.add(v)   # tables often give percentages without the sign
    return vals, pcts


def traced(claim, src):
    _, v, tol, pct = claim
    vals, pcts = src
    pool = pcts if pct else vals
    return any(abs(v - x) <= max(tol, 1e-9 * abs(x)) for x in pool)


def section_text(i):
    n = NODES[i]
    return " ".join([S1["paras"][p][0] for p in n["paras"]] + [section_text(c["id"]) for c in n["children"]])


rows = []   # (sid, where, claim text, tier: 1 traced, 2 traced to section, 0 untraced, sentence)
for sid, r in R.items():
    sec = values(section_text(sid))
    for p, s in r["paras"].items():
        own = values(S1["paras"][p][0])
        for c in claims(s):
            rows.append((sid, p, c[0], 1 if traced(c, own) else 2 if traced(c, sec) else 0, s))
    inputs = " ".join(list(r["paras"].values()) + [R[c["id"]]["tldr"] for c in NODES[sid]["children"] if c["id"] in R])
    src = values(inputs)
    for c in claims(r["tldr"]):
        rows.append((sid, "merged", c[0], 1 if traced(c, src) else 2 if traced(c, sec) else 0, r["tldr"]))

for kind in ("sentence", "merged"):
    mine = [x for x in rows if (x[1] == "merged") == (kind == "merged")]
    t1, t2, t0 = (sum(1 for x in mine if x[3] == k) for k in (1, 2, 0))
    print(f"{kind}: {len(mine)} numbers · {t1} traced to their source · {t2} only elsewhere in the section · {t0} untraced")
if "--list" in sys.argv:
    for sid, where, text, tier, s in rows:
        if tier == 0:
            print(f"{sid} {where}: {text!r} | {s[:160]}")
