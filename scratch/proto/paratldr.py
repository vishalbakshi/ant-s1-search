"""TLDRs built from the paragraphs up, for the sections in explain.py's PICKS:
1. one plain sentence per paragraph of the filing, saying what it means;
2. those sentences, in order, merged into one paragraph with repeats folded together.
The merge sees only the sentences, never the filing's text.

    python3 scratch/proto/paratldr.py     # writes scratch/proto/paratldr.json; answers cached in scratch/proto/build/paratldr/
"""
import concurrent.futures as cf
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "reader", "tools"))
sys.path.insert(0, HERE)
import distill as d  # noqa: E402
from explain import PICKS  # noqa: E402

BUILD = os.path.join(HERE, "build", "paratldr")

READER = ("an intelligent reader who is not a finance or AI specialist: someone deciding whether to invest, "
          "or trying to understand how this company and its IPO affect the economy")
RULES = """Rules: only facts that are in the text you are given; keep numbers, dates and names exact with their periods; "Anthropic", not "we"; everyday words and complete sentences; no dashes; no hype; never tell anyone to buy or sell. Turn jargon into the concrete thing it refers to, briefly, the first time it appears (compute is the specialized chips in data centers that train and run Claude; a CSP is a cloud provider such as Amazon, Google or Microsoft; take-or-pay means Anthropic pays even if it does not use what it bought). When the filing says something has already happened, say so. Adult, direct language: no "imagine", no made-up analogies, no talking down."""

SENTENCES = f"""You explain Anthropic, PBC's Form S-1 (its October 2026 IPO registration statement) to {READER}.

You get one section of the filing, with every paragraph marked by an id like [p123]. For EACH paragraph, write one sentence (at most 40 words) that says what the paragraph means: its point and why it matters, not a restatement of its wording. A paragraph that only introduces a list can get a short sentence; a paragraph that repeats the one before can say so briefly.

Here is the level we want. For the passage "The majority of our compute is procured under multi-year commitments that are typically entered into well in advance of the periods in which that compute is deployed... If the magnitude or mix of workloads... differs materially from those forecasts... the composition of our available compute may not optimally match our needs", a good sentence is: "Anthropic locks in chips and data center capacity years before it uses them, so it has to guess what it will need, and if demand or research moves differently it will not have the right capacity in time to serve customers or train models."

{RULES}

Answer with only a JSON object, no code fence: {{"p123": "sentence", ...}} with an entry for every paragraph id."""

MERGE = f"""You write the TLDR of one section of Anthropic, PBC's Form S-1 (its October 2026 IPO registration statement) for {READER}.

You get one sentence per paragraph of the section, in the filing's order. Merge them into ONE short paragraph, at most 110 words, that a reader can take in at a glance:
- keep every distinct point, but say each once, in as few words as it takes; fold sentences that make the same point into one;
- when many sentences are items of one kind (a list of causes, contracts, lawsuits), group them under what they have in common and name only the biggest or most telling ones, with their numbers;
- lead with the most consequential point, then follow the filing's order; join related points so it reads as connected prose, not a list;
- drop list introductions, boilerplate, and explanations of a term after the first.
Do not add anything that is not in the sentences. Do not mention paragraphs or "the section".

{RULES}

Answer with only the paragraph, no preamble."""


def claude(system, prompt, as_json):
    key = hashlib.sha256((system + prompt).encode()).hexdigest()[:20]
    path = os.path.join(BUILD, key + ".json")
    if os.path.exists(path):
        return json.load(open(path))
    cmd = [d.CLAUDE, "-p", "--output-format", "json", "--no-session-persistence", "--tools", "", "--system-prompt", system]
    for attempt in range(3):
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=BUILD)
        text = ""
        try:
            text = json.loads(proc.stdout)["result"].strip()
            answer = json.loads(text[text.index("{"):text.rindex("}") + 1]) if as_json else text
            if not answer:
                raise ValueError("empty")
            json.dump(answer, open(path, "w"), indent=1, ensure_ascii=False)
            return answer
        except Exception as e:  # noqa: BLE001
            print(f"  retry {attempt + 1}: {e} | said: {text[:300]!r} {proc.stdout[-300:] if not text else ''} {proc.stderr[-300:]}", file=sys.stderr)
    raise RuntimeError("no answer")


def main():
    os.makedirs(BUILD, exist_ok=True)
    root, nodes, para, _ = d.build_tree()

    def paras_of(n):
        return n["paras"] + [p for c in n["children"] for p in paras_of(c)]

    def one(sid):
        n = nodes[sid]
        ids = paras_of(n)
        sentences = claude(SENTENCES, f"Section: {n['title']}\n\n{d.render_full(n, para)}", True)
        missing = [p for p in ids if p not in sentences]
        if missing:
            print(f"  {sid}: no sentence for {missing}", file=sys.stderr)
        ordered = [d.billions(sentences[p].strip()) for p in ids if p in sentences]
        merged = claude(MERGE, "\n".join(ordered), False)
        print(f"done {sid}: {len(ids)} paragraphs", file=sys.stderr)
        return sid, {"paras": {p: d.billions(sentences[p].strip()) for p in ids if p in sentences},
                     "tldr": d.billions(merged.strip())}

    with cf.ThreadPoolExecutor(6) as pool:
        out = dict(pool.map(one, PICKS))
    json.dump(out, open(os.path.join(HERE, "paratldr.json"), "w"), ensure_ascii=False, indent=1)
    print(f"Wrote scratch/proto/paratldr.json: {len(out)} sections", file=sys.stderr)


if __name__ == "__main__":
    main()
