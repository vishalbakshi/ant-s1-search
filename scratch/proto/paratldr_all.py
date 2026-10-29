"""Paragraph-built TLDRs for every section of the filing, in batches.

1. One sentence per paragraph, for every paragraph: sections go into a call together, in reading
   order, up to about BATCH_CHARS of filing text, each with its heading path.
2. Merge bottom-up, deepest sections first: a section's TLDR merges its own paragraphs' sentences
   and its subsections' TLDRs, in the filing's order. A section that is a single paragraph keeps
   that sentence. Many sections are merged per call.
The sections already reviewed (paratldr25.json) are kept as they are.

    python3 scratch/proto/paratldr_all.py     # writes scratch/proto/paratldr.json; cached in scratch/proto/build/paratldr/
"""
import concurrent.futures as cf
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paratldr as pt  # noqa: E402  (prompts, rules and the cached claude call)
from paratldr import d  # noqa: E402

BATCH_CHARS = 40_000   # filing text per sentence call
MERGE_ITEMS = 30       # sections per merge call
JOBS = 6

SENTENCES = pt.SENTENCES.replace(
    "You get one section of the filing, with every paragraph marked by an id like [p123].",
    "You get several sections of the filing, each with its heading and where it sits, and every paragraph marked by an id like [p123]. Read each paragraph in the light of its own section.")

MERGE = pt.MERGE.replace(
    "You write the TLDR of one section of",
    "You write TLDRs of sections of").replace(
    "You get one sentence per paragraph of the section, in the filing's order. Merge them into ONE short paragraph",
    "For each section you get, in the filing's order, one sentence per paragraph of its own text and, marked \"Subsection\", the TLDR of each of its subsections. "
    "A section's TLDR must cover the WHOLE section, subsections included, so when there are many subsections, pick out and combine the points that matter most across them. "
    "For each section, merge its items into ONE short paragraph").replace(
    "Answer with only the paragraph, no preamble.",
    'Answer with only a JSON object, no code fence: {"<section id>": "<tldr>", ...} with an entry for every section id.')


def main():
    os.makedirs(pt.BUILD, exist_ok=True)
    root, nodes, para, _ = d.build_tree()
    kept = json.load(open(os.path.join(HERE, "paratldr25.json")))
    parent, depth, path = {}, {}, {}

    def walk(n, p, k, trail):
        parent[n["id"]], depth[n["id"]], path[n["id"]] = p, k, " > ".join(trail)
        for c in n["children"]:
            walk(c, n["id"], k + 1, trail + [c["title"]])

    walk(root, None, 0, [])
    order = [i for i in depth if i != "root"]   # reading order

    # ---------- 1. sentences ----------
    sentences = {p: s for r in kept.values() for p, s in r["paras"].items()}
    todo = [i for i in order if i not in kept and nodes[i]["paras"]]
    batches, size = [], BATCH_CHARS
    for i in todo:
        n = sum(len(para[p]["text"]) for p in nodes[i]["paras"])
        if size + n > BATCH_CHARS:
            batches.append([])
            size = 0
        batches[-1].append(i)
        size += n

    def render(i):
        n = nodes[i]
        return "\n\n".join([f"## [{i}] {n['title']}\n(in: {path[i] or 'top level'})"] + [f"[{p}] {para[p]['text']}" for p in n["paras"]])

    def sentence_batch(group):
        want = [p for i in group for p in nodes[i]["paras"]]
        got = pt.claude(SENTENCES, "\n\n".join(render(i) for i in group), True)
        missing = [i for i in group if any(p not in got for p in nodes[i]["paras"])]
        if missing and len(group) > 1:   # ask again for just those sections
            for i in missing:
                got.update(sentence_batch([i]))
        print(f"sentences: {len([p for p in want if p in got])}/{len(want)} paragraphs", file=sys.stderr)
        return {p: got[p] for p in want if p in got}

    with cf.ThreadPoolExecutor(JOBS) as pool:
        for got in pool.map(sentence_batch, batches):
            sentences.update({p: d.billions(s.strip()) for p, s in got.items()})
    print(f"{len(sentences)} of {len(para)} paragraphs have a sentence", file=sys.stderr)

    # ---------- 2. merges, deepest first ----------
    tldr = {i: r["tldr"] for i, r in kept.items()}

    def items(i):
        n = nodes[i]
        out = [sentences[p] for p in n["paras"] if p in sentences]
        out += [f"Subsection \"{c['title']}\": {tldr[c['id']]}" for c in n["children"]]
        return out

    levels = defaultdict(list)
    for i in order:
        levels[depth[i]].append(i)
    for k in sorted(levels, reverse=True):
        need = [i for i in levels[k] if i not in tldr]
        single = [i for i in need if len(items(i)) == 1 and not nodes[i]["children"]]
        for i in single:
            tldr[i] = items(i)[0]
        need = [i for i in need if i not in single]
        groups = [need[j:j + MERGE_ITEMS] for j in range(0, len(need), MERGE_ITEMS)]

        def merge(group):
            prompt = "\n\n".join(f"## [{i}] {nodes[i]['title']}\n" + "\n".join(f"- {x}" for x in items(i)) for i in group)
            got = pt.claude(MERGE, prompt, True)
            for i in [i for i in group if i not in got]:   # one at a time, if a batch dropped any
                got.update(pt.claude(MERGE, f"## [{i}] {nodes[i]['title']}\n" + "\n".join(f"- {x}" for x in items(i)), True))
            return got

        with cf.ThreadPoolExecutor(JOBS) as pool:
            for got in pool.map(merge, groups):
                tldr.update({i: d.billions(t.strip()) for i, t in got.items() if i in nodes})
        print(f"depth {k}: {len(levels[k])} sections merged", file=sys.stderr)

    out = {i: {"paras": {p: sentences[p] for p in nodes[i]["paras"] if p in sentences}, "tldr": tldr[i]}
           for i in order if i in tldr}
    for i, r in kept.items():
        out[i] = r
    json.dump(out, open(os.path.join(HERE, "paratldr.json"), "w"), ensure_ascii=False, indent=1)
    print(f"Wrote scratch/proto/paratldr.json: {len(out)} of {len(order)} sections", file=sys.stderr)


if __name__ == "__main__":
    main()
