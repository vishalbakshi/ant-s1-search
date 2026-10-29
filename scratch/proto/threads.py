"""Topic threads: for each section of reader/s1.json, the sections in *other* parts of the filing
that talk about the same thing (TF-IDF cosine over words and word pairs).

    python3 scratch/proto/threads.py      # writes scratch/proto/threads.json
"""
import json
import math
import os
import re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
S1 = json.load(open(os.path.join(HERE, "..", "..", "reader", "s1.json")))
PARAS = S1["paras"]
STOP = set("""the and for that with our are this from which have has will been their such its may any other not
can more than these into under also each all who would could should were was they them those there when where what
including include includes certain upon any related relating within without between through based any applicable
anthropic company companies business our we us of to in on by as or an be is it at""".split())

nodes, part_of, parent = {}, {}, {}


def walk(n, part=None, par=None):
    nodes[n["id"]] = n
    part_of[n["id"]] = part
    parent[n["id"]] = par
    for c in n["children"]:
        walk(c, part or c["id"], n["id"])


for c in S1["root"]["children"]:
    walk(c, c["id"], "root")


def terms(text):
    words = [w for w in re.findall(r"[a-z][a-z0-9\-]{2,}", text.lower()) if w not in STOP]
    return words + [f"{a} {b}" for a, b in zip(words, words[1:])]


def own_text(n):
    return n["title"] + " " + " ".join(PARAS[p][0] for p in n["paras"])


def subtree(n):
    yield n
    for c in n["children"]:
        yield from subtree(c)


counts = {i: Counter(t for m in subtree(n) for t in terms(own_text(m))) for i, n in nodes.items()}
own = {i: Counter(terms(own_text(n))) for i, n in nodes.items()}
df = Counter(t for c in own.values() for t in c)
N = len(own)
idf = {t: math.log(N / d) for t, d in df.items() if 1 < d < 0.4 * N}  # neither unique nor everywhere


def vec(c):
    v = {t: (1 + math.log(k)) * idf[t] for t, k in c.items() if t in idf}
    top = dict(sorted(v.items(), key=lambda kv: -kv[1])[:300])
    norm = math.sqrt(sum(x * x for x in top.values())) or 1
    return {t: x / norm for t, x in top.items()}


vecs = {i: vec(c) for i, c in counts.items()}


def ancestors(i):
    while i in parent and parent[i] != "root":
        i = parent[i]
        yield i


def shared(a, b):
    """The terms that most explain the match, for a label."""
    both = sorted((vecs[a][t] * vecs[b][t], t) for t in vecs[a].keys() & vecs[b].keys())
    out = []
    for _, t in reversed(both):
        if not any(t in o or o in t for o in out):
            out.append(t)
        if len(out) == 3:
            break
    return out


threads = {}
for a in nodes:
    va = vecs[a]
    scores = []
    for b in nodes:
        if part_of[b] == part_of[a] or not nodes[b]["paras"]:
            continue
        s = sum(x * vecs[b].get(t, 0) for t, x in va.items())
        if s > 0.08:
            scores.append((s, b))
    scores.sort(reverse=True)
    picked = []
    for s, b in scores:
        family = set(ancestors(b)) | {b}
        if any(p in family or b in set(ancestors(p)) for _, p in picked):
            continue  # a section and its own subsection say the same thing twice
        picked.append((s, b))
        if len(picked) == 5:
            break
    threads[a] = [{"id": b, "score": round(s, 3), "terms": shared(a, b)} for s, b in picked]

json.dump(threads, open(os.path.join(HERE, "threads.json"), "w"), ensure_ascii=False)
for probe in ["s36", "s35", "s442", "s60"]:
    print(nodes[probe]["title"][:80])
    for t in threads[probe]:
        print(f"   {t['score']:.2f} {nodes[t['id']]['title'][:70]} (p. {nodes[t['id']]['page']}) {t['terms']}")
print(sum(1 for v in threads.values() if v), "of", len(threads), "sections have threads")
