"""A TLDR of every section for the public: what matters to someone deciding whether to invest, or
wondering how this company and its IPO touch the wider economy. Same bottom-up scheme as
reader/tools/distill.py: small subtrees are read whole, big ones from their subsections' TLDRs.

    python3 scratch/proto/public_tldr.py      # writes scratch/proto/public.json; answers cached in scratch/proto/build/
"""
import concurrent.futures as cf
import hashlib
import json
import os
import re
import subprocess
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "reader", "tools"))
import distill as d  # noqa: E402  (tree building, rendering and the claude CLI path)

BUILD = os.path.join(HERE, "build")
JOBS = 6

SYSTEM = """You write TLDRs of parts of Anthropic, PBC's Form S-1 (its October 2026 IPO registration statement) for members of the public: people deciding whether to invest, and people wondering how this company and its IPO affect jobs, markets and the wider economy.

For each section you are asked about, write what such a reader should take away from the WHOLE section, including all of its subsections: the main points, not a description of what the section contains. 1 to 3 short sentences, at most 60 words. Lead with the most consequential point. Use concrete numbers, dates and names when they matter, exact and with their periods. If a section is a legal formality with little substance, say so in a few words and give the one point that matters, if any.

Write plainly: everyday words, complete sentences, no dashes, no jargon without a gloss, no hype. Say "Anthropic", not "we". Do not tell anyone to buy or sell, and do not predict the stock price. Tables come through as runs of numbers and labels; only state figures you are sure of.

Answer with only a JSON object, no code fence: {"<section id>": "<tldr>", ...} with an entry for every section id you are asked about."""

slots = threading.Semaphore(JOBS)
lock = threading.Lock()


def ask(prompt, ids):
    key = hashlib.sha256((SYSTEM + prompt).encode()).hexdigest()[:20]
    path = os.path.join(BUILD, key + ".json")
    if os.path.exists(path):
        return json.load(open(path))
    cmd = [d.CLAUDE, "-p", "--output-format", "json", "--no-session-persistence", "--tools", "", "--system-prompt", SYSTEM]
    for attempt in range(3):
        with slots:
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=BUILD)
        try:
            text = json.loads(proc.stdout)["result"].strip()
            answer = json.loads(text[text.index("{"):text.rindex("}") + 1])
            missing = [i for i in ids if not isinstance(answer.get(i), str) or not answer[i].strip()]
            if missing:
                raise ValueError(f"missing {missing[:5]}")
            json.dump(answer, open(path, "w"), indent=1, ensure_ascii=False)
            return answer
        except Exception as e:  # noqa: BLE001
            print(f"  retry {attempt + 1} for {ids[0]}: {e} {proc.stderr[-300:]}", file=sys.stderr)
    raise RuntimeError(f"no answer for {ids[0]}")


def main():
    os.makedirs(BUILD, exist_ok=True)
    root, nodes, para, _ = d.build_tree()
    done = {}

    def paths(node, trail):
        node["_path"] = " > ".join(trail) or "(top level)"
        for c in node["children"]:
            paths(c, trail + [node["title"]])

    def report(answer, ids):
        with lock:
            for i in ids:
                done[i] = d.billions(answer[i].strip())
            print(f"{len(done)}/{len(nodes)} sections", file=sys.stderr)

    def batch(group):
        ids = [i for n in group for i in d.heading_ids(n)]
        prompt = (f"Write the TLDR for every section below: {', '.join(ids)}. The parts above these are: {group[0]['_path']}.\n\n"
                  + "\n\n".join(d.render_full(n, para) for n in group))
        report(ask(prompt, ids), ids)

    def digest(node):
        out = [f"# [{node['id']}] {node['title']}"]
        out += [f"[{p}] {para[p]['text']}" for p in node["paras"]]
        for c in node["children"]:
            out.append(f"## {c['title']}\n(Subsection, given by its TLDR.) {done[c['id']]}")
        return "\n\n".join(out)

    def distill(node):
        groups, filled, jobs = [], d.LIMIT, []
        for c in node["children"]:
            n = d.size(c, para)
            if n > d.LIMIT:
                jobs.append((distill, c))
                continue
            if filled + n > d.LIMIT:
                groups.append([])
                filled = 0
            groups[-1].append(c)
            filled += n
        jobs += [(batch, g) for g in groups]
        with cf.ThreadPoolExecutor(max(1, len(jobs))) as kids:
            list(kids.map(lambda job: job[0](job[1]), jobs))
        prompt = (f"Write the TLDR for section {node['id']} only, covering everything in it. The parts above this one are: "
                  f"{node['_path']}.\n\n{digest(node)}")
        report(ask(prompt, [node["id"]]), [node["id"]])

    paths(root, [])
    distill(root)
    json.dump(done, open(os.path.join(HERE, "public.json"), "w"), ensure_ascii=False)
    print(f"Wrote scratch/proto/public.json: {len(done)} sections", file=sys.stderr)


if __name__ == "__main__":
    main()
