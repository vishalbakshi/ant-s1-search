"""Distill the S-1 outline into a tree of summaries for the reader.

Reads reader/outline.json (from `node scripts/build_index.mjs --outline`) and writes reader/s1.json:
every heading becomes a node with a one-sentence TLDR and a one-paragraph summary, and the summary's
key phrases link to the paragraphs or subsections they came from, written [[phrase|p123]] or [[phrase|s45]].

Summaries are made bottom-up by the `claude` CLI in print mode, so they use your Claude Code login.
A subtree small enough to read at once is summarized in one call; a bigger one is summarized from its
children's summaries plus its own paragraphs. Each call's answer is cached in reader/build/, so a rerun
only redoes what changed.

    python3 reader/tools/distill.py              # build everything (6 calls at a time)
    python3 reader/tools/distill.py --jobs 3
    CLAUDE_MODEL=sonnet python3 reader/tools/distill.py
"""
import concurrent.futures as cf
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "build")
LIMIT = 45_000  # characters of filing text one call reads at most
MODEL = os.environ.get("CLAUDE_MODEL")
CLAUDE = shutil.which("claude") or os.path.expanduser("~/.local/bin/claude")
JOBS = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 6

# Headings in the PDF that name a part badly.
TITLES = {"Report Of Independent Registered Public Accounting Firm": "Financial Statements and Notes"}
SKIP = {"Table of Contents"}

SYSTEM = """You distill parts of Anthropic, PBC's Form S-1 (its October 2026 IPO registration statement) for a reader that lets people zoom from a one-line table of contents down to the original paragraphs.

For each section you are asked about, write:
- "tldr": one sentence, at most 30 words, saying the most important thing in the section. Lead with the substance (a number, a claim, a risk), not "This section describes".
- "summary": one paragraph that distills the section. Length follows the section: one or two sentences for a short section, up to about 150 words for a long one. Never longer than the section itself.

In each summary, link the key phrases to where they came from, written [[phrase|id]]: the phrase is a few words of your summary (a number, a name, a claim), and id is the id of one of that section's own paragraphs (p…) or direct subsections (s…), as given in the input. Link 2 to 8 phrases per summary, fewer for short sections; link the facts a reader would most want to check. Only use ids listed directly under that section.

Write plainly: everyday words, complete sentences, no dashes, no hype, no hedging the filing doesn't make. Keep numbers, dates and names exact, with their units and periods ("$4.1 billion of revenue in the six months ended June 30, 2026"). Say "Anthropic" or "the company", not "we". Tables come through as runs of numbers and labels; read them carefully and only state figures you are sure of.

Answer with only a JSON object, no code fence: {"<section id>": {"tldr": "...", "summary": "..."}, ...} with an entry for every section id you are asked about."""


def build_tree():
    outline = json.load(open(os.path.join(ROOT, "outline.json")))
    root = {"id": "root", "title": "Anthropic, PBC Form S-1", "children": [], "paras": [], "page": 0}
    nodes = {"root": root}
    paras = []
    by_path = {(): root}
    for sec in outline:
        path = tuple(sec["headings"]) or ("Cover Page",)
        if path[0] in SKIP:
            continue
        for depth in range(1, len(path) + 1):
            key = path[:depth]
            if key not in by_path:
                node = {"id": f"s{len(nodes)}", "title": TITLES.get(key[-1], key[-1]), "children": [], "paras": [],
                        "page": sec["page"] + 1}
                nodes[node["id"]] = node
                by_path[key] = node
                by_path[key[:-1]]["children"].append(node)
        for p in sec["paragraphs"]:
            pid = f"p{len(paras)}"
            paras.append({"id": pid, "text": p["text"], "page": p["page"] + 1})
            by_path[path]["paras"].append(pid)
    return root, nodes, {p["id"]: p for p in paras}, paras


def size(node, para):
    return sum(len(para[p]["text"]) for p in node["paras"]) + sum(size(c, para) for c in node["children"])


def render_full(node, para, depth=1):
    """The subtree as text, with every section and paragraph id."""
    out = [f"{'#' * min(depth, 6)} [{node['id']}] {node['title']}"]
    out += [f"[{p}] {para[p]['text']}" for p in node["paras"]]
    out += [render_full(c, para, depth + 1) for c in node["children"]]
    return "\n\n".join(out)


def render_digest(node, para, done):
    """A section's own paragraphs in full, and its subsections by their summaries."""
    out = [f"# [{node['id']}] {node['title']}"]
    out += [f"[{p}] {para[p]['text']}" for p in node["paras"]]
    for c in node["children"]:
        d = done[c["id"]]
        out.append(f"## [{c['id']}] {c['title']}\n(Subsection, given here by its summary.) TLDR: {d['tldr']}\nSummary: {strip_links(d['summary'])}")
    return "\n\n".join(out)


def strip_links(s):
    return re.sub(r"\[\[(.+?)\|[ps]\d+\]\]", r"\1", s)


def heading_ids(node):
    yield node["id"]
    for c in node["children"]:
        yield from heading_ids(c)


def clean_links(summary, node):
    """Keep links to the section's own paragraphs and direct subsections; unlink the rest."""
    ok = set(node["paras"]) | {c["id"] for c in node["children"]}
    return re.sub(r"\[\[(.+?)\|([ps]\d+)\]\]", lambda m: m.group(0) if m.group(2) in ok else m.group(1), summary)


lock = threading.Lock()
slots = threading.Semaphore(JOBS)


def ask(prompt, ids):
    key = hashlib.sha256((SYSTEM + prompt + (MODEL or "")).encode()).hexdigest()[:20]
    path = os.path.join(BUILD, key + ".json")
    if os.path.exists(path):
        return json.load(open(path))
    cmd = [CLAUDE, "-p", "--output-format", "json", "--no-session-persistence", "--tools", "",
           "--system-prompt", SYSTEM]
    if MODEL:
        cmd += ["--model", MODEL]
    for attempt in range(3):
        with slots:
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=BUILD)
        try:
            text = json.loads(proc.stdout)["result"].strip()
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
            answer = json.loads(text[text.index("{"):text.rindex("}") + 1])
            missing = [i for i in ids if i not in answer or not answer[i].get("tldr") or not answer[i].get("summary")]
            if missing:
                raise ValueError(f"missing {missing[:5]}")
            json.dump(answer, open(path, "w"), indent=1, ensure_ascii=False)
            return answer
        except Exception as e:  # noqa: BLE001
            print(f"  retry {attempt + 1} for {ids[0]}: {e} {proc.stderr[-300:]}", file=sys.stderr)
    raise RuntimeError(f"no answer for {ids[0]}")


def main():
    os.makedirs(BUILD, exist_ok=True)
    root, nodes, para, paras = build_tree()
    done = {}
    total = len(nodes)

    def report(answer, subtree):
        with lock:
            for i in subtree:
                done[i] = {"tldr": answer[i]["tldr"].strip(), "summary": clean_links(answer[i]["summary"].strip(), nodes[i])}
            print(f"{len(done)}/{total} sections", file=sys.stderr)

    def batch(group):
        """Summarize small sibling subtrees, read in full, in one call."""
        ids = [i for n in group for i in heading_ids(n)]
        prompt = (f"Summarize every section below: {', '.join(ids)}. The parts above these are: {group[0]['_path']}.\n\n"
                  + "\n\n".join(render_full(n, para) for n in group))
        report(ask(prompt, ids), ids)

    def distill(node):
        """Summarize everything under node, then node itself from its subsections' summaries."""
        groups, size_so_far, jobs = [], LIMIT, []
        for c in node["children"]:
            n = size(c, para)
            if n > LIMIT:
                jobs.append((distill, c))
                continue
            if size_so_far + n > LIMIT:
                groups.append([])
                size_so_far = 0
            groups[-1].append(c)
            size_so_far += n
        jobs += [(batch, g) for g in groups]
        with cf.ThreadPoolExecutor(max(1, len(jobs))) as kids:
            list(kids.map(lambda job: job[0](job[1]), jobs))
        prompt = (f"Summarize section {node['id']} only. The parts above this one are: {node['_path']}.\n\n"
                  f"{render_digest(node, para, done)}")
        report(ask(prompt, [node["id"]]), [node["id"]])

    def paths(node, trail):
        node["_path"] = " > ".join(trail) or "(top level)"
        for c in node["children"]:
            paths(c, trail + [node["title"]])

    paths(root, [])
    # One thread per subtree that waits; at most JOBS claude calls run at once. The root's summary
    # is the filing as a whole, linking to its parts.
    distill(root)

    def out(node):
        d = done.get(node["id"], {})
        return {"id": node["id"], "title": node["title"], "page": node["page"], **d,
                "paras": node["paras"], "children": [out(c) for c in node["children"]]}

    doc = {"source": "https://www.sec.gov/Archives/edgar/data/2130357/000162828026067168/anthropicpbc-sx1.htm",
           "root": out(root), "paras": {p["id"]: [p["text"], p["page"]] for p in paras}}
    json.dump(doc, open(os.path.join(ROOT, "s1.json"), "w"), ensure_ascii=False, separators=(",", ":"))
    print(f"Wrote reader/s1.json: {len(done)} sections, {len(paras)} paragraphs", file=sys.stderr)


if __name__ == "__main__":
    main()
