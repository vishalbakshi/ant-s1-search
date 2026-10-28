"""Local server for the S-1 reader, with highlight-to-ask.

Serves the repo on http://localhost:8771 (the reader is at /reader/, next to ../s1.pdf) and answers
questions by running the `claude` CLI in print mode, so it uses your Claude Code login (no API key).

    python3 reader/server.py            # then open http://localhost:8771/reader/
    CLAUDE_MODEL=sonnet python3 reader/server.py

Asks, notes and highlights are saved in reader/annotations.json as character offsets into a block of
the reader (a paragraph of the filing, or a section's TLDR or summary), so they hold as long as
reader/s1.json is the same.
"""
import json
import os
import re
import shutil
import subprocess
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(HERE, "build", "ask")   # the claude CLI runs here, next to the filing as text
THREADS = os.path.join(HERE, "annotations.json")
PORT = int(os.environ.get("PORT", 8771))
MODEL = os.environ.get("CLAUDE_MODEL")  # e.g. "sonnet"; default = your Claude Code default
# Claude can search the web, and read and grep the filing text in DATA (permissions keep it to that folder)
TOOLS = "Read,Grep,WebSearch,WebFetch"
ALLOWED = f"Read({DATA}/**),Grep({DATA}/**),WebSearch,WebFetch"
CLAUDE = shutil.which("claude") or os.path.expanduser("~/.local/bin/claude")
CONTEXT_CHARS = 120_000  # filing text sent with a question, around the passage

os.makedirs(DATA, exist_ok=True)
save_lock = threading.Lock()

S1 = json.load(open(os.path.join(HERE, "s1.json")))
PARAS = S1["paras"]
NODES, PARENT, OWNER = {}, {}, {}


def index(node, parent=None):
    NODES[node["id"]] = node
    PARENT[node["id"]] = parent
    for p in node["paras"]:
        OWNER[p] = node["id"]
    for c in node["children"]:
        index(c, node["id"])


index(S1["root"])


def write_filing(node, out, depth=0):
    if node["id"] != "root":
        out.append(f"{'#' * min(depth, 6)} {node['title']} (page {node['page']})")
    for p in node["paras"]:
        out.append(f"[page {PARAS[p][1]}] {PARAS[p][0]}")
    for c in node["children"]:
        write_filing(c, out, depth + 1)
    return out


with open(os.path.join(DATA, "s1.md"), "w") as f:
    f.write("\n\n".join(write_filing(S1["root"], ["# Anthropic, PBC Form S-1"])))


def plain(summary):
    return re.sub(r"\[\[(.+?)\|[ps]\d+\]\]", r"\1", summary)


def path(nid):
    out = []
    while nid and nid != "root":
        out.append(NODES[nid]["title"])
        nid = PARENT[nid]
    return " > ".join(reversed(out))


def full_text(node, budget):
    """The section's own paragraphs and its subsections', in order, up to budget characters."""
    out = [f"## {node['title']} (page {node['page']})"]
    budget -= len(out[0])
    for p in node["paras"]:
        if budget <= 0:
            break
        out.append(PARAS[p][0])
        budget -= len(PARAS[p][0])
    for c in node["children"]:
        if budget <= 0:
            out.append("[... the rest of this section is left out for length ...]")
            break
        text = full_text(c, budget)
        out.append(text)
        budget -= len(text)
    return "\n\n".join(out)


def parts_map():
    return "\n".join(f"- {c['title']} (page {c['page']}): {c.get('tldr', '')}" for c in S1["root"]["children"])


SYSTEM = f"""You are a reading companion inside a local reader for Anthropic, PBC's Form S-1, the registration statement for its IPO, filed with the SEC on October 26, 2026. The reader zooms from one-sentence TLDRs, to one-paragraph summaries written by Claude, down to the filing's own paragraphs.

The reader highlights a passage and asks about it. Write ~90% caveman: short words, short clipped sentences, drop most articles and filler. But keep enough small words ("the", "is", "because", "but") that each sentence reads smooth on first pass; clarity beats purity. No preamble, no restating question, no pleasantries.
Facts must be right. Ground answers in the filing and quote it when it settles the question; say which section and page. You are given the section around the passage. The whole filing is in s1.md in your working directory (headings with page numbers, then each paragraph as "[page N] text"); Grep it and Read the parts you need whenever the answer may be elsewhere in the filing, rather than guessing. If the passage is from a summary, check it against the filing text and say plainly if the summary got something wrong. Say "guess" when guessing. You can search the web and fetch pages: do it when the reader asks you to or when the answer needs facts outside the filing, and name the sources you used. Usually 1-5 lines. Longer only if reader ask. Plain text; bullet list only if truly list. This is not investment advice and you should not give any.

The filing's parts, with their one-sentence TLDRs:
{parts_map()}"""


def build_prompt(req):
    block = req.get("block", "")
    nid = OWNER.get(block) or block.split(".")[0]
    node = NODES.get(nid, S1["root"])
    where = "a paragraph of the filing" if block in PARAS else ("a TLDR written by Claude" if block.endswith(".tldr") else "a summary written by Claude")
    parts = [f"Highlighted passage, from {where} in the section \"{path(node['id']) or 'the whole filing'}\":\n\"\"\"\n{req['quote']}\n\"\"\""]
    if node.get("summary"):
        parts.append(f"Claude's summary of that section:\n\"\"\"\n{plain(node['summary'])}\n\"\"\"")
    if node["id"] != "root":
        parts.append(f"The section's text in the filing:\n\"\"\"\n{full_text(node, CONTEXT_CHARS)}\n\"\"\"")
    history = req.get("messages", [])
    if len(history) > 1:
        who = {"user": "Reader", "note": "Reader's note", "assistant": "You"}
        convo = "\n\n".join(f"{who.get(m['role'], 'Reader')}: {m['content']}" for m in history[:-1])
        parts.append(f"The reader's notes and conversation so far about this passage:\n{convo}")
    parts.append(f"Reader's question: {history[-1]['content']}")
    return "\n\n".join(parts)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def log_message(self, fmt, *args):
        if "/api/" in (args[0] if args else ""):
            super().log_message(fmt, *args)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def read_json(self):
        n = int(self.headers.get("Content-Length", 0))
        if n > 5_000_000:
            raise ValueError("body too large")
        return json.loads(self.rfile.read(n) or b"null")

    def send_json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self.send_response(302)
            self.send_header("Location", "/reader/")
            return self.end_headers()
        if self.path == "/api/threads":
            try:
                return self.send_json(json.load(open(THREADS)))
            except FileNotFoundError:
                return self.send_json([])
        return super().do_GET()

    def do_PUT(self):
        if self.path != "/api/threads":
            return self.send_error(404)
        # the page rebuilds quotes from the reader's text, so keep the file free of filing text
        threads = [{k: v for k, v in t.items() if k != "quote"} for t in self.read_json()]
        with save_lock:
            tmp = THREADS + ".tmp"
            json.dump(threads, open(tmp, "w"), indent=1, ensure_ascii=False)
            os.replace(tmp, THREADS)
        self.send_json({"ok": True})

    def do_POST(self):
        if self.path != "/api/ask":
            return self.send_error(404)
        req = self.read_json()
        # web search, page fetching, and reading the filing: no shell, no other files
        cmd = [CLAUDE, "-p", "--output-format", "stream-json", "--verbose", "--include-partial-messages",
               "--no-session-persistence", "--tools", TOOLS, "--allowedTools", ALLOWED, "--system-prompt", SYSTEM]
        if MODEL:
            cmd += ["--model", MODEL]

        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()

        proc = subprocess.Popen(cmd, cwd=DATA, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, bufsize=1)
        proc.stdin.write(build_prompt(req))
        proc.stdin.close()
        streamed = False
        try:
            for line in proc.stdout:
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = None
                if ev.get("type") == "stream_event":
                    e = ev.get("event", {})
                    if e.get("type") == "content_block_delta" and e.get("delta", {}).get("type") == "text_delta":
                        text = e["delta"]["text"]
                    elif e.get("type") == "content_block_start" and e.get("content_block", {}).get("type") == "text" and streamed:
                        text = "\n\n"   # text before and after a web search: keep them as separate paragraphs
                elif ev.get("type") == "result" and not streamed:
                    text = ev.get("result") or ""
                if text:
                    streamed = True
                    self.wfile.write(text.encode())
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            proc.kill()  # reader closed the tooltip mid-answer
            return
        proc.wait()
        if not streamed:
            err = proc.stderr.read().strip()[-800:] or f"claude exited with code {proc.returncode}"
            self.wfile.write(f"\n\n**Error from claude CLI:** {err}".encode())


if __name__ == "__main__":
    print(f"Reading at http://localhost:{PORT}/reader/  (claude: {CLAUDE}{', model ' + MODEL if MODEL else ''})")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
