# S-1 reader

> "More detailed information could be displayed when facts had to be pinpointed; more global information could be displayed to answer questions of relevance and interrelationship." (Licklider and Taylor, 1968)

A zoomable reader for Anthropic's S-1. Every heading in the filing has a one-sentence TLDR and a one-paragraph summary written by Claude. Open a section and its summary takes over; dotted phrases in a summary open the filing paragraph they came from, solid ones open a subsection, and every section can show the filing's own text.

| Key | What it does |
|---|---|
| 1 / 2 | Zoom: the parts' TLDRs, or the parts' summaries |
| / | Find a word or phrase in the filing |
| A | Ask Claude about the selected text (cyan) |
| N | Write a note (navy); in a note's chat, click Ask to send a question along with your notes |
| H | Highlight (yellow) |
| X | Clear the marks on the selection |
| Esc | Close pop-ups |

## Run it

The page works as static files (read-only, with my saved annotations). To ask Claude, take notes and highlight, run the local server. It needs Python 3 and [Claude Code](https://claude.com/claude-code), logged in; questions go through the `claude` command, so no API key is needed.

```bash
python3 reader/server.py
```

Then open http://localhost:8771/summary.html. Claude gets the section you're reading (its summary and the filing's text) with every question, and can search the web.

## Rebuild the summaries

```bash
node scripts/build_index.mjs --outline   # reader/outline.json: the filing's headings and paragraphs, from s1.pdf
python3 reader/tools/distill.py          # reader/s1.json: TLDRs and summaries, bottom-up, with the claude CLI
```

`distill.py` reads small sections whole (up to 45,000 characters at a time) and summarizes bigger ones from their subsections' summaries plus their own paragraphs. Answers are cached in `reader/build/`, so a rerun only redoes what changed.
