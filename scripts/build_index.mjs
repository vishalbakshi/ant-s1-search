// Build the search index for s1.pdf: split the filing into chunks by section, encode them, and pack sign bits.
//
// 1. Read every text item on every page with pdf.js, the same library index.html renders with,
//    so the boxes saved here line up with the page in the browser. The font of each line tells
//    headings from body text (see HEADINGS).
// 2. Group lines into paragraphs (a paragraph may continue onto the next page) and paragraphs into
//    sections, one per heading: each risk factor is its own section.
// 3. Pack each section's paragraphs into evenly sized chunks of at most CHUNK_CHARS characters,
//    splitting long paragraphs between sentences. Every chunk starts with its section's headings.
// 4. Encode the chunks with pylate-rs (WebAssembly) in worker threads. A chunk that fills the
//    model's 300-token limit may have been cut off, so it is split in two and encoded again.
// 5. Write data/index.bin (sign bits of every token vector), data/vectors.f16 (the same vectors at
//    half precision, for reranking), and data/docs.json (chunk text and the boxes to highlight).
//
// Usage: node scripts/build_index.mjs [--workers 6] [--split-only] [--model repo@revision --out dir]
//   --split-only  stop after step 3 and write the chunk texts to cache/chunks.json
//   --outline     stop after step 2 and write the sections and their paragraphs to reader/outline.json
//   --model       encode with another PyLate model instead of the one in model.json
//   --out         write index.bin and docs.json there instead of data/
// The float32 vectors are also kept in cache/embeddings/, for comparing models and precisions.
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { availableParallelism } from "node:os";
import { join } from "node:path";
import { isMainThread, parentPort, Worker, workerData } from "node:worker_threads";
import { loadColBERT, MODEL, PYLATE_RS_VERSION, ROOT } from "./lib/colbert.mjs";

const PDF = join(ROOT, "s1.pdf");
const SOURCE = "https://www.sec.gov/Archives/edgar/data/2130357/000162828026067168/anthropicpbc-sx1.htm";
const CHUNK_CHARS = 1100; // headings included; about 250 tokens of prose
const HEADING_CHARS = 450; // longer heading chains keep only the innermost heading
const DOC_LENGTH = 300; // document_length in the model's config_sentence_transformers.json
const BATCH = 16;

// Heading levels by font and size. Body text is Georgia 9.5. Level 1 is a part of the prospectus
// like "Risk Factors", level 2 a group like "Risks Related to Our Business", level 3 a subsection,
// level 4 a risk factor's one-sentence heading, and level 5 a topic within one, like "Electrical
// Power and Fuel Constraints".
const HEADINGS = {
  "Helvetica-Bold 20.0": 1,
  "Georgia-Bold 12.0": 2,
  "Georgia-BoldItalic 10.0": 3,
  "Georgia-BoldItalic 9.5": 4,
  "Georgia-Italic 10.0": 5,
};

if (isMainThread) {
  await main();
} else {
  const { texts, model: which } = workerData;
  const model = await loadColBERT({ batchSize: BATCH, model: which });
  for (let i = 0; i < texts.length; i += BATCH) {
    // One batch per call: pylate-rs pads each batch to its longest text.
    const { embeddings } = model.encode({ sentences: texts.slice(i, i + BATCH) }, false);
    // Padding vectors are all zeros; real ones have unit length.
    parentPort.postMessage(embeddings.map((rows) => rows.filter((row) => row.some((v) => v !== 0))));
  }
}

function round(v) {
  return Math.round(v * 10) / 10;
}

// Lines of a page, top to bottom: { page, text, size, level, x0, x1, top, bottom } in PDF points
// from the page's top left. `level` is the heading level of the line's main font, 0 for body text.
function pageLines(page, items, fonts, height) {
  const rows = new Map(); // baseline (rounded) -> items on it
  for (const it of items) {
    if (!it.str.trim()) continue;
    const [, , , d, x, y] = it.transform;
    const key = Math.round(y);
    if (!rows.has(key)) rows.set(key, []);
    const size = Math.abs(d) || it.height;
    rows.get(key).push({ str: it.str, x, y, w: it.width, size, style: `${fonts(it.fontName)} ${size.toFixed(1)}` });
  }
  const lines = [];
  for (const row of rows.values()) {
    row.sort((a, b) => a.x - b.x);
    let text = "";
    let end = -Infinity;
    const chars = {}; // style -> characters in it
    for (const r of row) {
      if (text && r.x - end > r.size * 0.15 && !text.endsWith(" ")) text += " ";
      text += r.str;
      end = r.x + r.w;
      chars[r.style] = (chars[r.style] ?? 0) + r.str.trim().length;
    }
    const style = Object.keys(chars).reduce((a, b) => (chars[a] >= chars[b] ? a : b));
    const size = Math.max(...row.map((r) => r.size));
    const y = Math.min(...row.map((r) => r.y));
    lines.push({
      page,
      text: text.replace(/\s+/g, " ").trim(),
      size,
      level: HEADINGS[style] ?? 0,
      x0: row[0].x,
      x1: end,
      top: height - y - size * 0.9,
      bottom: height - y + size * 0.25,
    });
  }
  lines.sort((a, b) => a.top - b.top || a.x0 - b.x0);
  // The "Table of Contents" link at the top of each page and the page number at the bottom.
  return lines.filter((l) => !(l.top < 40 && l.text === "Table of Contents") && !(l.top > height - 60 && /^[\w-]{1,6}$/.test(l.text)));
}

// Paragraphs across the whole document: { level, lines }. A paragraph that doesn't end a sentence
// at the bottom of a page continues at the top of the next.
function paragraphs(lines) {
  const out = [];
  let prev = null;
  for (const line of lines) {
    const para = out.at(-1);
    const continues =
      prev &&
      line.level === para.level &&
      !/^[•●▪]/.test(line.text) &&
      (line.page === prev.page
        ? line.top - prev.bottom <= line.size * 0.6
        : line.level === 0 && !/[.:;!?”"]$/.test(prev.text));
    if (continues) para.lines.push(line);
    else out.push({ level: line.level, lines: [line] });
    prev = line;
  }
  return out;
}

// A paragraph's text, with the character range of each line in it. Words hyphenated across lines are rejoined.
function paragraphText(lines) {
  let text = "";
  const ranges = [];
  for (const l of lines) {
    if (!text) text = l.text;
    else if (/[a-z]-$/.test(text)) text = text.slice(0, -1) + l.text;
    else text += ` ${l.text}`;
    ranges.push([text.length - l.text.length, text.length]);
  }
  return { text, ranges };
}

// Sections: { headings, paragraphs }, where headings are the texts of the enclosing headings, outermost first.
function sections(paras) {
  const stack = [];
  const out = [];
  for (const p of paras) {
    if (p.level) {
      stack.length = p.level; // drop deeper headings
      stack[p.level - 1] = paragraphText(p.lines).text;
      out.push({ headings: stack.filter(Boolean), heading: p, paragraphs: [] });
    } else {
      if (!out.length) out.push({ headings: [], heading: null, paragraphs: [] });
      out.at(-1).paragraphs.push(p);
    }
  }
  return out.filter((s) => s.paragraphs.length);
}

// Boxes for characters from..to of a paragraph: [page, left, top, right, bottom] per line, cut in
// proportion to the characters when the range starts or ends partway through a line.
function boxes(lines, ranges, from, to) {
  const out = [];
  lines.forEach((l, k) => {
    const [s, e] = ranges[k];
    if (e <= from || s >= to) return;
    const w = (l.x1 - l.x0) / Math.max(1, e - s);
    out.push([l.page, round(l.x0 + Math.max(0, from - s) * w), round(l.top), round(l.x1 - Math.max(0, e - to) * w), round(l.bottom)]);
  });
  return out;
}

// A paragraph as evenly sized pieces of at most `budget` characters, split between sentences (or,
// within one very long sentence, between words): [{ text, boxes, starts }] where `starts` marks the
// paragraph's first piece.
function pieces(para, budget) {
  const { text, ranges } = paragraphText(para.lines);
  const ends = [...text.matchAll(/[.!?;][”")]*\s+(?=[A-Z“"(•])/g)].map((m) => m.index + m[0].length);
  const cuts = [0];
  for (let start = 0; text.length - start > budget; start = cuts.at(-1)) {
    // Aim for the rest to divide evenly; cut at the sentence end nearest that.
    const ideal = start + (text.length - start) / Math.ceil((text.length - start) / budget);
    const fits = ends.filter((e) => e > start && e - start <= budget);
    if (fits.length) {
      cuts.push(fits.reduce((a, b) => (Math.abs(b - ideal) < Math.abs(a - ideal) ? b : a)));
    } else {
      const space = text.lastIndexOf(" ", start + budget);
      cuts.push(space > start ? space + 1 : start + budget);
    }
  }
  cuts.push(text.length);
  return cuts.slice(1).map((to, k) => ({
    text: text.slice(cuts[k], to).trim(),
    boxes: boxes(para.lines, ranges, cuts[k], to),
    starts: k === 0,
  }));
}

// Pack a section into evenly sized chunks: [{ prefix, pieces }].
function chunkSection(section) {
  let headings = section.headings;
  while (headings.length > 1 && headings.join("\n").length > HEADING_CHARS) headings = headings.slice(1);
  const prefix = headings.join("\n");
  const budget = Math.max(200, CHUNK_CHARS - prefix.length);
  const units = section.paragraphs.flatMap((p) => pieces(p, budget));
  const total = units.reduce((n, u) => n + u.text.length + 1, 0);
  const target = Math.ceil(total / Math.ceil(total / budget));
  const chunks = [];
  let size = 0;
  for (const unit of units) {
    if (!chunks.length || (size && size + unit.text.length > target)) {
      chunks.push({ prefix, pieces: [] });
      size = 0;
    }
    chunks.at(-1).pieces.push(unit);
    size += unit.text.length + 1;
  }
  // The heading itself is highlighted with the section's first chunk.
  if (section.heading) {
    const { ranges } = paragraphText(section.heading.lines);
    chunks[0].headingBoxes = boxes(section.heading.lines, ranges, 0, Infinity);
  }
  return chunks;
}

// Sections with their whole paragraphs, for the reader: [{ headings, level, page, paragraphs: [{ text, page }] }].
function outline(found) {
  return found.map((s) => ({
    headings: s.headings,
    level: s.heading?.level ?? 0,
    page: (s.heading ?? s.paragraphs[0]).lines[0].page,
    paragraphs: s.paragraphs.map((p) => ({ text: paragraphText(p.lines).text, page: p.lines[0].page })),
  }));
}

function chunkText(chunk) {
  const body = chunk.pieces.reduce((t, p) => (!t ? p.text : `${t}${p.starts ? "\n" : " "}${p.text}`), "");
  return chunk.prefix ? `${chunk.prefix}\n${body}` : body;
}

async function extract() {
  const { getDocument } = await import("pdfjs-dist/legacy/build/pdf.mjs");
  const doc = await getDocument({ data: new Uint8Array(await readFile(PDF)), verbosity: 0 }).promise;
  const pages = [];
  const lines = [];
  for (let p = 1; p <= doc.numPages; p++) {
    const page = await doc.getPage(p);
    const { width, height } = page.getViewport({ scale: 1 });
    pages.push([round(width), round(height)]);
    await page.getOperatorList(); // loads the page's fonts, whose names say bold or italic
    const fonts = (id) => page.commonObjs.get(id).name.replace(/^\w+\+/, "");
    const { items } = await page.getTextContent();
    lines.push(...pageLines(p - 1, items, fonts, height));
  }
  const found = sections(paragraphs(lines));
  if (process.argv.includes("--outline")) return { outline: outline(found) };
  const chunks = found
    .flatMap((s, section) => chunkSection(s).map((c) => ({ ...c, section })))
    .filter((c) => c.pieces.map((p) => p.text).join("").replace(/[^A-Za-z]/g, "").length >= 20); // stray labels and numbers
  return { pages, chunks, sections: found.map((s) => s.headings) };
}

// Encode texts across worker threads; returns each text's token vectors in input order.
async function encode(texts, workers, model) {
  const n = Math.max(1, Math.min(workers, Math.ceil(texts.length / BATCH)));
  const size = Math.ceil(texts.length / n);
  const parts = Array.from({ length: n }, () => []);
  let done = 0;
  const start = performance.now();
  await Promise.all(
    parts.map((part, w) => {
      const worker = new Worker(new URL(import.meta.url), { workerData: { texts: texts.slice(w * size, (w + 1) * size), model } });
      worker.on("message", (batch) => {
        part.push(...batch);
        done += batch.length;
        const secs = (performance.now() - start) / 1000;
        process.stderr.write(`\r${done}/${texts.length} chunks, ${(done / secs).toFixed(1)}/s   `);
      });
      return new Promise((resolve, reject) => {
        worker.on("error", reject);
        worker.on("exit", (code) => (code === 0 ? resolve() : reject(new Error(`worker ${w} exited with ${code}`))));
      });
    }),
  );
  process.stderr.write("\n");
  return parts.flat();
}

async function main() {
  const args = process.argv.slice(2);
  const i = args.indexOf("--workers");
  const workers = i >= 0 ? Number(args[i + 1]) : Math.max(1, availableParallelism() - 2);
  const option = (name) => (args.includes(name) ? args[args.indexOf(name) + 1] : null);
  const [repo, revision] = option("--model")?.split("@") ?? [];
  const model = repo ? { repo, revision } : MODEL;
  const out = option("--out") ? join(ROOT, option("--out")) : join(ROOT, "data");

  const extracted = await extract();
  if (extracted.outline) {
    await writeFile(join(ROOT, "reader", "outline.json"), JSON.stringify(extracted.outline));
    console.error(`Wrote reader/outline.json: ${extracted.outline.length} sections`);
    return;
  }
  let { pages, chunks, sections: sectionHeadings } = extracted;
  console.error(`${pages.length} pages, ${chunks.length} chunks`);
  if (args.includes("--split-only")) {
    await mkdir(join(ROOT, "cache"), { recursive: true });
    await writeFile(join(ROOT, "cache", "chunks.json"), JSON.stringify(chunks.map(chunkText), null, 1));
    return;
  }

  // Encode; any chunk at the token limit is split in two between pieces and encoded again.
  let vectors = new Array(chunks.length);
  let todo = chunks.map((_, k) => k);
  while (todo.length) {
    const encoded = await encode(todo.map((k) => chunkText(chunks[k])), workers, model);
    todo.forEach((k, j) => (vectors[k] = encoded[j]));
    const full = new Set(todo.filter((k) => vectors[k].length >= DOC_LENGTH && chunks[k].pieces.length > 1));
    if (!full.size) break;
    console.error(`${full.size} chunks reached ${DOC_LENGTH} tokens: splitting them`);
    const next = [];
    const nextVectors = [];
    chunks.forEach((c, k) => {
      if (!full.has(k)) {
        next.push(c);
        nextVectors.push(vectors[k]);
        return;
      }
      const half = Math.ceil(c.pieces.length / 2);
      next.push({ ...c, pieces: c.pieces.slice(0, half) }, { prefix: c.prefix, section: c.section, pieces: c.pieces.slice(half) });
      nextVectors.push(null, null);
    });
    chunks = next;
    vectors = nextVectors;
    todo = vectors.flatMap((v, k) => (v ? [] : [k]));
  }
  const truncated = vectors.filter((v) => v.length >= DOC_LENGTH).length;
  if (truncated) console.error(`${truncated} single-piece chunks still reach ${DOC_LENGTH} tokens and may be cut off`);

  const dim = vectors[0][0].length;
  const offsets = [0];
  for (const v of vectors) offsets.push(offsets.at(-1) + v.length);
  const bits = new Uint8Array((offsets.at(-1) * dim) / 8);
  let at = 0;
  for (const rows of vectors) {
    for (const row of rows) {
      for (let d = 0; d < dim; d++) if (row[d] > 0) bits[at + (d >> 3)] |= 0x80 >> (d & 7); // most significant bit first
      at += dim / 8;
    }
  }

  const floats = new Float32Array(offsets.at(-1) * dim);
  vectors.flat().forEach((row, k) => floats.set(row, k * dim));
  const emb = join(ROOT, "cache", "embeddings", `${model.repo.replace("/", "__")}@${model.revision}`);
  await mkdir(join(emb, ".."), { recursive: true });
  await writeFile(`${emb}.f32`, floats);
  await mkdir(out, { recursive: true });
  await writeFile(join(out, "index.bin"), bits);
  // Half-precision copies of the same vectors, for reranking: the page fetches only the rows of
  // its shortlisted chunks, with range requests.
  await writeFile(join(out, "vectors.f16"), new Float16Array(floats));
  const docs = {
    format: 2,
    model,
    pylateRs: PYLATE_RS_VERSION,
    dim,
    bits: 1,
    bitOrder: "most significant bit first",
    vectors: "vectors.f16: the same token vectors as float16 (little-endian), dim numbers each, in offsets order",
    source: SOURCE,
    built: new Date().toISOString().slice(0, 10),
    pages, // [width, height] in PDF points
    sections: sectionHeadings, // each section's headings, outermost first; every heading starts a section
    chunks: chunks.map((c) => ({
      section: c.section, // index into sections
      text: chunkText(c), // the section's headings, then the chunk's own text
      // [page (0-based), left, top, right, bottom] in PDF points from the page's top left, one per line.
      boxes: [...(c.headingBoxes ?? []), ...c.pieces.flatMap((p) => p.boxes)],
    })),
    offsets,
  };
  await writeFile(join(out, "docs.json"), JSON.stringify(docs));
  console.error(`Wrote ${chunks.length} chunks, ${offsets.at(-1)} token vectors (${(bits.length / 1e6).toFixed(1)} MB of bits)`);
}
