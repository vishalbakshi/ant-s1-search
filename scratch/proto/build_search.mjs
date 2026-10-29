// The search index for relevance.html: the same model and packing as scripts/build_index.mjs, but
// its documents are paragraphs, so every result lands on a section of the outline. Each paragraph is
// indexed twice: by the filing's text (split between sentences if it is long) and by the one sentence
// paratldr.py wrote for it. Every document starts with its section's title.
//
//   .tools/node/bin/node scratch/proto/build_search.mjs     # writes scratch/proto/search/
import { readFileSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { availableParallelism } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { isMainThread, parentPort, Worker, workerData } from "node:worker_threads";
import { loadColBERT, MODEL, PYLATE_RS_VERSION, ROOT } from "../../scripts/lib/colbert.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const PIECE_CHARS = 1000; // about 230 tokens with the title, under the model's 300
const BATCH = 16;

if (!isMainThread) {
  const model = await loadColBERT({ batchSize: BATCH });
  for (let i = 0; i < workerData.length; i += BATCH) {
    const { embeddings } = model.encode({ sentences: workerData.slice(i, i + BATCH) }, false);
    parentPort.postMessage(embeddings.map((rows) => rows.filter((row) => row.some((v) => v !== 0))));
  }
} else {
  await main();
}

// a long paragraph in pieces of at most PIECE_CHARS, cut between sentences
function pieces(text) {
  const out = [""];
  for (const s of text.match(/[^.!?]+[.!?]*\s*/g) ?? [text]) {
    if (out.at(-1) && out.at(-1).length + s.length > PIECE_CHARS) out.push("");
    out[out.length - 1] += s;
  }
  return out.map((p) => p.trim()).filter(Boolean);
}

async function encode(texts) {
  const n = Math.max(1, Math.min(availableParallelism() - 2, Math.ceil(texts.length / BATCH)));
  const size = Math.ceil(texts.length / n);
  let done = 0;
  const parts = await Promise.all(Array.from({ length: n }, (_, w) => new Promise((resolve, reject) => {
    const part = [];
    const worker = new Worker(new URL(import.meta.url), { workerData: texts.slice(w * size, (w + 1) * size) });
    worker.on("message", (batch) => { part.push(...batch); done += batch.length; process.stderr.write(`\r${done}/${texts.length}`); });
    worker.on("error", reject);
    worker.on("exit", (code) => (code === 0 ? resolve(part) : reject(new Error(`worker ${w} exited with ${code}`))));
  })));
  process.stderr.write("\n");
  return parts.flat();
}

async function main() {
  const s1 = JSON.parse(readFileSync(join(ROOT, "reader", "s1.json"), "utf8"));
  const tl = JSON.parse(readFileSync(join(HERE, "paratldr.json"), "utf8"));
  const docs = []; // [paragraph id, "t" for the filing's text or "s" for the sentence, text]
  const walk = (n) => {
    for (const p of n.paras) {
      const title = n.title.slice(0, 200);
      for (const piece of pieces(s1.paras[p][0])) docs.push([p, "t", `${title}\n${piece}`]);
      const sentence = tl[n.id]?.paras?.[p];
      if (sentence) docs.push([p, "s", `${title}\n${sentence}`]);
    }
    n.children.forEach(walk);
  };
  walk(s1.root);
  console.error(`${docs.length} documents`);

  const vectors = await encode(docs.map((d) => d[2]));
  const dim = vectors[0][0].length;
  const offsets = [0];
  for (const v of vectors) offsets.push(offsets.at(-1) + v.length);
  const bits = new Uint8Array((offsets.at(-1) * dim) / 8);
  let at = 0;
  for (const rows of vectors) for (const row of rows) {
    for (let d = 0; d < dim; d++) if (row[d] > 0) bits[at + (d >> 3)] |= 0x80 >> (d & 7);
    at += dim / 8;
  }
  const floats = new Float32Array(offsets.at(-1) * dim);
  vectors.flat().forEach((row, k) => floats.set(row, k * dim));

  const out = join(HERE, "search");
  await mkdir(out, { recursive: true });
  await writeFile(join(out, "index.bin"), bits);
  await writeFile(join(out, "vectors.f16"), new Float16Array(floats));
  await writeFile(join(out, "docs.json"), JSON.stringify({ model: MODEL, pylateRs: PYLATE_RS_VERSION, dim, docs, offsets }));
  console.error(`Wrote scratch/proto/search/: ${docs.length} documents, ${offsets.at(-1)} token vectors`);
}
