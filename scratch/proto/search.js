// Hybrid search with rerank over the filing's paragraphs, shared by relevance.html and eval.html.
// As on the hosted search page: ColBERT and BM25 each rank the documents, reciprocal rank fusion
// merges the lists, and the top RERANK are reranked by exact MaxSim on their full vectors. The
// documents are paragraphs, each twice (build_search.mjs): the filing's text, searched with the query
// in the filing's voice ("we"), and the sentence written for it, searched with the query as typed (the
// sentences say "Anthropic"). Numbers in the query add a third list, matched by value across units,
// so "$16B" finds "16,286" in a table given in millions. The best TOP paragraphs match; a section's
// rank is its best paragraph's.
//
//   prepare(s1, paratldr)  once the data is loaded
//   await load(say)        the index and the model; say(text) reports progress
//   await search(text)     { text, terms, nums, rank, via, secs, ms }, or null if a newer search started
//   mark(text, q), snippet(text, q)   HTML with the query's words and numbers marked
import init, { ColBERT } from 'https://cdn.jsdelivr.net/npm/pylate-rs@1.0.4/pylate_rs.js';
import { lookupTables, tokenScores } from '../../scorer.js';
import { ourVoice } from '../../query.js';
import { bm25 } from '../../fts.js';
import { rerank, rrf, vectorStore } from '../../hybrid.js';

const FUSE_DEPTH = 100, RERANK = 50, TOP = 15;
const MODEL_FILES = ['model.safetensors', '1_Dense/model.safetensors', 'tokenizer.json', 'config.json',
  'config_sentence_transformers.json', '1_Dense/config.json', 'special_tokens_map.json'];
const here = path => new URL(path, import.meta.url).href;
let ENGINE = null;
export const HAY = {}, PSEC = {};   // paragraph id -> { s: its sentence, t: the filing's text }, and -> its section
const VALS = {};
const STOP = new Set(('a an and are as at be been but by can could did do does for from had has have how i if in into is it its ' +
  'me my no not of on or our out so than that the their them then there these they this those to too up us was we were what ' +
  'when where which while who whom why will with would you your about any much many more most should just also very').split(' '));
const UNITS = { thousand: 1e3, million: 1e6, billion: 1e9, trillion: 1e12, b: 1e9, m: 1e6 };
const NUM = /(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s*(%|thousand|million|billion|trillion|B\b|M\b)?/gi;
const reEsc = s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
export const esc = s => s.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

export function prepare(s1, paratldr) {
  const walk = n => {
    for (const p of n.paras) { PSEC[p] = n.id; HAY[p] = { s: paratldr[n.id]?.paras?.[p] || '', t: s1.paras[p][0] }; }
    n.children.forEach(walk);
  };
  walk(s1.root);
}
// a number the query asks about, or null when it is a year or a small count (searched as a word instead)
function claim(m) {
  const whole = m[1], frac = m[2] || '', unit = (m[3] || '').toLowerCase();
  const v = parseFloat(whole.replace(/,/g, '') + frac);
  if (unit !== '%' && !frac && (v < 10 || (v >= 1900 && v <= 2100))) return null;
  const scale = UNITS[unit] || 1;
  return { v: v * scale, tol: frac ? 0.5 * 10 ** -(frac.length - 1) * scale : scale > 1 ? 0.5 * scale : 0, pct: unit === '%', unitless: !unit };
}
// every value a text could mean: as written, and scaled the way a table in thousands, millions or billions would be
function values(text) {
  const vals = [], pcts = [], raw = [];
  for (const m of text.matchAll(NUM)) {
    const v = parseFloat(m[1].replace(/,/g, '') + (m[2] || '')), u = (m[3] || '').toLowerCase();
    raw.push(v);
    if (u === '%') { pcts.push(v); continue; }
    if (UNITS[u]) vals.push(v * UNITS[u]); else vals.push(v, v * 1e3, v * 1e6, v * 1e9);
    pcts.push(v);   // tables often give percentages without the sign
  }
  return { vals, pcts, raw };
}
const near = (c, x) => Math.abs(c.v - x) <= Math.max(c.tol, 1e-9 * Math.abs(x));
function hasNum(c, src) { return (c.pct ? src.pcts : c.unitless ? [...src.vals, ...src.raw] : src.vals).some(x => near(c, x)); }
function valsOf(p) { return VALS[p] ||= { s: values(HAY[p].s), t: values(HAY[p].t) }; }
// a word matches its forms ("outvoted" finds "outvote", "subsidiaries" finds "subsidiary"); a "quoted phrase" matches as written
function wordRe(w) {
  const stem = w.length > 4 ? w.replace(/(ies|es|s|ing|ed|ly)$/, '') : w;
  return new RegExp('\\b' + reEsc(stem.length >= 3 ? stem : w) + '\\w*', 'i');
}
function parseQuery(text) {
  const terms = [], nums = [];
  text = text.replace(/"([^"]+)"/g, (_, ph) => { terms.push({ re: new RegExp('\\b' + reEsc(ph.trim()) + '\\b', 'i') }); return ' '; });
  text = text.replace(/\$?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s*(%|thousand|million|billion|trillion|B\b|M\b)?/gi, (...m) => {
    const c = claim([m[0], m[1], m[2], m[3]]);
    if (!c) return ' ' + m[1] + ' ';
    nums.push(c); return ' ';
  });
  for (const w of text.toLowerCase().match(/[a-z0-9][a-z0-9'-]*/g) || [])
    if (!STOP.has(w) && !terms.some(t => t.w === w)) terms.push({ w, re: wordRe(w) });
  return { terms, nums };
}

// model files are pinned to a revision, so a cached copy never goes stale
async function modelFile(url, onProgress) {
  let cache;
  try { cache = await caches.open('ant-s1-search'); } catch {}
  const hit = await cache?.match(url);
  if (hit) return new Uint8Array(await hit.arrayBuffer());
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
  const total = Number(res.headers.get('content-length')) || 0, reader = res.body.getReader(), parts = [];
  let got = 0;
  for (;;) { const { done, value } = await reader.read(); if (done) break; parts.push(value); got += value.length; onProgress(got, total); }
  const bytes = new Uint8Array(got);
  let at = 0;
  for (const p of parts) { bytes.set(p, at); at += p.length; }
  await cache?.put(url, new Response(bytes)).catch(() => {});
  return bytes;
}
export async function load(say = () => {}) {
  say('loading the index…');
  const [ix, bits] = await Promise.all([
    fetch(here('search/docs.json')).then(r => r.json()),
    fetch(here('search/index.bin')).then(r => r.arrayBuffer()).then(b => new Uint8Array(b)),
  ]);
  const files = [];
  for (const name of MODEL_FILES)
    files.push(await modelFile(`https://huggingface.co/${ix.model.repo}/resolve/${ix.model.revision}/${name}`,
      (got, total) => say(`downloading the model, first visit only: ${(got / 1e6).toFixed(0)}${total ? ' of ' + (total / 1e6).toFixed(0) : ''} MB`)));
  say('starting the model…');
  await init();
  const offsets = Int32Array.from(ix.offsets);
  const kind = ix.docs.map(d => d[1]), pid = ix.docs.map(d => d[0]);
  const of = k => ix.docs.flatMap((d, i) => d[1] === k ? [i] : []);
  const ids = { s: of('s'), t: of('t') };
  const words = { s: bm25(ids.s.map(i => ix.docs[i][2])), t: bm25(ids.t.map(i => ix.docs[i][2])) };
  ENGINE = { model: new ColBERT(...files, 32), bits, offsets, dim: ix.dim, kind, pid, ids, words, vectors: vectorStore(here('search/vectors.f16'), offsets, ix.dim) };
  say('');
}
const encode = text => ENGINE.model.encode({ sentences: [text] }, true).embeddings[0];

let seq = 0;
export async function search(text) {
  const mine = ++seq;
  const { terms, nums } = parseQuery(text.trim());
  await new Promise(r => setTimeout(r, 0));
  const t0 = performance.now(), E = ENGINE;
  const voiced = ourVoice(text);
  const qs = encode(text), qt = voiced === text ? qs : encode(voiced);
  // ColBERT on the sign bits, best FUSE_DEPTH of each kind; each query scores only its own kind's documents
  const colbert = (q, k) => {
    const tables = lookupTables(Float32Array.from(q.flat()), E.dim), scores = new Map();
    for (const i of E.ids[k]) scores.set(i, tokenScores(tables, E.bits, E.offsets, i).reduce((a, b) => a + b, 0));
    return E.ids[k].slice().sort((a, b) => scores.get(b) - scores.get(a)).slice(0, FUSE_DEPTH);
  };
  const keyword = (q, k) => E.words[k](q, FUSE_DEPTH).map(h => E.ids[k][h.chunk]);
  const lists = [colbert(qs, 's'), colbert(qt, 't'), keyword(text, 's'), keyword(voiced, 't')];
  if (nums.length) lists.push(E.pid.flatMap((p, i) => nums.some(c => hasNum(c, valsOf(p)[E.kind[i]])) ? [i] : []));
  const fused = rrf(lists, { depth: FUSE_DEPTH }).slice(0, RERANK).map(h => h.chunk);
  const [rs, rt] = await Promise.all([
    rerank(qs, fused.filter(i => E.kind[i] === 's'), E.vectors, E.dim),
    rerank(qt, fused.filter(i => E.kind[i] === 't'), E.vectors, E.dim),
  ]);
  if (mine !== seq) return null;   // a newer search started meanwhile
  // documents to paragraphs, best first; each paragraph remembers which of its documents ranked best
  const rank = new Map(), via = new Map();
  for (const h of [...rs, ...rt].sort((a, b) => b.score - a.score)) {
    const p = E.pid[h.chunk];
    if (rank.has(p)) continue;
    rank.set(p, rank.size); via.set(p, E.kind[h.chunk]);
    if (rank.size === TOP) break;
  }
  const secs = new Map();
  for (const [p, r] of rank) if (!secs.has(PSEC[p])) secs.set(PSEC[p], r);
  return { text, terms, nums, rank, via, secs, ms: performance.now() - t0 };
}

// the text with the query's words and numbers marked
export function mark(text, q) {
  if (!q) return esc(text);
  const spans = [];
  for (const t of q.terms) for (const m of text.matchAll(new RegExp(t.re.source, 'gi'))) spans.push([m.index, m.index + m[0].length]);
  if (q.nums.length) for (const m of text.matchAll(NUM)) {
    const src = values(m[0]);
    if (q.nums.some(c => hasNum(c, src))) spans.push([m.index, m.index + m[0].length]);
  }
  spans.sort((a, b) => a[0] - b[0]);
  let out = '', at = 0;
  for (const [a, b] of spans) { if (a < at) continue; out += esc(text.slice(at, a)) + '<mark>' + esc(text.slice(a, b)) + '</mark>'; at = b; }
  return out + esc(text.slice(at));
}
// a few dozen words of the filing around its first match
export function snippet(text, q) {
  const firsts = q.terms.map(t => text.search(t.re)).filter(i => i >= 0);
  if (q.nums.length) for (const m of text.matchAll(NUM)) if (q.nums.some(c => hasNum(c, values(m[0])))) { firsts.push(m.index); break; }
  const at = firsts.length ? Math.min(...firsts) : 0;
  const a = Math.max(0, text.lastIndexOf(' ', Math.max(0, at - 90)));
  const b = text.indexOf(' ', at + 160);
  return (a > 0 ? '… ' : '') + mark(text.slice(a, b < 0 ? undefined : b).trim(), q) + (b < 0 ? '' : ' …');
}
