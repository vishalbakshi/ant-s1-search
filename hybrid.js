// Hybrid search: fuse the ColBERT and BM25 rankings with reciprocal rank fusion (RRF), then
// rerank the fused shortlist with exact MaxSim on the chunks' half-precision vectors.

// Fuse ranked lists of chunk ids: each list adds 1 / (k + rank) for its top `depth` chunks.
// Returns [{ chunk, score }], best first.
export function rrf(lists, { k = 60, depth = 100 } = {}) {
  const scores = new Map();
  for (const list of lists) {
    list.slice(0, depth).forEach((chunk, r) => scores.set(chunk, (scores.get(chunk) ?? 0) + 1 / (k + r + 1)));
  }
  return [...scores].sort((a, b) => b[1] - a[1] || a[0] - b[0]).map(([chunk, score]) => ({ chunk, score }));
}

// Token vectors of chunks, read from a float16 file (as written by scripts/build_index.mjs)
// with range requests, so only the rows needed are downloaded. Runs of consecutive chunks share
// one request, and rows already fetched are kept. Returns vectors(chunks) -> Map chunk -> Float32Array.
export function vectorStore(url, offsets, dim, fetchRange = fetch) {
  const cache = new Map();
  let whole = null; // the entire file, if the server ignored a range request

  async function fetchRun(first, last) {
    const from = offsets[first] * dim * 2;
    const to = offsets[last + 1] * dim * 2;
    let bytes;
    if (whole) {
      bytes = whole.slice(from, to);
    } else {
      const res = await fetchRange(url, { headers: { Range: `bytes=${from}-${to - 1}` } });
      if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
      const body = await res.arrayBuffer();
      if (res.status === 206) {
        bytes = body;
      } else {
        whole = body; // no range support: the server sent everything
        bytes = whole.slice(from, to);
      }
    }
    const rows = new Float16Array(bytes);
    for (let c = first; c <= last; c++) {
      const start = (offsets[c] - offsets[first]) * dim;
      cache.set(c, new Float32Array(rows.subarray(start, start + (offsets[c + 1] - offsets[c]) * dim)));
    }
  }

  return async function vectors(chunks) {
    const missing = [...new Set(chunks)].filter((c) => !cache.has(c)).sort((a, b) => a - b);
    const runs = [];
    for (const c of missing) {
      if (runs.length && runs.at(-1)[1] === c - 1) runs.at(-1)[1] = c;
      else runs.push([c, c]);
    }
    await Promise.all(runs.map(([first, last]) => fetchRun(first, last)));
    return new Map(chunks.map((c) => [c, cache.get(c)]));
  };
}

// Exact MaxSim of one chunk: for each query vector, its best dot product with the chunk's
// token vectors (`rows`, `dim` numbers each), summed.
export function maxsim(query, rows, dim) {
  let total = 0;
  for (const q of query) {
    let best = -Infinity;
    for (let at = 0; at < rows.length; at += dim) {
      let dot = 0;
      for (let d = 0; d < dim; d++) dot += q[d] * rows[at + d];
      if (dot > best) best = dot;
    }
    total += best;
  }
  return total;
}

// Rerank chunks by exact MaxSim against the query's vectors: [{ chunk, score }], best first.
export async function rerank(query, chunks, vectors, dim) {
  const rows = await vectors(chunks);
  return chunks
    .map((chunk) => ({ chunk, score: maxsim(query, rows.get(chunk), dim) }))
    .sort((a, b) => b.score - a.score || a.chunk - b.chunk);
}
