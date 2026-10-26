// Score every chunk against a query: exact MaxSim with float query vectors
// against binary (sign-bit) document vectors, as packed in data/index.bin.
//
// Each document number is +1 or -1, so a query vector q dotted with a document vector is
//   2 * (sum of q where the document bit is 1) - (sum of all of q).
// The first part is looked up a byte at a time: for every query vector and byte position,
// a 256-entry table holds the sum of q over each possible pattern of 8 bits.

// Tables for a query: Float32Array of query vectors, `dim` numbers each.
export function lookupTables(query, dim) {
  const nq = query.length / dim;
  const bytes = dim / 8;
  const lut = new Float32Array(nq * bytes * 256);
  const totals = new Float64Array(nq);
  for (let i = 0; i < nq; i++) {
    for (let b = 0; b < bytes; b++) {
      const q = query.subarray(i * dim + b * 8, i * dim + b * 8 + 8);
      const base = (i * bytes + b) * 256;
      for (let v = 0; v < 256; v++) {
        let sum = 0;
        for (let k = 0; k < 8; k++) if (v & (0x80 >> k)) sum += q[k]; // most significant bit first
        lut[base + v] = sum;
      }
    }
    for (let d = 0; d < dim; d++) totals[i] += query[i * dim + d];
  }
  return { lut, totals, nq, bytes };
}

// For each query vector, fills `best` with its highest table sum over token vectors from..to.
function bestSums({ lut, nq, bytes }, bits, from, to, best) {
  best.fill(-Infinity);
  for (let t = from; t < to; t++) {
    const at = t * bytes;
    for (let i = 0; i < nq; i++) {
      const base = i * bytes * 256;
      let sum = 0;
      for (let b = 0; b < bytes; b++) sum += lut[base + b * 256 + bits[at + b]];
      if (sum > best[i]) best[i] = sum;
    }
  }
}

// MaxSim score of every chunk. `bits` holds each token vector's sign bits back to back;
// chunk p's token vectors are offsets[p] up to offsets[p + 1].
export function scoreChunks(tables, bits, offsets) {
  const n = offsets.length - 1;
  const scores = new Float64Array(n);
  const best = new Float64Array(tables.nq);
  for (let p = 0; p < n; p++) {
    bestSums(tables, bits, offsets[p], offsets[p + 1], best);
    let score = 0;
    for (let i = 0; i < tables.nq; i++) score += 2 * best[i] - tables.totals[i];
    scores[p] = score;
  }
  return scores;
}

// Each query vector's part of one chunk's score: its best match anywhere in the chunk.
// They add up to the chunk's score from scoreChunks.
export function tokenScores(tables, bits, offsets, chunk) {
  const best = new Float64Array(tables.nq);
  bestSums(tables, bits, offsets[chunk], offsets[chunk + 1], best);
  return best.map((sum, i) => 2 * sum - tables.totals[i]);
}

// The `k` highest-scoring chunks: [{ chunk, score }].
export function topChunks(scores, k) {
  const order = Array.from(scores.keys()).sort((a, b) => scores[b] - scores[a] || a - b);
  return order.slice(0, k).map((chunk) => ({ chunk, score: scores[chunk] }));
}
