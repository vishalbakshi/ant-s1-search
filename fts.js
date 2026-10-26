// Full-text (keyword) search over the chunks with BM25, built in the browser from docs.json.
// Words are lowercased, common English words are dropped, and simple plurals are folded
// ("risks" matches "risk"), so only the query's content words count.

const STOPWORDS = new Set(
  ("a about above after again against all am an and any are as at be because been before being below between both but by can " +
    "could did do does doing down during each few for from further had has have having he her here hers him his how i if in " +
    "into is it its itself just me more most my no nor not now of off on once only or other our ours out over own same she " +
    "should so some such than that the their theirs them then there these they this those through to too under until up us " +
    "very was we were what when where which while who whom why will with would you your yours").split(" "),
);

// Content words of a text, in order.
export function terms(text) {
  return (text.toLowerCase().match(/[a-z0-9]+(?:[.'’][a-z0-9]+)*/g) ?? [])
    .map((w) => w.replace(/['’]s$/, ""))
    .filter((w) => !STOPWORDS.has(w))
    .map((w) => (w.length > 4 && w.endsWith("ies") ? `${w.slice(0, -3)}y` : w.length > 3 && /[^s]s$/.test(w) ? w.slice(0, -1) : w));
}

// Index texts for BM25; returns search(query, k) -> [{ chunk, score }], best first.
export function bm25(texts, { k1 = 1.2, b = 0.75 } = {}) {
  const postings = new Map(); // term -> [[doc, count], ...]
  const lengths = new Float64Array(texts.length);
  texts.forEach((text, doc) => {
    const counts = new Map();
    for (const t of terms(text)) counts.set(t, (counts.get(t) ?? 0) + 1);
    lengths[doc] = [...counts.values()].reduce((a, c) => a + c, 0);
    for (const [t, c] of counts) {
      if (!postings.has(t)) postings.set(t, []);
      postings.get(t).push([doc, c]);
    }
  });
  const avg = lengths.reduce((a, l) => a + l, 0) / texts.length;

  return function search(query, k) {
    const scores = new Float64Array(texts.length);
    for (const t of new Set(terms(query))) {
      const list = postings.get(t);
      if (!list) continue;
      const idf = Math.log(1 + (texts.length - list.length + 0.5) / (list.length + 0.5));
      for (const [doc, c] of list) scores[doc] += (idf * c * (k1 + 1)) / (c + k1 * (1 - b + (b * lengths[doc]) / avg));
    }
    return Array.from(scores.keys())
      .filter((doc) => scores[doc] > 0)
      .sort((a, c) => scores[c] - scores[a] || a - c)
      .slice(0, k)
      .map((chunk) => ({ chunk, score: scores[chunk] }));
  };
}
