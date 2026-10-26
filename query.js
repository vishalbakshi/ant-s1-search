// The S-1 calls the company "we", "our" and "us"; "Anthropic" by name appears mostly on the cover,
// in the mission statement and in the founders' letter. A query that names Anthropic would match
// those pages, so the name is rewritten in the filing's own voice before encoding.
const AUX = /\b(?:is|was|are|were|does|did|do|has|have|had|will|would|can|could|should|may|might|must|how|why|when|where)\s+$/i;
const VERB = /^\s+(?:is|was|has|had|does|did|will|would|can|could|should|may|might|must|plans?|expects?|makes?|earns?|uses?|faces?|owns?|pays?|sells?|spends?|gets?|needs?|wants?|competes?)\b/i;

const PREPOSITION = /^\s+(?:after|before|during|in|on|at|for|with|to|from|by|of|over|and|or|but|if|when|than|as)\b/i;

export function ourVoice(query) {
  return query
    .replace(/\bAnthropic,?\s+PBC\b/gi, "Anthropic")
    .replace(/\bAnthropic(['’]s)?\b/gi, (match, possessive, at, text) => {
      const before = text.slice(0, at);
      const after = text.slice(at + match.length);
      if (possessive) return "our";
      if (AUX.test(before) || VERB.test(after)) return "we"; // the subject of a verb
      if (/^\s+[a-z]/i.test(after) && !PREPOSITION.test(after)) return "our"; // before a noun: "Anthropic agents"
      return "us"; // an object: "who could sue Anthropic?"
    });
}
