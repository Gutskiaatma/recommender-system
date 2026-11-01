// Pure formatting helpers (unit-tested in tests/format.test.js).

export function formatPrice(price) {
  if (price === null || price === undefined || Number.isNaN(price)) return "Price n/a";
  return `$${price.toFixed(price >= 100 ? 0 : 2)}`;
}

export function pct(value, digits = 0) {
  if (value === null || value === undefined) return "–";
  return `${(value * 100).toFixed(digits)}%`;
}

export function score100(value) {
  if (value === null || value === undefined) return "–";
  return Math.round(value * 100).toString();
}

export function compactNumber(n) {
  if (n === null || n === undefined) return "–";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 10_000) return `${Math.round(n / 1000)}k`;
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return String(n);
}

export function shortTitle(title, max = 70) {
  if (!title) return "";
  return title.length > max ? `${title.slice(0, max - 1).trimEnd()}…` : title;
}

// Split text into parts so a matched aspect term can be highlighted safely
// (no innerHTML). Returns [{text, match}].
export function highlightParts(text, term) {
  if (!term) return [{ text, match: false }];
  const escaped = term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/\s+/g, "\\s+");
  // With a capturing group, split() puts matches at odd indices.
  const parts = text.split(new RegExp(`(${escaped})`, "i"));
  return parts
    .map((p, i) => ({ text: p, match: i % 2 === 1 }))
    .filter((p) => p.text !== "");
}

export const METHOD_LABELS = {
  hybrid: "Review-aware",
  semantic: "Semantic only",
  tfidf: "TF-IDF baseline",
};
